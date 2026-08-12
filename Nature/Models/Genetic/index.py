import numpy as np
from deap import base, creator, tools, algorithms
from ...Processing.Problem.index import Problem
from ...Processing.Randomizer.index import Randomizer
from ...Processing.Pool.index import Pool
from ...Processing.Stopper.index import Stopper
from ...Processing.Recorder.index import Recorder
from ...Processing.Memory.index import Memory
from ...Processing.Portrait.index import Portrait


# MODELO DE ALGORITMO GENÉTICO (μ+λ SOBRE A DEAP) — CRIADO PELO NatureSelector COM name='genetic'.
class GeneticOptimizer:
    # Cruzamento SBX + mutação polinomial de eta adaptativo (refina conforme as gerações avançam).
    CX_ETA         = 15.0
    MUT_ETA        = 20.0
    MUT_ETA_GROWTH = 5.0

    def __init__(self, objective, variables, maximize=True, population=100, generations=200, crossover=0.7,
                 mutation=0.2, patience=None, target=None, constraints=None, seed=None, memory=None, workers=1, backend='thread', verbose=True):
        if population < 2:
            raise ValueError('population deve ser >= 2.')
        if crossover + mutation > 1.0 + 1e-9:
            raise ValueError(f'crossover + mutation deve ser <= 1 (varOr); recebido {crossover + mutation}.')
        self.problem     = Problem(objective, variables, maximize, constraints, backend)
        self.population  = int(population)
        self.generations = int(generations)
        self.crossover   = float(crossover)
        self.mutation    = float(mutation)
        self.patience    = patience
        self.target      = target
        self.seed        = seed
        self.memoryPath  = memory
        self.workers     = workers
        self.backend     = backend
        self.verbose     = verbose
        self.eta         = self.MUT_ETA

    def setup(self):
        # Classes dinâmicas da DEAP com nome fixo, recriadas a cada run
        # (delattr evita o warning de redefinição e acúmulo no módulo).
        for name in ('GAFitness', 'GAIndividual'):
            if hasattr(creator, name):
                delattr(creator, name)
        creator.create('GAFitness', base.Fitness, weights=(self.problem.weight,))
        creator.create('GAIndividual', list, fitness=creator.GAFitness)
        self.indCls = creator.GAIndividual

        self.toolbox = base.Toolbox()
        self.toolbox.register('individual', tools.initIterate, self.indCls, self.genome)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        # Genes int/cat/bool evoluem como reais e são discretizados no decode.
        self.toolbox.register('mate', tools.cxSimulatedBinaryBounded, eta=self.CX_ETA, low=list(self.problem.low), up=list(self.problem.up))
        self.toolbox.register('mutate', self.mutate)
        self.toolbox.register('select', tools.selBest)

    def update(self):
        self.setup()
        rand          = Randomizer(self.seed)  # capturado só p/ guardar o estado (a DEAP evolui pelo random global)
        rng           = rand.reset()
        self.recorder = Recorder(self.problem.genes, self.generations, self.verbose, 'Evoluindo AG')
        self.memory   = Memory(self.memoryPath, 'genetic', self.config(), self.problem, self.recorder)
        self.hof      = tools.HallOfFame(1)
        self.stopped  = self.generations
        stop          = Stopper(self.patience, self.target, self.problem.weight)
        pool          = Pool(self.workers, self.backend, self.seed)

        mapFunc = pool.start()
        try:
            state = self.memory.start()
            if state and state[0]['genomes'].shape[0] != self.population:
                state = None  # population mudou: nova campanha (best.json preserva o melhor)
            if state is None:
                pop  = self.toolbox.population(n=self.population)
                self.score(pop, mapFunc)
                gen0 = 0
            else:
                arrays, meta = state
                pop = self.toolbox.population(n=self.population)  # molde; genoma e fitness reais vêm do estado salvo
                for ind, g, fit in zip(pop, arrays['genomes'], arrays['fitness']):
                    ind[:] = list(map(float, g))
                    ind.fitness.values = tuple(map(float, fit))
                gen0 = meta['done']
                rand.set(meta['rng'])
            # Retomada não re-grava: o histórico salvo já termina nesta geração, com o nevals original (que no
            # AG conta só os filhos inválidos, não a população inteira). O record() vem antes do sample()
            # porque é dele que a nuvem tira a geração de cada ponto.
            if state is None:
                self.recorder.record(gen0, len(pop), [ind.fitness.values for ind in pop])
            self.hof.update(pop)
            if state is None:
                self.recorder.sample(np.array([list(ind) for ind in pop]), np.array([ind.fitness.values for ind in pop], float))
            origin = self.memory.origin(gen0)
            self.recorder.span = self.stopped = origin + self.generations
            if state and not origin:
                stop.set(state[1].get('stop'))  # retomada: a paciência continua de onde parou (extensão reinicia)
            stop.check(self.signal(pop))
            bar = self.recorder.bar(gen0 + 1)

            for gen in bar:
                # Eta da mutação cresce com as gerações, refinando o fim da busca.
                if self.generations > 1:
                    self.eta = self.MUT_ETA * (1.0 + (self.MUT_ETA_GROWTH - 1.0) * (gen - origin - 1) / (self.generations - 1))

                children = algorithms.varOr(pop, self.toolbox, self.population, self.crossover, self.mutation)
                invalid  = [ind for ind in children if not ind.fitness.valid]
                self.score(invalid, mapFunc)
                pop = self.toolbox.select(pop + children, self.population)
                self.recorder.record(gen, len(invalid), [ind.fitness.values for ind in pop])
                self.memory.save({'genomes': np.array([list(ind) for ind in pop]), 'fitness': np.array([ind.fitness.values for ind in pop], float)},
                                 {'done': gen, 'rng': rand.get(), 'stop': stop.get()})

                self.hof.update(pop)
                self.recorder.sample(np.array([list(ind) for ind in pop]), np.array([ind.fitness.values for ind in pop], float))
                if self.verbose:
                    bar.set_postfix(best=f'{self.hof[0].fitness.values[0]:.6g}')

                if stop.check(self.signal(pop)):
                    self.stopped = gen
                    if self.verbose:
                        bar.set_description(f'Early stopping @ gen {gen}')
                    break
        finally:
            pool.stop()

        winner         = self.hof[0]
        self.best      = self.problem.decode(winner)
        self.bestScore = winner.fitness.values[0]
        self.best, self.bestScore = self.memory.commit(self.best, self.bestScore, list(winner), self.stopped,
                                                        {'genomes': np.array([list(ind) for ind in pop]), 'fitness': np.array([ind.fitness.values for ind in pop], float)},
                                                        {'done': self.stopped, 'rng': rand.get(), 'stop': stop.get()})
        return self.best, self.bestScore

    def config(self):
        return {'population': self.population, 'generations': self.generations, 'crossover': self.crossover, 'mutation': self.mutation, 'patience': self.patience, 'target': self.target}

    def genome(self):
        return [np.random.uniform(g['low'], g['up']) if g['type'] == 'real'
                else float(np.random.randint(int(g['low']), int(g['up']) + 1))
                for g in self.problem.genes]

    def mutate(self, ind):
        return tools.mutPolynomialBounded(ind, self.eta, list(self.problem.low), list(self.problem.up), 1.0 / self.problem.nVars)

    def score(self, individuals, mapFunc):
        if not individuals:
            return
        if self.backend == 'vector':
            for ind, row in zip(individuals, self.problem.score(individuals, mapFunc)):
                ind.fitness.values = (float(row[0]),)
            return
        # Aos workers vai só o genoma (lista) e o Problem, nunca a classe Individual.
        genomes = [list(i) for i in individuals]
        for ind, fit in zip(individuals, list(mapFunc(self.problem.evaluate, genomes))):
            ind.fitness.values = fit

    def signal(self, pop):
        return max(ind.fitness.wvalues[0] for ind in pop)

    # O RETRATO DESTE RUN — O NatureSelector PENDURA plotMetrics/plotGraph/plotVariables NELE
    def portrait(self):
        if not hasattr(self, 'recorder'):
            # Sem update() nesta sessão: reconstrói o retrato só com o que a pasta de memory guardou.
            self.recorder = Recorder(self.problem.genes, self.generations, False, 'Evoluindo AG')
            self.memory   = Memory(self.memoryPath, 'genetic', self.config(), self.problem, self.recorder)
            if self.memory.start() is None:
                raise RuntimeError('Execute update() antes de plotar, ou aponte memory= para uma pasta com campanha salva.')
            self.best, self.bestScore = self.memory.best, self.memory.score
        return Portrait(self.problem, self.recorder, self.best, self.bestScore)
