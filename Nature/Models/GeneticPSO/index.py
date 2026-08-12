import numpy as np
from deap import tools
from ...Processing.Problem.index import Problem
from ...Processing.Randomizer.index import Randomizer
from ...Processing.Pool.index import Pool
from ...Processing.Stopper.index import Stopper
from ...Processing.Recorder.index import Recorder
from ...Processing.Memory.index import Memory
from ...Processing.Portrait.index import Portrait


# MODELO HÍBRIDO AG + PSO (METADE MELHOR SE REPRODUZ, METADE PIOR VOA) — CRIADO PELO NatureSelector COM
# name='genetic_pso'. MONO-OBJETIVO.
class GeneticPSO:
    # Kao & Zahara (2008): memória pbest/gbest compartilhada entre as duas metades. A metade que voa usa os
    # mesmos coeficientes variáveis no tempo do PSO (TVAC, Ratnaweera 2004), com os limites recomendados já
    # como padrão; um escalar no lugar do par devolve a constante fixa do PSO clássico.
    CX_ETA  = 15.0
    MUT_ETA = 20.0
    VMAX    = 0.5

    def __init__(self, objective, variables, maximize=True, population=100, generations=200, crossover=0.9,
                 mutation=0.3, inertia=(0.9, 0.4), cognitive=(2.5, 0.5), social=(0.5, 2.5),
                 patience=None, target=None, constraints=None, seed=None, memory=None, workers=1, backend='thread', verbose=True):
        if population < 4:
            raise ValueError('population deve ser >= 4 (metade AG, metade PSO).')
        self.problem = Problem(objective, variables, maximize, constraints, backend)

        self.population  = int(population)
        self.generations = int(generations)
        self.crossover   = float(crossover)
        self.mutation    = float(mutation)
        self.inertia     = self.pair(inertia)
        self.cognitive   = self.pair(cognitive)
        self.social      = self.pair(social)
        self.patience    = patience
        self.target      = target
        self.seed        = seed
        self.memoryPath  = memory
        self.workers     = workers
        self.backend     = backend
        self.verbose     = verbose

    def setup(self):
        self.vmax = self.VMAX * (self.problem.up - self.problem.low)

    def update(self):
        self.setup()
        rand          = Randomizer(self.seed)
        rng           = rand.reset()
        self.recorder = Recorder(self.problem.genes, self.generations, self.verbose, 'Híbrido AG+PSO')
        self.memory   = Memory(self.memoryPath, 'genetic_pso', self.config(), self.problem, self.recorder)
        self.stopped  = self.generations
        stop          = Stopper(self.patience, self.target, self.problem.weight)
        pool          = Pool(self.workers, self.backend, self.seed)

        mapFunc = pool.start()
        try:
            state = self.memory.start()  # estado: posição, velocidade e pbest (onde vive o gbest) + RNGs
            if state and state[0]['X'].shape[0] != self.population:
                state = None  # population mudou: nova campanha (best.json preserva o melhor)
            if state is None:
                X   = rng.uniform(self.problem.low, self.problem.up, (self.population, self.problem.nVars))
                raw = self.problem.score(X, mapFunc)
                V   = rng.uniform(-self.vmax, self.vmax, X.shape)
                pbest, pbestRaw, gen0 = X.copy(), raw.copy(), 0
            else:
                arrays, meta = state
                X, raw, V, pbest, pbestRaw = arrays['X'], arrays['raw'], arrays['V'], arrays['pbest'], arrays['pbestRaw']
                gen0 = meta['done']
                rand.set(meta['rng'])
            origin  = self.memory.origin(gen0)
            wf      = raw[:, 0] * self.problem.weight
            pbestWf = pbestRaw[:, 0] * self.problem.weight
            self.recorder.span = self.stopped = origin + self.generations
            if state is None:
                self.recorder.record(gen0, self.population, raw)  # retomada não re-grava: o histórico salvo já
                self.recorder.sample(X, raw)                      # termina nesta geração, com os valores originais
            if state and not origin:
                stop.set(state[1].get('stop'))  # retomada: a paciência continua de onde parou (extensão reinicia)
            stop.check(float(np.max(pbestWf)))
            half = self.population // 2
            bar  = self.recorder.bar(gen0 + 1)

            for gen in bar:
                phase = (gen - origin - 1) / max(self.generations - 1, 1)
                order = np.argsort(-wf)
                top, low = order[:half], order[half:]
                gbest = pbest[np.argmax(pbestWf)]

                X[top] = self.breed(X[top], wf[top], rng)
                X[top[0]] = gbest  # elitismo: o melhor conhecido nunca se perde
                X[low], V[low] = self.fly(X[low], V[low], pbest[low], gbest, phase, rng)

                raw    = self.problem.score(X, mapFunc)
                wf     = raw[:, 0] * self.problem.weight
                better = wf > pbestWf
                pbest[better], pbestRaw[better], pbestWf[better] = X[better], raw[better], wf[better]

                self.recorder.record(gen, self.population, raw)
                self.recorder.sample(X, raw)
                self.memory.save({'X': X, 'raw': raw, 'V': V, 'pbest': pbest, 'pbestRaw': pbestRaw}, {'done': gen, 'rng': rand.get(), 'stop': stop.get()})
                if self.verbose:
                    bar.set_postfix(best=f'{pbestRaw[np.argmax(pbestWf)][0]:.6g}')

                if stop.check(float(np.max(pbestWf))):
                    self.stopped = gen
                    if self.verbose:
                        bar.set_description(f'Early stopping @ gen {gen}')
                    break
        finally:
            pool.stop()

        winner         = int(np.argmax(pbestWf))
        self.best      = self.problem.decode(pbest[winner])
        self.bestScore = float(pbestRaw[winner][0])
        self.best, self.bestScore = self.memory.commit(self.best, self.bestScore, pbest[winner], self.stopped,
                                                        {'X': X, 'raw': raw, 'V': V, 'pbest': pbest, 'pbestRaw': pbestRaw},
                                                        {'done': self.stopped, 'rng': rand.get(), 'stop': stop.get()})
        return self.best, self.bestScore

    def config(self):
        return {'population': self.population, 'generations': self.generations, 'crossover': self.crossover,
                'mutation': self.mutation, 'inertia': list(self.inertia), 'cognitive': list(self.cognitive),
                'social': list(self.social), 'patience': self.patience, 'target': self.target}

    def breed(self, elite, eliteWf, rng):
        low, up = list(self.problem.low), list(self.problem.up)

        def pick():
            a, b = rng.integers(len(elite), size=2)
            return list(elite[a if eliteWf[a] >= eliteWf[b] else b])

        children = []
        while len(children) < len(elite):
            c1, c2 = pick(), pick()
            if rng.random() < self.crossover:
                tools.cxSimulatedBinaryBounded(c1, c2, self.CX_ETA, low, up)
            for c in (c1, c2):
                if rng.random() < self.mutation:
                    tools.mutPolynomialBounded(c, self.MUT_ETA, low, up, 1.0 / self.problem.nVars)
            children += [c1, c2]
        return np.array(children[:len(elite)])

    # LIMITES DE PARTIDA E CHEGADA DE UM COEFICIENTE: ESCALAR VIRA CONSTANTE, PAR VIRA RAMPA LINEAR
    def pair(self, value):
        return tuple(map(float, value)) if isinstance(value, (list, tuple)) else (float(value), float(value))

    # phase É O PROGRESSO DENTRO DO CICLO (0 A 1): OS TRÊS COEFICIENTES SÃO INTERPOLADOS ENTRE SEUS LIMITES
    def fly(self, X, V, pbest, gbest, phase, rng):
        w, c1, c2 = (a + (b - a) * phase for a, b in (self.inertia, self.cognitive, self.social))
        r1, r2 = rng.random((2,) + X.shape)
        V   = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V   = np.clip(V, -self.vmax, self.vmax)
        X   = X + V
        out = (X < self.problem.low) | (X > self.problem.up)
        X   = np.clip(X, self.problem.low, self.problem.up)
        V[out] = 0.0
        return X, V

    # O RETRATO DESTE RUN — O NatureSelector PENDURA plotMetrics/plotGraph/plotVariables NELE
    def portrait(self):
        if not hasattr(self, 'recorder'):
            # Sem update() nesta sessão: reconstrói o retrato só com o que a pasta de memory guardou.
            self.recorder = Recorder(self.problem.genes, self.generations, False, 'Híbrido AG+PSO')
            self.memory   = Memory(self.memoryPath, 'genetic_pso', self.config(), self.problem, self.recorder)
            if self.memory.start() is None:
                raise RuntimeError('Execute update() antes de plotar, ou aponte memory= para uma pasta com campanha salva.')
            self.best, self.bestScore = self.memory.best, self.memory.score
        return Portrait(self.problem, self.recorder, self.best, self.bestScore)
