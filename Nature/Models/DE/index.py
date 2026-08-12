import numpy as np
from ...Processing.Problem.index import Problem
from ...Processing.Randomizer.index import Randomizer
from ...Processing.Pool.index import Pool
from ...Processing.Stopper.index import Stopper
from ...Processing.Recorder.index import Recorder
from ...Processing.Memory.index import Memory
from ...Processing.Portrait.index import Portrait


# MODELO DE EVOLUÇÃO DIFERENCIAL CLÁSSICA — CRIADO PELO NatureSelector COM name='de'. MONO-OBJETIVO.
class DifferentialEvolution:
    # Storn & Price (1997): mutação v = base + F·Σ(xr - xs), crossover bin/exp e seleção gulosa 1-a-1.
    # A estratégia é escrita como 'base/diffs/cx'. O default é current-to-best/1/bin, e não o rand/1/bin
    # canônico do artigo, por medição: na suíte CEC com 5 sementes e 70k avaliações ele ganha em rank médio
    # tanto em D=10 (1,19 contra 3,38 do rand/1/exp) quanto em D=30 (1,63 contra 3,25), inclusive nas
    # multimodais — em bent_cigar D=10 a diferença é de 6,8 para 3,1e-23.
    BASES      = ('rand', 'best', 'current-to-best')
    CROSSOVERS = ('bin', 'exp')

    def __init__(self, objective, variables, maximize=True, population=50, generations=200, F=(0.5, 1.0),
                 CR=0.9, strategy='current-to-best/1/bin', patience=None, target=None, constraints=None, seed=None, memory=None,
                 workers=1, backend='thread', verbose=True):
        parts = strategy.split('/')
        if len(parts) != 3 or parts[0] not in self.BASES or parts[1] not in ('1', '2') or parts[2] not in self.CROSSOVERS:
            raise ValueError(f"strategy inválida '{strategy}'; use 'base/diffs/cx' com base {self.BASES}, diffs 1|2, cx {self.CROSSOVERS}.")
        self.base, self.diffs, self.cx = parts[0], int(parts[1]), parts[2]

        # rand precisa do vetor-base aleatório; best/current-to-best não. Cada diferença gasta 2 índices.
        needed = (self.base == 'rand') + 2 * self.diffs
        if population < needed + 1:
            raise ValueError(f"population deve ser >= {needed + 1} para a estratégia '{strategy}'.")

        self.F = (float(F), float(F)) if isinstance(F, (int, float)) else tuple(sorted(map(float, F)))
        if not 0.0 <= self.F[0] <= self.F[1] <= 2.0:
            raise ValueError('F deve estar em [0, 2] (escalar ou (min, max) para dither).')
        if not 0.0 <= CR <= 1.0:
            raise ValueError('CR deve estar em [0, 1].')

        self.problem = Problem(objective, variables, maximize, constraints, backend)

        self.population  = int(population)
        self.generations = int(generations)
        self.CR          = float(CR)
        self.strategy    = strategy
        self.patience    = patience
        self.target      = target
        self.seed        = seed
        self.memoryPath  = memory
        self.workers     = workers
        self.backend     = backend
        self.verbose     = verbose

    def update(self):
        rand          = Randomizer(self.seed)
        rng           = rand.reset()
        self.recorder = Recorder(self.problem.genes, self.generations, self.verbose, 'Evolução Diferencial')
        self.memory   = Memory(self.memoryPath, 'de', self.config(), self.problem, self.recorder)
        self.stopped  = self.generations
        stop          = Stopper(self.patience, self.target, self.problem.weight)
        pool          = Pool(self.workers, self.backend, self.seed)

        mapFunc = pool.start()
        try:
            state = self.memory.start()
            if state and state[0]['X'].shape[0] != self.population:
                state = None  # population mudou: nova campanha (best.json preserva o melhor)
            if state is None:
                X    = rng.uniform(self.problem.low, self.problem.up, (self.population, self.problem.nVars))
                raw  = self.problem.score(X, mapFunc)
                gen0 = 0
            else:
                arrays, meta = state
                X, raw, gen0 = arrays['X'], arrays['raw'], meta['done']
                rand.set(meta['rng'])
            origin = self.memory.origin(gen0)
            wf     = raw[:, 0] * self.problem.weight
            self.recorder.span = self.stopped = origin + self.generations
            if state is None:
                self.recorder.record(gen0, self.population, raw)  # retomada não re-grava: o histórico salvo já
                self.recorder.sample(X, raw)                      # termina nesta geração, com os valores originais
            if state and not origin:
                stop.set(state[1].get('stop'))  # retomada: a paciência continua de onde parou (extensão reinicia)
            stop.check(float(np.max(wf)))
            bar = self.recorder.bar(gen0 + 1)

            for gen in bar:
                f = rng.uniform(*self.F)  # dither: um F por geração amplia a robustez (Storn & Price)
                U = self.cross(X, self.donor(X, wf, f, rng), rng)
                U = np.clip(U, self.problem.low, self.problem.up)

                rawU = self.problem.score(U, mapFunc)
                wfU  = rawU[:, 0] * self.problem.weight
                win  = wfU >= wf  # aceitar empate move a população por platôs e regiões inviáveis
                X[win], raw[win], wf[win] = U[win], rawU[win], wfU[win]

                self.recorder.record(gen, self.population, raw)
                self.recorder.sample(X, raw)
                self.memory.save({'X': X, 'raw': raw}, {'done': gen, 'rng': rand.get(), 'stop': stop.get()})
                if self.verbose:
                    bar.set_postfix(best=f'{raw[np.argmax(wf)][0]:.6g}')

                if stop.check(float(np.max(wf))):
                    self.stopped = gen
                    if self.verbose:
                        bar.set_description(f'Early stopping @ gen {gen}')
                    break
        finally:
            pool.stop()

        winner         = int(np.argmax(wf))
        self.best      = self.problem.decode(X[winner])
        self.bestScore = float(raw[winner][0])
        self.best, self.bestScore = self.memory.commit(self.best, self.bestScore, X[winner], self.stopped,
                                                        {'X': X, 'raw': raw}, {'done': self.stopped, 'rng': rand.get(), 'stop': stop.get()})
        return self.best, self.bestScore

    def config(self):
        return {'population': self.population, 'generations': self.generations, 'F': list(self.F), 'CR': self.CR, 'strategy': self.strategy, 'patience': self.patience, 'target': self.target}

    def donor(self, X, wf, f, rng):
        best = X[np.argmax(wf)]
        if self.base == 'rand':
            r = self.samples(1 + 2 * self.diffs, rng)
            return X[r[:, 0]] + f * self.delta(X, r[:, 1:])
        if self.base == 'best':
            return best + f * self.delta(X, self.samples(2 * self.diffs, rng))
        return X + f * (best - X) + f * self.delta(X, self.samples(2 * self.diffs, rng))

    def samples(self, count, rng):
        # Sorteio por rejeição, O(n·count) de RAM. A versão anterior permutava uma matriz (n, n-1) por
        # geração: com população grande isso é O(n²) e domina memória e tempo (pop 8000 -> 1,5 GB de pico).
        rows = np.arange(self.population)
        r    = self.pick(self.population, rows, rng)[:, None]
        for _ in range(count - 1):
            new = self.pick(self.population, rows, rng)
            while True:
                clash = (r == new[:, None]).any(1)
                if not clash.any():
                    break
                new[clash] = self.pick(int(clash.sum()), rows[clash], rng)
            r = np.hstack([r, new[:, None]])
        return r

    def pick(self, size, avoid, rng):
        # Sorteia em [0, n-2] e pula o próprio índice: sai uniforme sobre os n-1 candidatos != avoid.
        r = rng.integers(self.population - 1, size=size)
        return r + (r >= avoid)

    def delta(self, X, pairs):
        return sum(X[pairs[:, 2 * k]] - X[pairs[:, 2 * k + 1]] for k in range(self.diffs))

    def cross(self, X, V, rng):
        n, d = X.shape
        if self.cx == 'bin':
            mask = rng.random((n, d)) < self.CR
            mask[np.arange(n), rng.integers(d, size=n)] = True
            return np.where(mask, V, X)
        # exp: bloco contíguo (com wrap) a partir de um gene aleatório enquanto rand < CR — garante 1 gene.
        streak  = np.cumprod(np.hstack([np.ones((n, 1)), rng.random((n, d - 1)) < self.CR]), axis=1).astype(bool)
        offsets = (rng.integers(d, size=n)[:, None] + np.arange(d)) % d
        mask    = np.zeros((n, d), bool)
        mask[np.repeat(np.arange(n), d), offsets.ravel()] = streak.ravel()
        return np.where(mask, V, X)

    # O RETRATO DESTE RUN — O NatureSelector PENDURA plotMetrics/plotGraph/plotVariables NELE
    def portrait(self):
        if not hasattr(self, 'recorder'):
            # Sem update() nesta sessão: reconstrói o retrato só com o que a pasta de memory guardou.
            self.recorder = Recorder(self.problem.genes, self.generations, False, 'Evolução Diferencial')
            self.memory   = Memory(self.memoryPath, 'de', self.config(), self.problem, self.recorder)
            if self.memory.start() is None:
                raise RuntimeError('Execute update() antes de plotar, ou aponte memory= para uma pasta com campanha salva.')
            self.best, self.bestScore = self.memory.best, self.memory.score
        return Portrait(self.problem, self.recorder, self.best, self.bestScore)
