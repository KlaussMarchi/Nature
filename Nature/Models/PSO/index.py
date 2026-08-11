import numpy as np
from Processing.Problem.index import Problem
from Processing.Randomizer.index import Randomizer
from Processing.Pool.index import Pool
from Processing.Stopper.index import Stopper
from Processing.Recorder.index import Recorder
from Processing.Memory.index import Memory
from Processing.Portrait.index import Portrait


# MODELO DE ENXAME DE PARTÍCULAS — CRIADO PELO NatureSelector COM name='pso'. MONO-OBJETIVO.
class PSO:
    # Kennedy & Eberhart (1995) com inércia em decaimento linear (Shi & Eberhart, 1998) e coeficientes de
    # aceleração variáveis no tempo (TVAC, Ratnaweera/Halgamuge/Watson 2004): o cognitivo cai de 2,5 a 0,5
    # e o social sobe de 0,5 a 2,5. A partícula começa explorando em torno da própria memória e termina
    # convergindo para o líder, sem que ninguém tenha de reajustar constantes ao trocar de sistema.
    VMAX      = 0.5
    NEIGHBORS = 2

    def __init__(self, objective, variables, maximize=True, particles=50, iterations=200,
                 inertia=(0.9, 0.4), cognitive=(2.5, 0.5), social=(0.5, 2.5), topology='global',
                 patience=None, constraints=None, seed=None, memory=None, workers=1, backend='thread', verbose=True):
        if particles < 2:
            raise ValueError('particles deve ser >= 2.')
        if topology not in ('global', 'ring'):
            raise ValueError("topology deve ser 'global' ou 'ring'.")
        self.problem = Problem(objective, variables, maximize, constraints, backend)

        self.particles  = int(particles)
        self.iterations = int(iterations)
        self.inertia    = self.pair(inertia)
        self.cognitive  = self.pair(cognitive)
        self.social     = self.pair(social)
        self.topology   = topology
        self.patience   = patience
        self.seed       = seed
        self.memoryPath = memory
        self.workers    = workers
        self.backend    = backend
        self.verbose    = verbose

    def setup(self):
        self.vmax  = self.VMAX * (self.problem.up - self.problem.low)
        offsets    = np.arange(-self.NEIGHBORS, self.NEIGHBORS + 1)
        self.ring  = (np.arange(self.particles)[:, None] + offsets) % self.particles

    def update(self):
        self.setup()
        rand          = Randomizer(self.seed)
        rng           = rand.reset()
        self.recorder = Recorder(self.problem.genes, self.iterations, self.verbose, 'Enxame PSO', 'it')
        self.memory   = Memory(self.memoryPath, 'pso', self.config(), self.problem, self.recorder)
        self.stopped  = self.iterations
        stop          = Stopper(self.patience)
        pool          = Pool(self.workers, self.backend, self.seed)

        mapFunc = pool.start()
        try:
            state = self.memory.start()  # o estado do PSO inclui posição, velocidade e pbest (onde vive o gbest)
            if state and state[0]['X'].shape[0] != self.particles:
                state = None  # particles mudou: nova campanha (best.json preserva o melhor)
            if state is None:
                X   = rng.uniform(self.problem.low, self.problem.up, (self.particles, self.problem.nVars))
                raw = self.problem.score(X, mapFunc)
                V   = rng.uniform(-self.vmax, self.vmax, X.shape)
                pbest, pbestRaw, it0 = X.copy(), raw.copy(), 0
            else:
                arrays, meta = state
                X, raw, V, pbest, pbestRaw = arrays['X'], arrays['raw'], arrays['V'], arrays['pbest'], arrays['pbestRaw']
                it0 = meta['done']
                rand.set(meta['rng'])
            origin  = self.memory.origin(it0)
            wf      = raw[:, 0] * self.problem.weight
            pbestWf = pbestRaw[:, 0] * self.problem.weight
            self.recorder.span = self.stopped = origin + self.iterations
            if state is None:
                self.recorder.record(it0, self.particles, raw)  # retomada não re-grava: o histórico salvo já
                self.recorder.sample(X, raw)                    # termina nesta iteração, com os valores originais
            if state and not origin:
                stop.set(state[1].get('stop'))  # retomada: a paciência continua de onde parou (extensão reinicia)
            stop.check(float(np.max(pbestWf)))
            bar = self.recorder.bar(it0 + 1)

            for it in bar:
                w, c1, c2 = (self.ramp(p, it, origin) for p in (self.inertia, self.cognitive, self.social))
                r1, r2 = rng.random((2,) + X.shape)
                V   = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (self.leaders(pbest, pbestWf) - X)
                V   = np.clip(V, -self.vmax, self.vmax)
                X   = X + V
                out = (X < self.problem.low) | (X > self.problem.up)
                X   = np.clip(X, self.problem.low, self.problem.up)
                V[out] = 0.0  # fronteira absorvente: evita partículas coladas no limite

                raw    = self.problem.score(X, mapFunc)
                wf     = raw[:, 0] * self.problem.weight
                better = wf > pbestWf
                pbest[better], pbestRaw[better], pbestWf[better] = X[better], raw[better], wf[better]

                self.recorder.record(it, self.particles, raw)
                self.recorder.sample(X, raw)
                self.memory.save({'X': X, 'raw': raw, 'V': V, 'pbest': pbest, 'pbestRaw': pbestRaw}, {'done': it, 'rng': rand.get(), 'stop': stop.get()})
                if self.verbose:
                    bar.set_postfix(best=f'{pbestRaw[np.argmax(pbestWf)][0]:.6g}')

                if stop.check(float(np.max(pbestWf))):
                    self.stopped = it
                    if self.verbose:
                        bar.set_description(f'Early stopping @ it {it}')
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

    # LIMITES DE PARTIDA E CHEGADA DE UM COEFICIENTE: ESCALAR VIRA CONSTANTE, PAR VIRA RAMPA LINEAR
    def pair(self, value):
        return tuple(map(float, value)) if isinstance(value, (list, tuple)) else (float(value), float(value))

    # A FASE É O PROGRESSO DENTRO DO CICLO, NUNCA DO TOTAL — É ISSO QUE FAZ A RAMPA REINICIAR NUMA EXTENSÃO
    def ramp(self, pair, it, origin):
        return pair[0] + (pair[1] - pair[0]) * (it - origin - 1) / max(self.iterations - 1, 1)

    def config(self):
        return {'particles': self.particles, 'iterations': self.iterations, 'inertia': list(self.inertia),
                'cognitive': list(self.cognitive), 'social': list(self.social), 'topology': self.topology, 'patience': self.patience}

    def leaders(self, pbest, pbestWf):
        if self.topology == 'global':
            return pbest[np.argmax(pbestWf)][None, :]
        return pbest[self.ring[np.arange(self.particles), np.argmax(pbestWf[self.ring], axis=1)]]

    # O RETRATO DESTE RUN — O NatureSelector PENDURA plotMetrics/plotGraph/plotVariables NELE
    def portrait(self):
        if not hasattr(self, 'recorder'):
            # Sem update() nesta sessão: reconstrói o retrato só com o que a pasta de memory guardou.
            self.recorder = Recorder(self.problem.genes, self.iterations, False, 'Enxame PSO', 'it')
            self.memory   = Memory(self.memoryPath, 'pso', self.config(), self.problem, self.recorder)
            if self.memory.start() is None:
                raise RuntimeError('Execute update() antes de plotar, ou aponte memory= para uma pasta com campanha salva.')
            self.best, self.bestScore = self.memory.best, self.memory.score
        return Portrait(self.problem, self.recorder, self.best, self.bestScore, 'Iteração')
