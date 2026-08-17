import numpy as np
from ...Processing.Engine.index import Engine


# HÍBRIDO AG+PSO (Kao & Zahara 2008): A METADE MELHOR SE REPRODUZ, A PIOR VOA, E AS DUAS COMPARTILHAM A
# MEMÓRIA pbest/gbest. UM ESCALAR NO LUGAR DO PAR DEVOLVE A CONSTANTE FIXA DO PSO CLÁSSICO.
class GeneticPSO(Engine):
    MODEL   = 'genetic_pso'
    DESC    = 'Híbrido AG+PSO'
    CX_ETA  = 15.0
    MUT_ETA = 20.0
    VMAX    = 0.5

    def __init__(self, objective, variables, maximize=True, population=100, generations=200, crossover=0.9,
                 mutation=0.3, inertia=(0.9, 0.4), cognitive=(2.5, 0.5), social=(0.5, 2.5),
                 patience=None, target=None, constraints=None, seed=None, memory=None, workers=1, backend='thread', verbose=True):
        if population < 4:
            raise ValueError('population deve ser >= 4 (metade AG, metade PSO).')
        super().__init__(objective, variables, maximize, constraints, backend, workers, seed, memory, verbose, patience, target)
        self.size      = int(population)
        self.span      = int(generations)
        self.crossover = float(crossover)
        self.mutation  = float(mutation)
        self.inertia   = self.pair(inertia)
        self.cognitive = self.pair(cognitive)
        self.social    = self.pair(social)

    def setup(self):
        self.vmax = self.VMAX * (self.problem.up - self.problem.low)

    def update(self):
        self.setup()
        rng = self.open()

        try:
            arrays = self.resume('X')
            if arrays is None:
                X   = rng.uniform(self.problem.low, self.problem.up, (self.size, self.problem.nVars))
                raw = self.score(X)
                V   = rng.uniform(-self.vmax, self.vmax, X.shape)
                pbest, pbestRaw, gen0 = X.copy(), raw.copy(), 0
            else:
                X, raw, V, pbest, pbestRaw = (arrays[k] for k in ('X', 'raw', 'V', 'pbest', 'pbestRaw'))
                gen0 = self.meta['done']
            self.pack = lambda: {'X': X, 'raw': raw, 'V': V, 'pbest': pbest, 'pbestRaw': pbestRaw}
            wf        = raw[:, 0] * self.problem.weight
            pbestWf   = pbestRaw[:, 0] * self.problem.weight
            self.recorder.span = self.stopped = self.begin(gen0, X, raw, self.size, float(np.max(pbestWf)), self.span)
            half = self.size // 2

            for gen in self.track(gen0 + 1):
                order    = np.argsort(-wf)
                top, low = order[:half], order[half:]
                gbest    = pbest[np.argmax(pbestWf)]

                X[top] = self.breed(X[top], wf[top], rng)
                X[top[0]] = gbest  # elitismo: o melhor conhecido nunca se perde
                X[low], V[low] = self.fly(X[low], V[low], pbest[low], gbest, self.phase(gen), rng)

                raw    = self.score(X)
                wf     = raw[:, 0] * self.problem.weight
                better = wf > pbestWf
                pbest[better], pbestRaw[better], pbestWf[better] = X[better], raw[better], wf[better]

                if self.tick(gen, self.size, X, raw, float(np.max(pbestWf))):
                    break
        finally:
            self.pool.stop()

        winner = int(np.argmax(pbestWf))
        return self.finish(pbest[winner], pbestRaw[winner][0], self.stopped)

    def config(self):
        return {'population': self.size, 'generations': self.span, 'crossover': self.crossover,
                'mutation': self.mutation, 'inertia': list(self.inertia), 'cognitive': list(self.cognitive),
                'social': list(self.social), 'patience': self.patience, 'target': self.target}

    # TORNEIO BINÁRIO ESCOLHE OS PAIS, O PAR CRUZA COM CHANCE crossover E CADA FILHO MUTA COM CHANCE mutation
    def breed(self, elite, eliteWf, rng):
        n     = len(elite)
        pairs = -(-n // 2)
        duel  = rng.integers(n, size=(2, 2 * pairs))
        C     = elite[np.where(eliteWf[duel[0]] >= eliteWf[duel[1]], duel[0], duel[1])]
        c1, c2 = C[:pairs], C[pairs:]

        cross = rng.random(pairs) < self.crossover
        if cross.any():
            c1[cross], c2[cross] = self.sbx(c1[cross], c2[cross], self.CX_ETA, rng)
        children = np.stack([c1, c2], axis=1).reshape(2 * pairs, -1)[:n]
        hit      = rng.random(n) < self.mutation
        if hit.any():
            children[hit] = self.polynomial(children[hit], self.MUT_ETA, rng)
        return children

    # OS TRÊS COEFICIENTES INTERPOLADOS ENTRE SEUS LIMITES PELO PROGRESSO DO CICLO
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
