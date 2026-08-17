import numpy as np
from ...Processing.Engine.index import Engine


# ALGORITMO GENÉTICO (μ+λ): SBX + MUTAÇÃO NÃO-UNIFORME, VARIAÇÃO varOr (CRUZAR, MUTAR OU COPIAR O PAI) E
# SOBREVIVÊNCIA ELITISTA DOS μ MELHORES ENTRE PAIS E FILHOS, TUDO SOBRE O LOTE EM VEZ DE GENE A GENE.
class GeneticOptimizer(Engine):
    MODEL  = 'genetic'
    DESC   = 'Evoluindo AG'
    CX_ETA = 15.0

    def __init__(self, objective, variables, maximize=True, population=100, generations=200, crossover=0.7,
                 mutation=0.2, patience=None, target=None, constraints=None, seed=None, memory=None, workers=1, backend='thread', verbose=True):
        if population < 2:
            raise ValueError('population deve ser >= 2.')
        if crossover + mutation > 1.0 + 1e-9:
            raise ValueError(f'crossover + mutation deve ser <= 1 (varOr); recebido {crossover + mutation}.')
        super().__init__(objective, variables, maximize, constraints, backend, workers, seed, memory, verbose, patience, target)
        self.size      = int(population)
        self.span      = int(generations)
        self.crossover = float(crossover)
        self.mutation  = float(mutation)

    def update(self):
        rng = self.open()

        try:
            arrays = self.resume('genomes')
            if arrays is None:
                X    = self.spawn(rng)
                raw  = self.score(X)
                gen0 = 0
            else:
                X, raw, gen0 = arrays['genomes'], arrays['fitness'], self.meta['done']
            self.pack = lambda: {'genomes': X, 'fitness': raw}
            wf        = raw[:, 0] * self.problem.weight
            # nevals da geração zero é a população inteira; no laço, só os filhos que a variação alterou
            self.recorder.span = self.stopped = self.begin(gen0, X, raw, self.size, float(np.max(wf)), self.span)

            for gen in self.track(gen0 + 1):
                U, base, fresh = self.vary(X, self.phase(gen), rng)

                rawU = raw[base]   # quem só copiou o pai herda a aptidão dele e não é reavaliado
                if fresh.any():
                    rawU[fresh] = self.score(U[fresh])
                pool = np.vstack([X, U])
                pot  = np.vstack([raw, rawU])
                keep = np.argsort(-pot[:, 0] * self.problem.weight, kind='stable')[:self.size]   # estável: pai na frente no empate
                X, raw = pool[keep], pot[keep]
                wf = raw[:, 0] * self.problem.weight

                if self.tick(gen, int(fresh.sum()), X, raw, float(np.max(wf))):
                    break
        finally:
            self.pool.stop()

        winner = int(np.argmax(wf))
        return self.finish(X[winner], raw[winner][0], self.stopped)

    def config(self):
        return {'population': self.size, 'generations': self.span, 'crossover': self.crossover, 'mutation': self.mutation, 'patience': self.patience, 'target': self.target}

    # O GENE DISCRETO SORTEIA INTEIRO: ARREDONDAR UM UNIFORME DARIA METADE DA CHANCE ÀS PONTAS DO RANGE
    def spawn(self, rng):
        lo, up = self.problem.low, self.problem.up
        X      = rng.uniform(lo, up, (self.size, self.problem.nVars))
        step   = np.array([g['type'] != 'real' for g in self.problem.genes])
        if step.any():
            X[:, step] = np.floor(rng.uniform(lo[step], up[step] + 1.0, (self.size, int(step.sum()))))
        return X

    # DEVOLVE OS FILHOS, O PAI DE CADA UM E QUEM DE FATO MUDOU E PRECISA SER AVALIADO
    def vary(self, X, phase, rng):
        op   = rng.random(self.size)
        cx   = op < self.crossover
        mut  = ~cx & (op < self.crossover + self.mutation)
        base = rng.integers(self.size, size=self.size)
        U    = X[base]   # indexação avançada já copia

        if cx.any():
            mate  = self.pick(int(cx.sum()), base[cx], rng)   # o segundo pai é sempre outro
            U[cx] = self.sbx(U[cx], X[mate], self.CX_ETA, rng)[0]
        if mut.any():
            U[mut] = self.nonuniform(U[mut], phase, rng)
        return U, base, cx | mut
