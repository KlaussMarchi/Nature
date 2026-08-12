import numpy as np
from tqdm.auto import tqdm
from ...Processing.Problem.index import Problem
from ...Processing.Randomizer.index import Randomizer
from ...Processing.Pool.index import Pool
from ...Processing.Stopper.index import Stopper
from ...Processing.Recorder.index import Recorder
from ...Processing.Memory.index import Memory
from ...Processing.Portrait.index import Portrait


# MODELO DE EVOLUÇÃO DIFERENCIAL AUTO-ADAPTATIVA (JADE / SHADE / L-SHADE) — CRIADO PELO NatureSelector COM
# name='adaptive_de'. MONO-OBJETIVO E O MAIS FORTE EM ALTA DIMENSÃO.
class AdaptiveDE:
    # DE auto-adaptativa da família current-to-pbest/1/bin com arquivo externo (Zhang & Sanderson 2009;
    # Tanabe & Fukunaga 2013/2014). variant escolhe como F/CR se adaptam e se a população encolhe:
    #   jade   — média única μ_F/μ_CR atualizada com taxa C
    #   shade  — memória de sucesso de tamanho H (M_F/M_CR)
    #   lshade — shade + LPSR (população cai linearmente de population até N_MIN com o nº de avaliações)
    VARIANTS = ('jade', 'shade', 'lshade')
    H        = 6
    C        = 0.1
    N_MIN    = 4
    P_MAX    = 0.2  # teto do p sorteado por indivíduo na shade (Tanabe & Fukunaga 2013: p_i ~ U[2/N, 0.2])

    def __init__(self, objective, variables, maximize=True, population=None, generations=300, variant='lshade',
                 pbest=0.11, archive=2.6, patience=None, target=None, constraints=None, seed=None, memory=None,
                 workers=1, backend='thread', verbose=True):
        if variant not in self.VARIANTS:
            raise ValueError(f"variant inválida '{variant}'; use {self.VARIANTS}.")
        if not 0.0 < pbest <= 1.0:
            raise ValueError('pbest deve estar em (0, 1].')
        if archive < 0.0:
            raise ValueError('archive (taxa do arquivo) deve ser >= 0.')
        self.problem = Problem(objective, variables, maximize, constraints, backend)

        self.nInit = 18 * self.problem.nVars if population is None else int(population)
        if self.nInit < 8:
            raise ValueError('population deve ser >= 8 (ou None para 18·nVars).')
        self.variant     = variant
        self.jade        = variant == 'jade'
        self.shade       = variant == 'shade'
        self.lpsr        = variant == 'lshade'
        self.hsize       = 1 if self.jade else self.H  # jade usa uma média única; shade/lshade, memória H
        self.generations = int(generations)
        self.maxNfe      = self.nInit * self.generations
        self.pbest       = float(pbest)
        self.archive     = float(archive)
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
        self.recorder = Recorder(self.problem.genes, self.generations, self.verbose, f'DE {self.variant.upper()}')
        self.memory   = Memory(self.memoryPath, self.variant, self.config(), self.problem, self.recorder)
        self.stopped  = self.generations
        stop          = Stopper(self.patience, self.target, self.problem.weight)
        pool          = Pool(self.workers, self.backend, self.seed)

        mapFunc = pool.start()
        try:
            state = self.memory.start()
            if state is None:
                X    = rng.uniform(self.problem.low, self.problem.up, (self.nInit, self.problem.nVars))
                raw, nfe = self.problem.score(X, mapFunc), self.nInit
                mF, mCR  = np.full(self.hsize, 0.5), np.full(self.hsize, 0.5)
                arch = np.empty((0, self.problem.nVars))
                k    = gen = 0
            else:
                arrays, meta = state
                X, raw, arch, mF, mCR = arrays['X'], arrays['raw'], arrays['arch'], arrays['mF'], arrays['mCR']
                nfe, k, gen = meta['nfe'], meta['k'], meta['done']
                rand.set(meta['rng'])
            origin = self.memory.origin(nfe)  # orçamento aqui é avaliação, não geração
            if origin and len(X) < self.nInit:
                # Extensão de uma campanha fechada: o LPSR já tinha encolhido a população até N_MIN e sem
                # reinflar o ciclo novo rodaria inteiro com 4 indivíduos. Os sobreviventes seguem como elite.
                fresh = rng.uniform(self.problem.low, self.problem.up, (self.nInit - len(X), self.problem.nVars))
                X, raw = np.vstack([X, fresh]), np.vstack([raw, self.problem.score(fresh, mapFunc)])
                nfe   += len(fresh)
            wf    = raw[:, 0] * self.problem.weight
            limit = origin + self.maxNfe
            if state is None:
                self.recorder.record(gen, len(X), raw)  # retomada não re-grava: o histórico salvo já termina
                self.recorder.sample(X, raw)            # nesta geração, com os valores originais
            if state and not origin:
                stop.set(state[1].get('stop'))  # retomada: a paciência continua de onde parou (extensão reinicia)
            stop.check(float(np.max(wf)))
            bar = tqdm(total=limit, initial=nfe, desc=f'DE {self.variant.upper()}', unit='ev') if self.verbose else None

            while nfe < limit:
                gen += 1
                idx = rng.integers(self.hsize, size=len(X))
                F   = self.scale(mF[idx], rng)
                CR  = np.clip(mCR[idx] + 0.1 * rng.standard_normal(len(X)), 0.0, 1.0)
                U   = np.clip(self.cross(X, self.mutate(X, wf, F, arch, rng), CR, rng), self.problem.low, self.problem.up)

                rawU = self.problem.score(U, mapFunc)
                wfU  = rawU[:, 0] * self.problem.weight
                nfe += len(X)
                better  = wfU > wf                # sucesso estrito: alimenta arquivo e adaptação de F/CR
                replace = wfU >= wf               # aceitar empate move a população por platôs/regiões inviáveis
                if better.any():
                    arch = np.vstack([arch, X[better]]) if len(arch) else X[better].copy()
                    k    = self.remember(mF, mCR, F[better], CR[better], wfU[better] - wf[better], k)
                X[replace], raw[replace], wf[replace] = U[replace], rawU[replace], wfU[replace]
                X, raw, wf, arch = self.reduce(X, raw, wf, arch, nfe - origin, rng)

                self.recorder.record(gen, len(X), raw)
                self.recorder.sample(X, raw)
                self.memory.save({'X': X, 'raw': raw, 'arch': arch, 'mF': mF, 'mCR': mCR}, {'done': gen, 'nfe': nfe, 'k': k, 'rng': rand.get(), 'stop': stop.get()})
                if bar is not None:
                    bar.update(len(U)); bar.set_postfix(best=f'{raw[np.argmax(wf)][0]:.6g}', pop=len(X))

                if stop.check(float(np.max(wf))):
                    self.stopped = gen
                    break
            if bar is not None:
                bar.close()
        finally:
            pool.stop()

        winner         = int(np.argmax(wf))
        self.best      = self.problem.decode(X[winner])
        self.bestScore = float(raw[winner][0])
        self.best, self.bestScore = self.memory.commit(self.best, self.bestScore, X[winner], self.stopped,
                                                        {'X': X, 'raw': raw, 'arch': arch, 'mF': mF, 'mCR': mCR},
                                                        {'done': gen, 'nfe': nfe, 'k': k, 'rng': rand.get(), 'stop': stop.get()})
        return self.best, self.bestScore

    def mutate(self, X, wf, F, arch, rng):
        n = len(X)
        # jade/lshade usam p fixo; a shade sorteia um p por indivíduo em [2/N, 0.2] (Tanabe & Fukunaga 2013).
        pn     = np.round(rng.uniform(2.0 / n, self.P_MAX, n) * n) if self.shade else np.full(n, round(self.pbest * n))
        pn     = np.maximum(pn, 2).astype(int)
        order  = np.argsort(-wf)
        xpbest = X[order[(rng.random(n) * pn).astype(int)]]  # cada alvo puxa para um pbest sorteado do seu topo
        XA     = np.vstack([X, arch]) if len(arch) else X     # r2 vem de população ∪ arquivo (JADE/with)
        r1     = self.distinct(n, n, rng, np.arange(n))
        r2     = self.distinct(n, len(XA), rng, np.arange(n), r1)
        return X + F[:, None] * (xpbest - X) + F[:, None] * (X[r1] - XA[r2])

    def distinct(self, n, high, rng, *avoid):
        r = rng.integers(high, size=n)
        while True:
            clash = np.zeros(n, bool)
            for a in avoid:
                clash |= r == a
            if not clash.any():
                return r
            r[clash] = rng.integers(high, size=int(clash.sum()))

    def scale(self, m, rng):
        # F_i ~ Cauchy(m, 0.1): reamostra os <= 0 e satura em 1 (Tanabe & Fukunaga).
        F = m + 0.1 * rng.standard_cauchy(len(m))
        while (F <= 0).any():
            bad = F <= 0
            F[bad] = m[bad] + 0.1 * rng.standard_cauchy(int(bad.sum()))
        return np.minimum(F, 1.0)

    def cross(self, X, V, CR, rng):
        n, d = X.shape
        mask = rng.random((n, d)) < CR[:, None]
        mask[np.arange(n), rng.integers(d, size=n)] = True
        return np.where(mask, V, X)

    def remember(self, mF, mCR, sF, sCR, gains, k):
        w   = gains / gains.sum()
        lF  = np.sum(w * sF ** 2) / np.sum(w * sF)  # média de Lehmer ponderada (F)
        aCR = float(np.sum(w * sCR))               # média aritmética ponderada (CR)
        if self.jade:
            mF[0]  = (1 - self.C) * mF[0] + self.C * lF
            mCR[0] = (1 - self.C) * mCR[0] + self.C * aCR
            return 0
        mF[k], mCR[k] = lF, aCR
        return (k + 1) % self.hsize

    def reduce(self, X, raw, wf, arch, nfe, rng):
        if self.lpsr:
            # LPSR (Tanabe & Fukunaga 2014, eq. 8): N cai linearmente de nInit a N_MIN ao longo do ciclo.
            n = max(self.N_MIN, int(round((self.N_MIN - self.nInit) / self.maxNfe * nfe + self.nInit)))
            if n < len(X):
                keep = np.argsort(-wf)[:n]
                X, raw, wf = X[keep], raw[keep], wf[keep]
        cap = int(round(self.archive * len(X)))
        if len(arch) > cap:
            arch = arch[rng.choice(len(arch), cap, replace=False)] if cap else np.empty((0, X.shape[1]))
        return X, raw, wf, arch

    def config(self):
        return {'population': self.nInit, 'generations': self.generations, 'variant': self.variant,
                'pbest': self.pbest, 'archive': self.archive, 'patience': self.patience, 'target': self.target, 'maxNfe': self.maxNfe}

    # O RETRATO DESTE RUN — O NatureSelector PENDURA plotMetrics/plotGraph/plotVariables NELE
    def portrait(self):
        if not hasattr(self, 'recorder'):
            # Sem update() nesta sessão: reconstrói o retrato só com o que a pasta de memory guardou.
            self.recorder = Recorder(self.problem.genes, self.generations, False, f'DE {self.variant.upper()}')
            self.memory   = Memory(self.memoryPath, self.variant, self.config(), self.problem, self.recorder)
            if self.memory.start() is None:
                raise RuntimeError('Execute update() antes de plotar, ou aponte memory= para uma pasta com campanha salva.')
            self.best, self.bestScore = self.memory.best, self.memory.score
        return Portrait(self.problem, self.recorder, self.best, self.bestScore)
