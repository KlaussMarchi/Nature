import numpy as np
from tqdm.auto import tqdm
from ...Processing.Engine.index import Engine


# DE current-to-pbest/1/bin COM ARQUIVO (Zhang & Sanderson 2009; Tanabe & Fukunaga 2013/2014). variant:
# jade = média única com taxa C, shade = memória de sucesso H, lshade = shade + LPSR encolhendo a população,
# graded = a proposta do trabalho final (passo de atração graduado pela distância + deflação de bacia).
class AdaptiveDE(Engine):
    VARIANTS = ('jade', 'shade', 'lshade', 'graded')
    H        = 6
    C        = 0.1
    N_MIN    = 4
    P_MAX    = 0.2   # teto do p sorteado por indivíduo na shade (Tanabe & Fukunaga 2013: p_i ~ U[2/N, 0.2])
    RHO      = 0.5   # raio da bacia proibida, em arestas da caixa
    TAU      = 1e-5  # nuvem morta: desvio médio por coordenada abaixo desta fração da aresta
    HOLD     = 0.05  # e o melhor de todos parado por esta fração do orçamento

    def __init__(self, objective, variables, maximize=True, population=None, generations=300, variant='lshade',
                 pbest=0.11, archive=2.6, patience=None, target=None, constraints=None, seed=None, memory=None,
                 workers=1, backend='thread', verbose=True):
        if variant not in self.VARIANTS:
            raise ValueError(f"variant inválida '{variant}'; use {self.VARIANTS}.")
        if not 0.0 < pbest <= 1.0:
            raise ValueError('pbest deve estar em (0, 1].')
        if archive < 0.0:
            raise ValueError('archive (taxa do arquivo) deve ser >= 0.')
        super().__init__(objective, variables, maximize, constraints, backend, workers, seed, memory, verbose, patience, target)

        self.size = 18 * self.problem.nVars if population is None else int(population)
        if self.size < 8:
            raise ValueError('population deve ser >= 8 (ou None para 18·nVars).')
        self.MODEL   = variant
        self.DESC    = f'DE {variant.upper()}'
        self.variant = variant
        self.jade    = variant == 'jade'
        self.shade   = variant == 'shade'
        self.graded  = variant == 'graded'
        self.lpsr    = variant in ('lshade', 'graded')
        self.hsize   = 1 if self.jade else self.H
        self.span    = int(generations)
        self.maxNfe  = self.size * self.span
        self.pbest   = float(pbest)
        self.archive = float(archive)
        self.radius  = self.RHO * float(np.mean(self.problem.up - self.problem.low))

    def update(self):
        rng = self.open()

        try:
            arrays = self.resume()
            if arrays is None:
                X   = rng.uniform(self.problem.low, self.problem.up, (self.size, self.problem.nVars))
                raw, self.nfe = self.score(X), self.size
                mF, mCR = np.full(self.hsize, 0.5), np.full(self.hsize, 0.5)
                arch    = np.empty((0, self.problem.nVars))
                tabu    = np.empty((0, self.problem.nVars))
                self.k  = gen = 0
            else:
                X, raw, arch, mF, mCR = (arrays[k] for k in ('X', 'raw', 'arch', 'mF', 'mCR'))
                tabu                  = arrays.get('tabu', np.empty((0, self.problem.nVars)))
                self.nfe, self.k, gen = self.meta['nfe'], self.meta['k'], self.meta['done']
            self.pack = lambda: {'X': X, 'raw': raw, 'arch': arch, 'mF': mF, 'mCR': mCR, 'tabu': tabu}

            done = self.nfe   # origem do ciclo antes de reinflar, senão a extensão ganha crédito
            # extensão de campanha fechada: sem reinflar, o ciclo novo rodaria inteiro com N_MIN indivíduos
            if self.memory.origin(done) and len(X) < self.size:
                fresh     = rng.uniform(self.problem.low, self.problem.up, (self.size - len(X), self.problem.nVars))
                X, raw    = np.vstack([X, fresh]), np.vstack([raw, self.score(fresh)])
                self.nfe += len(fresh)
            true  = raw[:, 0] * self.problem.weight
            wf    = self.forbid(X, true, tabu)
            crown = int(np.argmax(true))
            elite = (X[crown].copy(), raw[crown].copy())
            mark  = spent = self.nfe
            # o orçamento aqui é avaliação, não geração: a L-SHADE encolhe a população e faz mais gerações
            limit = self.begin(done, X, raw, len(X), float(true[crown]), self.maxNfe)
            self.bar = tqdm(total=limit, initial=self.nfe, desc=self.DESC, unit='ev') if self.verbose else None

            while self.nfe < limit:
                gen += 1
                idx = rng.integers(self.hsize, size=len(X))
                F   = self.scale(mF[idx], rng)
                CR  = np.clip(mCR[idx] + 0.1 * rng.standard_normal(len(X)), 0.0, 1.0)
                U   = self.cross(X, self.repair(X, self.mutate(X, wf, F, arch, rng)), CR, rng)

                rawU = self.score(U)
                trueU = rawU[:, 0] * self.problem.weight
                wfU   = self.forbid(U, trueU, tabu)
                self.nfe += len(X)
                gain      = trueU - true      # o ganho que alimenta a memória é sempre o real, nunca o deflacionado
                better    = (wfU > wf) & (gain > 0)  # sucesso estrito: alimenta arquivo e adaptação de F/CR
                replace   = wfU >= wf         # aceitar empate move a população por platôs/regiões inviáveis
                if better.any():
                    arch = np.vstack([arch, X[better]]) if len(arch) else X[better].copy()
                    self.remember(mF, mCR, F[better], CR[better], gain[better])
                X[replace], raw[replace], true[replace], wf[replace] = U[replace], rawU[replace], trueU[replace], wfU[replace]

                top = int(np.argmax(true))
                if true[top] > elite[1][0] * self.problem.weight:
                    elite, mark = (X[top].copy(), raw[top].copy()), self.nfe
                    tabu        = tabu[:0]   # bateu o incumbente: a bacia proibida só atrapalha daqui em diante
                X, raw, true, wf, arch = self.reduce(X, raw, true, wf, arch, self.nfe - self.origin, rng)
                signal = float(elite[1][0] * self.problem.weight)

                if self.bar is not None:
                    self.bar.update(len(U))
                # o gasto vem do contador, não do tamanho da população: o religamento também paga avaliações
                used, spent = self.nfe - spent, self.nfe
                if self.tick(gen, used, X, raw, signal, pop=len(X)):
                    break
                if self.exhausted(X, mark):
                    X, raw, true, wf, tabu, arch, mark = self.revive(X, raw, wf, tabu, rng)
                    mF[:], mCR[:], self.k = 0.5, 0.5, 0
            self.stopped = gen   # o open() deixou self.span aqui, que não é o nº de gerações do ciclo
            if self.bar is not None:
                self.bar.close()
        finally:
            self.pool.stop()

        return self.finish(elite[0], elite[1][0], gen)

    def config(self):
        return {'population': self.size, 'generations': self.span, 'variant': self.variant, 'pbest': self.pbest,
                'archive': self.archive, 'patience': self.patience, 'target': self.target, 'maxNfe': self.maxNfe}

    # ALÉM DA GERAÇÃO, O CHECKPOINT CARREGA O ORÇAMENTO GASTO E O PONTEIRO DA MEMÓRIA DE SUCESSO
    def mark(self, gen):
        return {**super().mark(gen), 'nfe': self.nfe, 'k': self.k}

    def mutate(self, X, wf, F, arch, rng):
        n = len(X)
        # jade/lshade usam p fixo; a shade sorteia p_i ~ U[2/N, 0.2] (Tanabe & Fukunaga 2013)
        pn     = np.round(rng.uniform(2.0 / n, self.P_MAX, n) * n) if self.shade else np.full(n, round(self.pbest * n))
        pn     = np.maximum(pn, 2).astype(int)
        order  = np.argsort(-wf)
        xpbest = X[order[(rng.random(n) * pn).astype(int)]]   # cada alvo puxa para um pbest do seu topo
        XA     = np.vstack([X, arch]) if len(arch) else X     # r2 vem de população ∪ arquivo
        r1     = self.distinct(n, n, rng, np.arange(n))
        r2     = self.distinct(n, len(XA), rng, np.arange(n), r1)
        return X + self.pull(X, X[order[0]], F)[:, None] * (xpbest - X) + F[:, None] * (X[r1] - XA[r2])

    # PROPOSTA, PARTE 1 — PASSO DE ATRAÇÃO DE COMPRIMENTO UNIFORME. NA DE O PASSO F(x_pbest − x_i) É
    # PROPORCIONAL À DISTÂNCIA AO ATRATOR, ENTÃO A NUVEM CONTRAI EM PROGRESSÃO GEOMÉTRICA E O PASSO MORRE
    # JUNTO COM A DIVERSIDADE. O FATOR (d_max/d_i) FAZ TODO INDIVÍDUO AVANÇAR O MESMO COMPRIMENTO — O RAIO
    # DA NUVEM — E A CONTRAÇÃO VIRA ARITMÉTICA. A RAMPA (1−t)² DEVOLVE A L-SHADE PURA NO FIM DO ORÇAMENTO,
    # QUE É ONDE A F2 PRECISA DE REFINO LIMPO, E A TRAVA EM 1 É O PRÓPRIO TETO DE F DO SHADE: SEM ELA QUEM
    # ESTÁ EM CIMA DO INCUMBENTE É ARREMESSADO PARA LONGE.
    def pull(self, X, best, F):
        if not self.graded:
            return F
        d   = np.linalg.norm(X - best, axis=1)
        far = float(d.max())
        if far <= 0.0:
            return F
        ramp = (1.0 - min((self.nfe - self.origin) / self.maxNfe, 1.0)) ** 2
        return np.minimum(F * np.where(d > 0, far / np.where(d > 0, d, 1.0), 1.0) ** ramp, 1.0)

    # PROPOSTA, PARTE 2 — A BACIA JÁ EXAURIDA VIRA REGIÃO PROIBIDA: QUEM CAI DENTRO NUNCA É SELECIONADO.
    # A AVALIAÇÃO CONTINUA VALENDO PARA O INCUMBENTE, ENTÃO NENHUMA AVALIAÇÃO É PERDIDA.
    def forbid(self, X, true, tabu):
        if not len(tabu):
            return true
        near = np.linalg.norm(X[:, None, :] - tabu[None, :, :], axis=2).min(1) < self.radius
        return np.where(near, -self.problem.WORST, true)

    # DISPARO DA PARTE 2: A NUVEM NÃO DISTINGUE MAIS NADA **E** O MELHOR DE TODOS NÃO ANDA HÁ UMA JANELA DO
    # ORÇAMENTO. AS DUAS CONDIÇÕES JUNTAS: SÓ A PRIMEIRA MATA O REFINO FINAL DA F2/F6/F7, SÓ A SEGUNDA
    # RELIGA A F9/F13, QUE AINDA ESTÃO VIVAS QUANDO O ORÇAMENTO ACABA.
    def exhausted(self, X, mark):
        if not self.graded or len(X) < 8 or self.nfe - mark < self.HOLD * self.maxNfe:
            return False
        return float(np.mean(np.std(X, 0) / (self.problem.up - self.problem.low))) < self.TAU

    def revive(self, X, raw, wf, tabu, rng):
        tabu   = np.vstack([tabu, X[int(np.argmax(wf))]])
        X      = rng.uniform(self.problem.low, self.problem.up, (len(X), self.problem.nVars))
        raw    = self.score(X)
        true   = raw[:, 0] * self.problem.weight
        self.nfe += len(X)
        return X, raw, true, self.forbid(X, true, tabu), tabu, np.empty((0, self.problem.nVars)), self.nfe

    # REPARO DE BORDA DO CÓDIGO ORIGINAL DA L-SHADE: O GENE FORA DA CAIXA VAI PARA O MEIO ENTRE O PAI E A
    # BORDA. O np.clip EMPILHA A POPULAÇÃO INTEIRA SOBRE A FACE E CONGELA A COORDENADA, PORQUE A DIFERENÇA
    # ENTRE DOIS INDIVÍDUOS GRUDADOS NO MESMO LIMITE É ZERO.
    def repair(self, X, V):
        lo, up = self.problem.low, self.problem.up
        return np.where(V < lo, 0.5 * (lo + X), np.where(V > up, 0.5 * (up + X), V))

    def distinct(self, n, high, rng, *avoid):
        r = rng.integers(high, size=n)
        while True:
            clash = np.zeros(n, bool)
            for a in avoid:
                clash |= r == a
            if not clash.any():
                return r
            r[clash] = rng.integers(high, size=int(clash.sum()))

    # F_i ~ Cauchy(m, 0.1): REAMOSTRA OS <= 0 E SATURA EM 1 (Tanabe & Fukunaga)
    def scale(self, m, rng):
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

    def remember(self, mF, mCR, sF, sCR, gains):
        w   = gains / gains.sum()
        lF  = np.sum(w * sF ** 2) / np.sum(w * sF)  # média de Lehmer ponderada (F)
        aCR = float(np.sum(w * sCR))               # média aritmética ponderada (CR)
        if self.jade:
            mF[0]  = (1 - self.C) * mF[0] + self.C * lF
            mCR[0] = (1 - self.C) * mCR[0] + self.C * aCR
            return
        mF[self.k], mCR[self.k] = lF, aCR
        self.k = (self.k + 1) % self.hsize

    def reduce(self, X, raw, true, wf, arch, nfe, rng):
        if self.lpsr:
            # LPSR (Tanabe & Fukunaga 2014, eq. 8): N cai linearmente de size a N_MIN ao longo do ciclo
            n = max(self.N_MIN, int(round((self.N_MIN - self.size) / self.maxNfe * nfe + self.size)))
            if n < len(X):
                keep = np.argsort(-wf)[:n]
                X, raw, true, wf = X[keep], raw[keep], true[keep], wf[keep]
        cap = int(round(self.archive * len(X)))
        if len(arch) > cap:
            arch = arch[rng.choice(len(arch), cap, replace=False)] if cap else np.empty((0, X.shape[1]))
        return X, raw, true, wf, arch
