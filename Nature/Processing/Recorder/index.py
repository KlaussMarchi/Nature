import numpy as np
from tqdm.auto import tqdm


# CLASSE QUE ACUMULA O HISTÓRICO DA BUSCA (ESTATÍSTICAS POR GERAÇÃO + NUVEM DE AMOSTRAS) — ALIMENTADA NO
# LOOP DE TODO MODELO E CONSUMIDA DEPOIS PELO Plotter.
class Recorder:
    SAMPLES = 200      # gerações retidas na nuvem (subamostra temporal p/ o scatter do Plotter)
    PER_GEN = 40       # teto de indivíduos por geração (limita memória; a cobertura do range é feita no Plotter)
    BUDGET  = 1000000  # teto de floats da nuvem inteira: ela entra no state.npz a cada Memory.EVERY gerações
    KEYS    = ('gen', 'nevals', 'min', 'max')

    def __init__(self, genes, span, verbose, desc, unit='gen'):
        self.genes   = genes
        self.span    = span
        self.verbose = verbose
        self.desc    = desc
        self.unit    = unit
        # Sem teto por dimensão a nuvem virava 97% do checkpoint em alta dimensão (38 de 39,5 MB com D=500),
        # reescritos a cada 10 gerações. O Plotter desenha no máximo 800 pontos por variável de qualquer jeito.
        self.perGen  = max(1, min(self.PER_GEN, self.BUDGET // (2 * self.SAMPLES * len(genes))))
        self.records = {k: [] for k in self.KEYS}
        self.cloudX  = []
        self.cloudY  = []
        self.cloudG  = []
        self.srng    = np.random.default_rng(0)  # isolado do rng da busca: amostrar não altera a otimização

    def bar(self, start=1):
        span = range(start, self.span + 1)
        return tqdm(span, desc=self.desc, unit=self.unit) if self.verbose else span

    def record(self, gen, nevals, fits):
        if self.records['gen'] and gen <= self.records['gen'][-1]:
            # Retomada: o modelo re-registra a geração onde parou. Descarta o trecho refeito p/ não duplicar
            # ponto no eixo; se a campanha recomeça do zero (config mudou), a nuvem antiga também não vale.
            keep = int(np.searchsorted(self.records['gen'], gen))
            for k in self.KEYS:
                self.records[k] = self.records[k][:keep]
            if not keep:
                self.cloudX, self.cloudY, self.cloudG = [], [], []
        fits = np.asarray(fits)
        self.records['gen'].append(gen)
        self.records['nevals'].append(nevals)
        self.records['min'].append(np.min(fits, axis=0))
        self.records['max'].append(np.max(fits, axis=0))

    def sample(self, genomes, fits):
        # Nuvem (geração -> genoma -> métrica) do que foi explorado, p/ os scatters do Plotter. Amostra perGen
        # indivíduos por geração (senão a população inicial gigante domina) e afina pela metade ao passar de
        # 2·SAMPLES gerações — memória limitada. A geração vem do record(), que todo modelo chama antes daqui.
        f    = np.asarray(fits, float)
        y    = f[:, 0] if f.ndim > 1 else f
        pick = np.arange(len(y)) if len(y) <= self.perGen else self.srng.choice(len(y), self.perGen, replace=False)
        # np.take copia. Sem a cópia a nuvem guardava uma view do array que o modelo sobrescreve in-place a
        # cada geração, e todas as gerações acabavam mostrando a população final — só não aparecia quando a
        # população passava de perGen, porque aí o sorteio já copiava.
        self.cloudX.append(np.take(np.asarray(genomes, float), pick, axis=0))
        self.cloudY.append(np.take(y, pick))
        self.cloudG.append(self.records['gen'][-1])
        if len(self.cloudX) > 2 * self.SAMPLES:
            self.cloudX, self.cloudY, self.cloudG = self.cloudX[::2], self.cloudY[::2], self.cloudG[::2]

    def cloud(self):
        if not self.cloudX:
            return None, None, None
        gens = np.repeat(self.cloudG, [len(y) for y in self.cloudY])
        return np.vstack(self.cloudX), np.concatenate(self.cloudY), gens

    def get(self):
        # Snapshot p/ a Memory gravar junto do state.npz: sem ele a retomada perde tudo que veio antes e o
        # plot mostra só o trecho rodado depois do restart. A nuvem é ragged -> vai achatada + índices de corte.
        state = {'rec' + k.capitalize(): np.array(self.records[k]) for k in self.KEYS}
        state['recCloudX'] = np.vstack(self.cloudX) if self.cloudX else np.zeros((0, len(self.genes)))
        state['recCloudY'] = np.concatenate(self.cloudY) if self.cloudY else np.zeros(0)
        state['recCloudG'] = np.array(self.cloudG, int)
        state['recSplit']  = np.cumsum([len(y) for y in self.cloudY])[:-1].astype(int)
        return state

    def set(self, state):
        if 'recGen' not in state:
            return
        for k in self.KEYS:
            self.records[k] = list(state['rec' + k.capitalize()])
        cuts        = np.asarray(state['recSplit'], int)
        self.cloudX = [a for a in np.split(state['recCloudX'], cuts) if len(a)]
        self.cloudY = [a for a in np.split(state['recCloudY'], cuts) if len(a)]
        self.cloudG = list(state.get('recCloudG', []))
        if len(self.cloudG) != len(self.cloudY):
            self.cloudX, self.cloudY, self.cloudG = [], [], []  # campanha gravada antes da geração entrar na nuvem
