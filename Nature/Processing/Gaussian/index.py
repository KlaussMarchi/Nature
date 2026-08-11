import numpy as np
from scipy import stats


# CLASSE QUE ANALISA A AMOSTRA DAS n CORRIDAS DE UM ALGORITMO: O HISTOGRAMA COM A NORMAL AJUSTADA E O
# INTERVALO DE 95% CORRIGIDO POR t DE STUDENT, E O Q-Q PLOT COM O W DE SHAPIRO-WILK CONTRA O CRÍTICO DA
# TABELA. CRIADA PELO Plotter DEPOIS DO update(), COM O target QUANDO A MÉTRICA É ERRO.
class Gaussian:
    # t CRÍTICO BILATERAL A 95% POR GRAU DE LIBERDADE
    STUDENT = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201,
        12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
        22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021,
        60: 2.000, 80: 1.990, 100: 1.984, 120: 1.980, 'infty': 1.960
    }

    # W CRÍTICO DE SHAPIRO-WILK A 5% POR TAMANHO DE AMOSTRA
    SHAPIRO = {
        3: 0.767, 4: 0.748, 5: 0.762, 6: 0.788, 7: 0.803, 8: 0.818, 9: 0.829, 10: 0.842, 11: 0.850, 12: 0.859,
        13: 0.866, 14: 0.874, 15: 0.881, 16: 0.887, 17: 0.892, 18: 0.897, 19: 0.901, 20: 0.905, 21: 0.908,
        22: 0.911, 23: 0.914, 24: 0.916, 25: 0.918, 26: 0.920, 27: 0.923, 28: 0.924, 29: 0.926, 30: 0.927,
        31: 0.929, 32: 0.930, 33: 0.931, 34: 0.933, 35: 0.934, 36: 0.935, 37: 0.936, 38: 0.938, 39: 0.939,
        40: 0.940, 41: 0.941, 42: 0.942, 43: 0.943, 44: 0.944, 45: 0.945, 46: 0.945, 47: 0.946, 48: 0.947,
        49: 0.947, 50: 0.947
    }

    FLOOR  = 1e-8      # limiar de zero do CEC'14 (seção 2.1): piso do log, senão a corrida que achou o ótimo vira -infinito
    SIGMAS = 2.2       # desvios que a janela do painel cobre: o sino inteiro sem sobra de painel vazio
    DECADE = 2.0       # meia-janela quando a amostra não tem dispersão nenhuma (décadas no erro)
    BINS   = (5, 12)   # piso e teto de bins: abaixo de 5 não sobra forma para ler, acima de 12 cada bin vira ruído
    TINY   = 1e-12     # dispersão relativa abaixo disso é ruído de float: as corridas caíram todas no mesmo ponto

    def __init__(self, scores, target=None, up=False, name=''):
        self.name = name
        self.up   = up
        data      = np.asarray(scores, float)
        # Com alvo a métrica é erro ao ótimo, que varre ordens de grandeza: a análise inteira (normal, t e
        # Shapiro) acontece em log10, que é onde a amostra tem chance de ser normal e é o que o painel desenha.
        self.data = np.abs(data - target) if target is not None else data
        self.log  = target is not None
        self.u    = np.log10(np.maximum(self.data, self.FLOOR)) if self.log else self.data

    # t CRÍTICO DA TABELA PARA n-1 GRAUS DE LIBERDADE, PELA LINHA MAIS PRÓXIMA QUANDO NÃO HÁ A EXATA
    def get(self, n):
        df = n - 1
        if df >= 120:
            return self.STUDENT['infty']
        return self.STUDENT[min([k for k in self.STUDENT if isinstance(k, int)], key=lambda k: abs(k - df))]

    # W CRÍTICO A 5%, SATURADO NAS PONTAS DA TABELA
    def getW(self, n):
        return self.SHAPIRO[min(max(n, min(self.SHAPIRO)), max(self.SHAPIRO))]

    # AMOSTRA SEM DISPERSÃO: TODO MUNDO ACHOU O MESMO ÓTIMO E O QUE SOBRA É RUÍDO DO float. NÃO HÁ JANELA,
    # BINS, NORMAL NEM TESTE DE NORMALIDADE PARA TIRAR DAÍ — E O EIXO NEM CONSEGUE SEPARAR OS VALORES.
    def flat(self):
        lo, hi = float(self.u.min()), float(self.u.max())
        return hi - lo <= self.TINY * max(abs(lo), abs(hi), 1.0)

    # AS ESTATÍSTICAS DOS DOIS PAINÉIS: mean/median/std/best/worst NA ESCALA ORIGINAL, QUE É A DA TABELA DO
    # RELATÓRIO, E mu/sigma/lo/hi NA ESCALA DA ANÁLISE, JÁ COM A CORREÇÃO DE STUDENT SOBRE A MÉDIA.
    def info(self):
        n     = len(self.u)
        mu    = float(self.u.mean())
        sigma = float(self.u.std(ddof=1)) if n > 1 and not self.flat() else 0.0
        t     = self.get(n)
        half  = float(t * sigma / np.sqrt(n))
        crit  = self.getW(n)
        # Com dois pontos o W é 1 por construção, e com a amostra inteira no mesmo valor não há desvio algum
        # da reta para o teste achar: nos dois casos a normalidade passa, e o Shapiro nem chega a ser chamado.
        w = 1.0 if n < 3 or self.flat() else float(stats.shapiro(self.u).statistic)

        best, worst = (self.data.max(), self.data.min()) if self.up else (self.data.min(), self.data.max())
        return {
            'mean': float(self.data.mean()), 'median': float(np.median(self.data)),
            'std': float(self.data.std(ddof=1)) if n > 1 else 0.0,
            'best': float(best), 'worst': float(worst),
            'mu': mu, 'sigma': sigma, 't': t, 'student': half, 'lo': mu - half, 'hi': mu + half,
            'w': w, 'w_crit': crit, 'normal': w > crit
        }

    # DISTRIBUIÇÃO DAS n CORRIDAS: OS BINS, CADA CORRIDA COMO UM PONTO DENTRO DA SUA BARRA, A NORMAL AJUSTADA
    # E A MÉDIA COM OS LIMITES DE 95% POR STUDENT.
    def bell(self, ax):
        res         = self.info()
        n           = len(self.u)
        mu, sigma   = res['mu'], res['sigma']
        left, right = self.window(mu, sigma)

        # Sem dispersão não há faixa para fatiar: sai uma barra só, de meia unidade para cada lado do valor.
        bins         = 1 if self.flat() else int(min(np.clip(np.ceil(2 * n ** (1 / 3)), *self.BINS), n))  # regra de Rice
        span         = (mu - 0.5, mu + 0.5) if self.flat() else (float(self.u.min()), float(self.u.max()))
        count, edges = np.histogram(self.u, bins=bins, range=span)
        dens         = count / (n * (edges[1] - edges[0]))

        x   = np.linspace(left, right, 400)
        y   = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)) if sigma > 0 else None
        top = max(float(dens.max()), 0.0 if y is None else float(y.max()))

        ax.bar(self.spot(edges[:-1]), dens, width=self.spot(edges[1:]) - self.spot(edges[:-1]), align='edge',
               color='steelblue', alpha=0.40, edgecolor='#1f3b4d', linewidth=0.9,
               label=f'{bins} bins' if bins > 1 else 'corridas idênticas')

        # Cada corrida no seu valor real, empilhada dentro da própria barra: a barra é a pilha de pontos.
        slot = np.clip(np.digitize(self.u, edges) - 1, 0, bins - 1)
        rank = np.zeros(n)
        for b in np.unique(slot):
            rank[slot == b] = np.arange(count[b])
        ax.scatter(self.spot(self.u), dens[slot] * (rank + 0.5) / count[slot], s=18, color='#0d2b3a', zorder=5, label=f'{n} corridas')

        if y is not None:
            ax.plot(self.spot(x), y, color='#1f3b4d', linewidth=1.8, label='Normal ajustada')

        # Em log a média volta como média geométrica, que não é o µ da caixa: o rótulo diz qual das duas é a linha.
        ax.axvline(self.spot(mu), color='crimson', linestyle='--', linewidth=1.5, label='Média geométrica' if self.log else 'Média')
        ax.axvspan(self.spot(res['lo']), self.spot(res['hi']), color='crimson', alpha=0.12)
        for edge in (res['lo'], res['hi']):
            ax.axvline(self.spot(edge), color='crimson', linestyle=':', linewidth=1.4)
        ax.plot([], [], color='crimson', linestyle=':', linewidth=1.4, label='$\\pm t_{crit}$ (95%)')

        if self.log:
            ax.set_xscale('log')
        ax.set_xlim(self.spot(left), self.spot(right))
        ax.set_ylim(0, top * 1.45)

        ax.text(0.03, 0.97, self.summary(res), transform=ax.transAxes, verticalalignment='top', fontsize=8.5,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.legend(fontsize=7, loc='upper right')
        ax.set(title=self.tag('Distribuição das corridas'), xlabel='Erro ao ótimo' if self.log else 'Aptidão', ylabel='Densidade')
        ax.grid(True, alpha=0.3)

    # Q-Q PLOT NA MESMA ESCALA DO SINO: OS PONTOS SOBRE A RETA DA NORMAL AJUSTADA E O VEREDITO DE SHAPIRO
    def qq(self, ax):
        res  = self.info()
        q, v = stats.probplot(self.u, dist='norm', fit=False)
        # A reta sai dos próprios pontos ordenados, não de µ e σ: uma corrida fora da curva infla o σ e a
        # referência entraria torta na nuvem, dando ar de desvio onde o desvio é de uma corrida só.
        coef = [0.0, res['mu']] if self.flat() else np.polyfit(q, v, 1)

        ax.plot(q, np.polyval(coef, q), color='crimson', linewidth=1.5, label='Normal ajustada')
        ax.scatter(q, v, s=30, color='steelblue', edgecolors='#1f3b4d', linewidths=0.6, zorder=4, label='Corridas')
        ax.legend(fontsize=7, loc='upper left')
        ax.set(title=self.verdict(res), xlabel='Quantis teóricos',
               ylabel=f"Quantis da amostra{' em log' if self.log else ''}")
        ax.grid(True, alpha=0.3)

    # JANELA DO PAINEL NA ESCALA DA ANÁLISE: O SINO INTEIRO SEM CORTAR NENHUMA CORRIDA, E NO MODO ERRO SEM
    # DESCER ABAIXO DO PISO DO LOG NEM ABRIR UMA DÉCADA VAZIA DEPOIS DA PIOR.
    def window(self, mu, sigma):
        lo, hi = float(self.u.min()), float(self.u.max())
        half   = max(self.SIGMAS * sigma, hi - mu, mu - lo) * 1.15
        if self.flat():
            half = self.DECADE if self.log else max(abs(mu) * 0.5, 1.0)
        left, right = mu - half, mu + half
        if self.log:
            left, right = max(left, np.log10(self.FLOOR) - 1.5), min(right, hi + 1.0)
        return left, right

    def summary(self, res):
        lo, hi = float(self.spot(res['lo'])), float(self.spot(res['hi']))
        return (f"µ = {res['mean']:.4g}\nmediana = {res['median']:.4g}\nσ = {res['std']:.3g}\n"
                f"best = {res['best']:.4g}\nworst = {res['worst']:.4g}\n"
                f"IC 95% = [{lo:.3g}, {hi:.3g}]")

    def verdict(self, res):
        status = 'NORMAL' if res['normal'] else 'NÃO NORMAL'
        return f"W = {res['w']:.4f} | $W_{{crit}}$ = {res['w_crit']:.3f} → {status}"

    # O NOME DO ALGORITMO ENTRE PARÊNTESES: A AMOSTRA É DAS CORRIDAS DE UM SÓ, E O TÍTULO É QUEM DIZ DE QUAL
    def tag(self, title):
        return f'{title} ({self.name})' if self.name else title

    # A ESCALA DA ANÁLISE DE VOLTA PARA A ORIGINAL, QUE É ONDE OS EIXOS SÃO ROTULADOS
    def spot(self, v):
        return 10.0 ** v if self.log else v
