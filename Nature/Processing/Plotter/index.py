import numpy as np
import matplotlib.pyplot as plt
from ..Gaussian.index import Gaussian


# COMPARA OS ALGORITMOS DE UM MESMO PROBLEMA A PARTIR DO DataFrame QUE A CÉLULA MONTA: UM FIGURE COMPARATIVO
# (BARRAS + RELEVO DO VENCEDOR) E UM DE DETALHE DO VENCEDOR (DISTRIBUIÇÃO, Q-Q E CONVERGÊNCIA).
class Plotter:
    FLOOR = 1e-8   # limiar de zero do CEC'14 (seção 2.1): abaixo disso conta como ter achado o ótimo
    SPAN  = 100    # razão entre a maior e a menor barra a partir da qual o eixo vira log

    def __init__(self, board, value='erro', label='algorithm', title='Comparativo', maximize=False, target=None):
        self.board  = board
        self.target = target
        # sem a coluna de erro sobra a aptidão crua, e a tolerância da competição não se aplica
        self.value  = value if value in board else 'f'
        self.label  = label
        self.title  = title
        self.error  = self.value == 'erro'
        self.up     = maximize and not self.error

    # O COMPARATIVO, A CONVERGÊNCIA LADO A LADO E, POR FIM, O FIGURE DE DETALHE DO VENCEDOR
    def plot(self, best=None, runs=None, save=None):
        portrait = best.portrait() if best is not None else None
        fig      = plt.figure(figsize=(11.6, 4.6) if portrait else (6.4, 4.6))
        grid     = fig.add_gridspec(1, 1 + (portrait is not None))

        self.bars(fig.add_subplot(grid[0, 0]))
        if portrait:
            portrait.shape(fig.add_subplot(grid[0, 1], projection=portrait.projection()))
        self.close(self.path(save, 'board'))

        if runs:
            self.curves(plt.figure(figsize=(11.6, 4.2)).add_subplot(1, 1, 1), runs)
            self.close(self.path(save, 'curves'))
        if portrait is not None:
            self.detail(portrait, best, save)

    # CONVERGÊNCIA DOS ALGORITMOS NO MESMO EIXO: A CORRIDA VENCEDORA DE CADA UM, ERRO CONTRA AVALIAÇÕES
    def curves(self, ax, runs):
        for run in runs:
            rec  = run.optimizer.recorder.records
            walk = np.asarray(rec['max' if self.up else 'min'], float).ravel()
            walk = (np.maximum if self.up else np.minimum).accumulate(walk)
            ax.plot(np.cumsum(rec['nevals']), np.maximum(np.abs(walk - self.target), self.FLOOR)
                    if self.target is not None else walk, linewidth=1.4, label=str(run.name))
        if self.target is not None:
            ax.set_yscale('log')
            ax.axhline(self.FLOOR, color='crimson', linestyle='--', linewidth=1.2)
        ax.set(xlabel='Avaliações', ylabel='Erro ao ótimo' if self.target is not None else 'Aptidão',
               title=f'{self.title} — convergência da melhor corrida')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=8)

    # COM n_gaussian=1 NÃO HÁ AMOSTRA E OS DOIS PAINÉIS DE CIMA NÃO EXISTEM
    def detail(self, portrait, best, save):
        spread = len(best.scores) > 1
        fig    = plt.figure(figsize=(13.5, 9.2 if spread else 4.6))
        grid   = fig.add_gridspec(2 if spread else 1, 2)

        if spread:
            gauss = Gaussian(best.scores, self.target, self.up, best.name)
            gauss.bell(fig.add_subplot(grid[0, 0]))
            gauss.qq(fig.add_subplot(grid[0, 1]))
        portrait.metrics(fig.add_subplot(grid[-1, :]))
        self.close(self.path(save, 'runs'))

    def bars(self, ax):
        board  = self.board.sort_values(self.value, ascending=not self.up)  # do melhor para o pior
        names  = [str(n) for n in board[self.label]]
        raw    = board[self.value].to_numpy(float)
        height = np.maximum(raw, self.FLOOR) if self.error else raw
        hit    = int((raw <= self.FLOOR).sum())
        bars   = ax.bar(names, height, color=plt.cm.tab10(np.arange(len(names))))

        if height.min() > 0 and height.max() / height.min() > self.SPAN:
            ax.set_yscale('log')   # o erro varia ordens de grandeza entre algoritmos
        if self.error:
            ax.axhline(self.FLOOR, color='crimson', linestyle='--', linewidth=1.2, label=f'tolerância {self.FLOOR:g}')
            ax.legend(fontsize=8)
        for bar, v in zip(bars, raw):
            text = '≈0' if self.error and v <= self.FLOOR else f'{v:.3g}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set(title=f'{self.title} — {hit} de {len(raw)} no ótimo' if self.error else self.title, ylabel='Erro ao ótimo' if self.error else 'Aptidão')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', labelrotation=30)

    def close(self, save):
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()

    # UM save, DOIS ARQUIVOS: O SUFIXO ENTRA ANTES DA EXTENSÃO
    def path(self, save, tag):
        if not save:
            return None
        root, dot, ext = save.rpartition('.')
        return f'{root}-{tag}.{ext}' if dot else f'{save}-{tag}.png'
