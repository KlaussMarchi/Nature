import numpy as np
from ..Gaussian.index import Gaussian


# CLASSE QUE COMPARA AS CORRIDAS DE UM MESMO PROBLEMA A PARTIR DO DataFrame QUE A CÉLULA JÁ MONTA. SÃO DOIS
# FIGURES: O COMPARATIVO — BARRAS DO MELHOR PARA O PIOR COM A TOLERÂNCIA DA COMPETIÇÃO E O RELEVO DO
# VENCEDOR — E O DETALHE DELE, COM A DISTRIBUIÇÃO DAS n CORRIDAS, O Q-Q PLOT E A CONVERGÊNCIA.
class Plotter:
    FLOOR = 1e-8   # limiar de zero do CEC'14 (seção 2.1): abaixo disso conta como ter achado o ótimo
    SPAN  = 100    # razão entre a maior e a menor barra a partir da qual o eixo vira log

    def __init__(self, board, value='erro', label='algorithm', title='Comparativo', maximize=False, target=None):
        self.board  = board
        self.target = target
        # Sem a coluna de erro sobra a aptidão crua, e aí a tolerância da competição não se aplica.
        self.value  = value if value in board else 'f'
        self.label  = label
        self.title  = title
        self.error  = self.value == 'erro'
        self.up     = maximize and not self.error

    # O COMPARATIVO COM O RELEVO DO VENCEDOR E, LOGO DEPOIS, O FIGURE DE DETALHE DELE
    def plot(self, best=None, save=None):
        import matplotlib.pyplot as plt

        portrait = best.portrait() if best is not None else None
        fig      = plt.figure(figsize=(11.6, 4.6) if portrait else (6.4, 4.6))
        grid     = fig.add_gridspec(1, 1 + (portrait is not None))

        self.bars(fig.add_subplot(grid[0, 0]))
        if portrait:
            portrait.shape(fig.add_subplot(grid[0, 1], projection=portrait.projection()))
        self.close(plt, self.path(save, 'board'))

        if portrait is not None:
            self.detail(plt, portrait, best, save)

    # O VENCEDOR SOZINHO: A DISTRIBUIÇÃO DAS n CORRIDAS E O Q-Q PLOT NA PRIMEIRA LINHA, A CONVERGÊNCIA NA
    # SEGUNDA. COM n_gaussian=1 NÃO HÁ AMOSTRA E OS DOIS PAINÉIS DE CIMA NÃO EXISTEM.
    def detail(self, plt, portrait, best, save):
        spread = len(best.scores) > 1
        fig    = plt.figure(figsize=(13.5, 9.2 if spread else 4.6))
        grid   = fig.add_gridspec(2 if spread else 1, 2)

        if spread:
            gauss = Gaussian(best.scores, self.target, self.up, best.name)
            gauss.bell(fig.add_subplot(grid[0, 0]))
            gauss.qq(fig.add_subplot(grid[0, 1]))
        portrait.metrics(fig.add_subplot(grid[-1, :]))
        self.close(plt, self.path(save, 'runs'))

    def bars(self, ax):
        import matplotlib.pyplot as plt

        board  = self.board.sort_values(self.value, ascending=not self.up)  # do melhor para o pior
        names  = [str(n) for n in board[self.label]]
        raw    = board[self.value].to_numpy(float)
        height = np.maximum(raw, self.FLOOR) if self.error else raw
        hit    = int((raw <= self.FLOOR).sum())
        bars = ax.bar(names, height, color=plt.cm.tab10(np.arange(len(names))))
        
        if height.min() > 0 and height.max() / height.min() > self.SPAN:
            ax.set_yscale('log')  # o erro varia ordens de grandeza entre algoritmos
        if self.error:
            ax.axhline(self.FLOOR, color='crimson', linestyle='--', linewidth=1.2, label=f'tolerância {self.FLOOR:g}')
            ax.legend(fontsize=8)
        for bar, v in zip(bars, raw):
            text = '≈0' if self.error and v <= self.FLOOR else f'{v:.3g}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set(title=f'{self.title} — {hit} de {len(raw)} no ótimo' if self.error else self.title, ylabel='Erro ao ótimo' if self.error else 'Aptidão')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', labelrotation=30)

    def close(self, plt, save):
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        plt.show()

    # UM save, DOIS ARQUIVOS: O SUFIXO ENTRA ANTES DA EXTENSÃO PARA A CHAMADA CONTINUAR PASSANDO UM CAMINHO SÓ
    def path(self, save, tag):
        if not save:
            return None
        root, dot, ext = save.rpartition('.')
        return f'{root}-{tag}.{ext}' if dot else f'{save}-{tag}.png'
