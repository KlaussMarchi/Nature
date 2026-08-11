import numpy as np


# CLASSE DO EARLY STOPPING POR PACIÊNCIA — CONSULTADA A CADA GERAÇÃO NO LOOP DE TODO MODELO.
class Stopper:
    TOL = 1e-9

    def __init__(self, patience):
        self.patience = patience
        self.best     = -np.inf
        self.bad      = 0

    def check(self, signal):
        # Early stopping por sinal ponderado (maior = melhor); devolve True para parar.
        if self.patience is None:
            return False
        if signal > self.best + self.TOL:
            self.best, self.bad = signal, 0
            return False
        self.bad += 1
        return self.bad >= self.patience

    def get(self):
        # Vai junto do state.npz: sem a contagem de estagnação a retomada reinicia a paciência e para
        # numa geração diferente da do run ininterrupto.
        return {'best': float(self.best), 'bad': int(self.bad)}

    def set(self, snap):
        if snap is None:
            return  # campanha gravada antes deste campo existir
        self.best, self.bad = float(snap['best']), int(snap['bad'])
