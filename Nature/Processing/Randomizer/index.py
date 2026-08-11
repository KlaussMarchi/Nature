import random
import numpy as np


# CLASSE QUE SEMEIA E FOTOGRAFA OS DOIS GERADORES ALEATÓRIOS — CHAMADA NO INÍCIO DO update() DE TODO MODELO
# E A CADA GRAVAÇÃO DA Memory, PARA A RETOMADA SAIR BIT-IDÊNTICA.
class Randomizer:
    # Duas fontes de aleatoriedade: random alimenta os operadores da DEAP e np.random os genomas. As duas
    # precisam de semente e de snapshot, senão a retomada não sai bit-idêntica.
    def __init__(self, seed=None):
        self.seed = seed
        self.rng  = None

    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        return self.rng

    def get(self):
        return {'np': self.rng.bit_generator.state, 'py': list(random.getstate())}

    def set(self, snap):
        self.rng.bit_generator.state = snap['np']
        py = snap['py']
        random.setstate((py[0], tuple(py[1]), py[2]))
