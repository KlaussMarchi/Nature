from functools import partial
from .Models.Genetic.index import GeneticOptimizer
from .Models.PSO.index import PSO
from .Models.GeneticPSO.index import GeneticPSO
from .Models.DE.index import DifferentialEvolution
from .Models.AdaptiveDE.index import AdaptiveDE


# FACHADA QUE ESCOLHE O MODELO PELO NOME E DELEGA update()/portrait()/info() — O ÚNICO IMPORT DO NOTEBOOK.
class NatureSelector:
    MODELS = {'genetic': GeneticOptimizer, 'pso': PSO, 'genetic_pso': GeneticPSO, 'de': DifferentialEvolution,
              'adaptive_de': AdaptiveDE, 'graded_de': partial(AdaptiveDE, variant='graded')}
    NAMES  = tuple(MODELS)

    def __init__(self, name, params, memory=None, n_gaussian=1):
        if name not in self.MODELS:
            raise ValueError(f"algoritmo '{name}' não existe; use um de {self.NAMES}.")
        self.name   = name
        self.params = {**params, 'memory': memory} if memory else params
        self.n      = int(n_gaussian)
        if self.n < 1:
            raise ValueError('n_gaussian deve ser >= 1.')
        # Cada corrida estenderia a mesma campanha gravada, então a amostra sairia dependente.
        if self.n > 1 and self.params.get('memory'):
            raise ValueError('n_gaussian > 1 não combina com memory: as corridas estenderiam a mesma campanha.')
        self.best      = None
        self.score     = None
        self.scores    = []
        self.optimizer = self.get(self.params)

    def get(self, params):
        return self.MODELS[self.name](**params)

    # CONSECUTIVAS A PARTIR DA SEMENTE DADA, PARA A AMOSTRA INTEIRA SER REPRODUTÍVEL
    def seeds(self):
        seed = self.params.get('seed')
        return [None] * self.n if seed is None else [seed + i for i in range(self.n)]

    def update(self):
        up          = self.optimizer.problem.weight > 0
        self.scores = []
        keep        = None
        for seed in self.seeds():
            model       = self.get(self.params if seed is None else {**self.params, 'seed': seed})
            best, score = model.update()
            self.scores.append(score)
            # Fica o melhor das n: é dele o retrato que os plot* desenham e o f que vai para o quadro.
            if keep is None or (score > keep[1]) == up:
                keep = (best, score, model)
        self.best, self.score, self.optimizer = keep
        return self.best, self.score

    # O MODELO NÃO SABE COM QUE NOME FOI ESCOLHIDO, E SEM ISSO NENHUM PAINEL DIZ DE QUEM É
    def portrait(self):
        portrait      = self.optimizer.portrait()
        portrait.name = self.name
        return portrait

    def plotMetrics(self, save=None):
        self.portrait().plotMetrics(save)

    def plotGraph(self, save=None):
        self.portrait().plotGraph(save)

    def plotVariables(self, mode='range', save=None):
        self.portrait().plotVariables(mode, save)

    def info(self):
        if self.best is None:
            self.update()

        row = {k: round(v, 5) if isinstance(v, float) else v for k, v in self.best.items()}
        return {'algorithm': self.name, 'f': self.score, 'stopped': self.optimizer.stopped, **row}

    # PRECISÃO CHEIA E VARIÁVEIS NUM CAMPO SÓ, AO CONTRÁRIO DO info(), QUE É A LINHA ACHATADA DO QUADRO
    def getBest(self):
        if self.best is None:
            self.update()

        return {'algorithm': self.name, 'f': self.score, 'runs': len(self.scores), 'stopped': self.optimizer.stopped, 'variables': self.best}
