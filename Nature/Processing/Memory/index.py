import json
import os
import numpy as np
from datetime import datetime


# CLASSE DA PERSISTÊNCIA OPCIONAL (memory=<pasta>) — USADA NO update() DE TODO MODELO PARA RETOMAR UM RUN
# ANTERIOR, SALVAR O ESTADO A CADA N GERAÇÕES E GRAVAR O MELHOR GLOBAL NO FIM.
class Memory:
    # Backup ativo + CONTINUAÇÃO (memory=path): a busca prossegue como se nunca tivesse parado.
    # state.npz guarda o estado COMPLETO de um run (população + auxiliares + estado do RNG + histórico do
    # Recorder + contador `done`) e NUNCA é apagado. best.json acumula o melhor global entre campanhas e
    # history.json registra um objeto por run com os params que a geraram. O estado é opaco: cada algoritmo
    # empacota (arrays, meta) e a Memory só persiste/valida (problema + modelo) e diz, pelo origin(), se a
    # chamada nova termina a campanha anterior ou estende com um ciclo próprio.
    EVERY = 10

    def __init__(self, path, model, params, problem, recorder=None):
        self.path     = path
        self.model    = model
        self.problem  = problem
        self.recorder = recorder
        self.maximize = problem.weight > 0
        self.params   = params
        self.best     = None
        self.score    = None
        self.genome   = None
        self.runs     = 0
        self.ended    = False
        self.since    = 0
        if path is None:
            return
        os.makedirs(path, exist_ok=True)
        self.read()

    def start(self):
        if self.path is None:
            return None
        f = os.path.join(self.path, 'state.npz')
        if not os.path.exists(f):
            return None
        d    = np.load(f, allow_pickle=False)
        meta = json.loads(str(d['meta']))
        if meta.get('model') != self.model or meta.get('names') != self.problem.names:
            return None  # estado de outro problema/algoritmo na mesma pasta: começa limpo
        self.ended = bool(meta.get('ended'))
        if self.recorder is not None:
            self.recorder.set({k: d[k] for k in d.files if k.startswith('rec')})
        return {k: d[k] for k in d.files if k != 'meta' and not k.startswith('rec')}, meta

    def origin(self, done):
        # Origem do ciclo, pela marca 'ended' que só o commit() grava (queda de energia nunca chega lá):
        # sem ela a campanha morreu no meio e a chamada nova só a termina, no cronograma original; com ela
        # a campanha fechou e a chamada nova é uma EXTENSÃO, ciclo próprio com cronograma reiniciado.
        return done if self.ended else 0

    def save(self, arrays, meta):
        if self.path is None:
            return
        self.since += 1
        if self.since < self.EVERY:
            return
        self.since = 0
        self.dump(arrays, meta)

    def commit(self, best, score, genome, stopped, arrays, meta):
        if self.path is None:
            return best, score
        self.runs += 1
        improved = self.score is None or (score > self.score if self.maximize else score < self.score)
        if improved:
            self.best, self.score, self.genome = best, score, [float(g) for g in genome]
        stamp  = datetime.now().isoformat(timespec='seconds')
        record = {'model': self.model, 'score': self.score, 'best': self.best, 'genome': self.genome,
                  'names': self.problem.names, 'stopped': stopped, 'runs': self.runs, 'params': self.params, 'updated': stamp}
        self.write('best.json', record)
        self.log({'run': self.runs, 'at': stamp, 'model': self.model, 'score': float(score),
                   'improved': bool(improved), 'stopped': stopped, 'best': best, 'params': self.params})
        self.dump(arrays, {**meta, 'ended': True})  # estado final PERSISTE e vai marcado como campanha fechada
        return self.best, self.score

    def read(self):
        f = os.path.join(self.path, 'best.json')
        if not os.path.exists(f):
            return
        data = json.load(open(f))
        if data.get('names') != self.problem.names:
            raise ValueError(f"memory '{self.path}' pertence a outro problema {data.get('names')}; use outra pasta.")
        self.best, self.score, self.genome, self.runs = data['best'], data['score'], data['genome'], data.get('runs', 0)

    def dump(self, arrays, meta):
        meta = {**meta, 'model': self.model, 'names': self.problem.names}
        if self.recorder is not None:
            arrays = {**arrays, **self.recorder.get()}  # histórico junto do estado: o plot sobrevive ao restart
        tmp        = os.path.join(self.path, 'state.tmp.npz')
        np.savez(tmp, meta=json.dumps(meta), **{k: np.asarray(v) for k, v in arrays.items()})
        os.replace(tmp, os.path.join(self.path, 'state.npz'))  # troca atômica: resiste a crash no meio

    def log(self, row):
        # Um registro por run no formato do best.json (params + solução daquele run). Lê, acrescenta e
        # reescreve inteiro: é um registro por campanha, cabe na memória e vira DataFrame no json_normalize.
        f   = os.path.join(self.path, 'history.json')
        old = json.load(open(f)) if os.path.exists(f) else []
        self.write('history.json', old + [row])

    def write(self, name, obj):
        tmp = os.path.join(self.path, name + '.tmp')
        json.dump(obj, open(tmp, 'w'), indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, os.path.join(self.path, name))  # troca atômica também no best.json

    def info(self):
        return {'path': self.path, 'model': self.model, 'runs': self.runs, 'score': self.score, 'best': self.best}
