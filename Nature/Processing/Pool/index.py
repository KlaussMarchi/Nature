import os
import warnings


# CLASSE QUE DECIDE COMO A POPULAÇÃO É AVALIADA (SERIAL, THREADS OU PROCESSOS) — ABERTA NO INÍCIO DO
# update() DE TODO MODELO E FECHADA NO finally.
class Pool:
    THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')

    def __init__(self, workers, backend, seed=None):
        self.workers = (os.cpu_count() or 1) if workers == -1 else int(workers)
        self.backend = backend
        self.seed    = seed
        self.pool    = None
        if backend == 'vector' and self.workers > 1:
            warnings.warn("backend='vector' avalia a população de uma vez; workers é ignorado.", stacklevel=3)
        if backend == 'process' and self.workers > 1 and not any(os.environ.get(v) for v in self.THREADS):
            # Cada worker herda um BLAS multithread e abre o próprio pool: com 16 workers numa objective de
            # numpy pesado medimos 10,3 s contra 1,0 s com as threads limitadas. Não dá para corrigir daqui —
            # o OpenBLAS lê essas variáveis ao inicializar, antes deste ponto.
            warnings.warn(f"backend='process' com BLAS multithread: exporte {self.THREADS[0]}=1 (e "
                          f'{self.THREADS[1]}=1) antes de abrir o Python, senão os workers disputam núcleos.',
                          stacklevel=3)

    def start(self):
        if self.workers <= 1 or self.backend == 'vector':
            return map
        if self.backend == 'thread':
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(max_workers=self.workers)
            return self.pool.map
        try:
            from multiprocess import Pool as ProcessPool
        except ImportError:
            from multiprocessing import Pool as ProcessPool
            warnings.warn("Instale 'multiprocess' para backend='process' em notebooks.", stacklevel=3)
        if self.seed is not None:
            warnings.warn("Reprodutibilidade não garantida com backend='process'.", stacklevel=3)
        self.pool = ProcessPool(self.workers)
        return self.pool.map

    def stop(self):
        if self.pool is None:
            return
        if hasattr(self.pool, 'shutdown'):
            self.pool.shutdown(wait=True)
        else:
            self.pool.close()
            self.pool.join()
        self.pool = None
