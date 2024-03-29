"""
istarmap.py for Python 3.8+
Source:
https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm/57364423#57364423
"""


import multiprocessing.pool as mp


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mp.Pool._get_tasks(func, iterable, chunksize)
    result = mp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mp.Pool.istarmap = istarmap