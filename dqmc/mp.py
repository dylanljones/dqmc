# coding: utf-8
#
# This code is part of dqmc.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Multiprocessing tools."""

import time
import logging
import psutil
import concurrent.futures
from tqdm import tqdm

logger = logging.getLogger("dqmc")


def distribute(*args):
    """Distribute arguments to tuple of arguments."""
    def _is_multi(x):
        return hasattr(x, "__len__") and not isinstance(x, (tuple, str))

    # Check for argument lists and store max length
    num = 1
    for arg in args:
        if _is_multi(arg):
            num = max(num, len(arg))

    # Expand all arguments to equal length
    tmp = list()
    for arg in args:
        if _is_multi(arg):
            if len(arg) != num:
                raise ValueError(f"Parameter list {arg} does not contain "
                                 f"{num} arguments")
            item = arg
        else:
            item = [arg for _ in range(num)]
        tmp.append(item)

    # transpose argument list to set of arguments
    out = list()
    for i in range(num):
        out.append(tuple([arg[i] for arg in tmp]))
    return tuple(out)


class ProcessPool:

    def __init__(self, max_workers=None):
        if max_workers is None:
            max_workers = psutil.cpu_count(logical=True)
        logger.info("Using %s processes", max_workers)

        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers)
        self.jobs = list()

    @property
    def num_jobs(self):
        return len(self.jobs)

    def submit(self, fn, *args, callback=None):
        logger.info("Submitting job %s with args: %s", fn.__name__, args)

        job = self.executor.submit(fn, *args)
        if callback is not None:
            job.add_done_callback(callback)
        self.jobs.append(job)
        return job

    def map(self, fn, args):
        return self.executor.map(fn, *zip(*args))

    def distribute(self, fn, params):
        args = distribute(*params)
        return self.executor.map(fn, *zip(*args))

    def pdistribute(self, fn, params):
        args = distribute(*params)
        return list(tqdm(self.executor.map(fn, *zip(*args)), total=len(args)))

    def stop(self, wait=True):
        self.executor.shutdown(wait)

    def complete_all(self):
        t0 = time.perf_counter()

        logger.info("Waiting for %s jobs to complete", self.num_jobs)
        results = [f.result() for f in concurrent.futures.as_completed(self.jobs)]

        t = time.perf_counter() - t0
        logger.info("--------------------------")
        logger.info("Total time:   %10.2f s", t)
        logger.info("Process time: %10.2f s", t/self.num_jobs)
        logger.info("--------------------------")

        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def run_parallel(fn, params, max_workers=None):
    with ProcessPool(max_workers) as executor:
        return executor.distribute(fn, params)


def prun_parallel(fn, params, max_workers=None):
    with ProcessPool(max_workers) as executor:
        return executor.pdistribute(fn, params)
