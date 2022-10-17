"""utility measure time via a context-manager"""
__author__ = "Wout de Nolf"
__contact__ = "wout.de_nolf@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "17/10/2022"
__status__ = "development"

import time
from contextlib import contextmanager
import logging
logger = logging.getLogger(__name__)
from .file_io import get_isotime


class Timer():

    def __init__(self) -> None:
        self.start_time = get_isotime()
        self.t_start = time.perf_counter()
        self.rt_read = 0.0
        self.rt_rebin = 0.0
        self.rt_write = 0.0

    def print(self):
        print(f"Total execution time: {time.perf_counter()-self.t_start:.3f}s (of which read:{self.rt_read:.3f}s regrid:{self.rt_rebin:.3f} write:{self.rt_write:.3f}s)")

    @contextmanager
    def timeit_read(self):
        t0 = time.perf_counter()
        yield
        t1 = time.perf_counter()
        dt = t1 - t0
        self.rt_read += dt
        logger.info(f"HDF5 read time: {dt:.3f}s")

    @contextmanager
    def timeit_write(self):
        t0 = time.perf_counter()
        yield
        t1 = time.perf_counter()
        dt = t1 - t0
        self.rt_write += dt
        logger.info(f"HDF5 write time: {dt:.3f}s")

    @contextmanager
    def timeit_rebin(self):
        t0 = time.perf_counter()
        yield
        t1 = time.perf_counter()
        dt = t1 - t0
        self.rt_rebin += dt
        logger.info(f"Rebinning time: {dt:.3f}s")
