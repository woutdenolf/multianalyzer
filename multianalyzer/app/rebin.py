#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Multianalyzer data rebinning
#             https://github.com/kif/multianalyzer
#
#
#    Copyright (C) 2021-2021 European Synchrotron Radiation Facility, Grenoble, France
#
#    Authors: Jérôme Kieffer <Jerome.Kieffer@ESRF.eu>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""utility rebin multi-analyzer data"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "20/05/2022"
__status__ = "development"

import os
from argparse import ArgumentParser
from contextlib import contextmanager
import logging
import time
try:
    logging.basicConfig(level=logging.WARNING, force=True)
except ValueError:
    logging.basicConfig(level=logging.WARNING)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
import numpy
try:
    import hdf5plugin  # noqa
except ImportError:
    logger.debug("Unable to load hdf5plugin, backtrace:", exc_info=True)

try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
except ImportError:
    logger.debug("No socket opened for debugging. Please install rfoo")

from .. import _version
from .._multianalyzer import MultiAnalyzer
try:
    from ..opencl import OclMultiAnalyzer
except ImportError:
    OclMultiAnalyzer = None
from ..file_io import topas_parser, ID22_bliss_parser, save_rebin, all_entries, get_isotime


def parse():
    name = "id22rebin"
    description = """Rebin ROI-collection into useable powder diffraction patterns.
    """
    epilog = """This software is MIT-licenced and available from https://github.com/kif/multianalyzer"""
    usage = f"{name} [options] ROIcol.h5"

    version = f"{name} version {_version.version}"
    parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("-v", "--version", action='version', version=version)

    required = parser.add_argument_group('Required arguments')
    required.add_argument("args", metavar='FILE', type=str, nargs=1,
                        help="HDF5 file with ROI-collection")
    required.add_argument("-p", "--pars", metavar='FILE', type=str,
                          help="`topas` refinement file", required=True)

    optional = parser.add_argument_group('Optional arguments')
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output filename (in HDF5)")
    optional.add_argument("--entries", nargs="*", default=list(),
                           help="Entry names (aka scan names) in the input HDF5 file to process. It should be a `fscan`. "
                           "By default, the HDF5 is scanned and ALL `fscan` entries are processed.")
    optional.add_argument("-d", "--debug",
                        action="store_true", dest="debug", default=False,
                        help="switch to verbose/debug mode")
    optional.add_argument("-w", "--wavelength", type=float, default=None,
                        help="Wavelength of the incident beam (in Å). Default: use the one in `topas` file")
    optional.add_argument("-e", "--energy", type=float, default=None,
                        help="Energy of the incident beam (in keV). Replaces wavelength")

    subparser = parser.add_argument_group('Rebinning options')
    subparser.add_argument("-s", "--step", type=float, default=None,
                           help="Step size of the 2θ scale. Default: the step size of the scan of the arm")
    subparser.add_argument("-r", "--range", type=float, default=None, nargs=2,
                           help="2θ range in degree. Default: the scan of the arm + analyzer amplitude")
    subparser.add_argument("--phi", type=float, default=75,
                           help="φ_max: Maximum opening angle in azimuthal direction in degrees. Default: 75°")
    subparser.add_argument("--iter", type=int, default=250,
                           help="Maximum number of iteration for the 2theta convergence loop, default:100")
    subparser.add_argument("--startp", type=int, default=0,
                           help="Starting pixel on the detector, default:0")
    subparser.add_argument("--endp", type=int, default=1024,
                           help="End pixel on the detector to be considered, default:1024")
    subparser.add_argument("--pixel", type=float, default=75e-3,
                           help="Size of the pixel, default: 75e-3 mm")
    subparser.add_argument("--width", type=float, default=0.0,
                           help="Size of the beam-size on the sample, default from topas file: ~1 mm")
    subparser.add_argument("--delta2theta", type=float, default=0.0,
                           help="Resolution in 2θ, precision expected for 2 ROI being `width` appart on each side of the ROI of interest (disabled by default)")

    subparser = parser.add_argument_group('OpenCL options')
    subparser.add_argument("--device", type=str, default=None,
                           help="Use specified OpenCL device, comma separated (by default: Cython implementation)")

    options = parser.parse_args()

    if options.debug:
        logger.setLevel(logging.DEBUG)
        logging.root.setLevel(level=logging.DEBUG)

    return options


def rebin_result_generator(filename=None, entries=None, hdf5_data=None, output=None, timer=None, pars=None, device=None, debug=None, energy=None, wavelength=None,
               pixel=None, step=None, range=None, phi=None, width=None, delta2theta=None, iter=None, startp=None, endp=None):
    if not pars:
        raise ValueError("'pars' parameter is missing")
    if pixel is None:
        pixel = 75e-3
    if phi is None:
        phi = 75
    if width is None:
        width = 0.0
    if delta2theta is None:
        delta2theta = 0.0
    if startp is None:
        startp = 0
    if endp is None:
        endp = 1024
    if iter is None:
        iter = 250
    if timer is None:
        timer = Timer()
    if hdf5_data is None:
        output = output or os.path.splitext(filename)[0] + "_rebin.h5"
    else:
        if not output:
            raise ValueError("'output' parameter is missing")
    processed = all_entries(output)

    print(f"Load topas refinement file: {pars}")
    param = topas_parser(pars)
    # Ensure all units are consitent. Here lengths are in milimeters.
    L = param["L1"]
    L2 = param["L2"]

    # Angles are all given in degrees
    center = numpy.array(param["centre"])
    psi = numpy.rad2deg(param["offset"])
    rollx = numpy.rad2deg(param["rollx"])
    rolly = numpy.rad2deg(param["rolly"])

    # tha = hdf5_data["tha"]
    # thd = hdf5_data["thd"]
    tha = numpy.rad2deg(param["manom"])
    thd = numpy.rad2deg(param["mantth"])

    # Finally initialize the rebinning engine.
    if device and OclMultiAnalyzer:
        mma = OclMultiAnalyzer(L, L2, pixel, center, tha, thd, psi, rollx, rolly, device=device.split(","))
        print(f"Using device {mma.ctx.devices[0]}")
    else:
        mma = MultiAnalyzer(L, L2, pixel, center, tha, thd, psi, rollx, rolly)
        print("Using Cython+OpenMP")

    if hdf5_data is None:
        print(f"Read ROI-collection from HDF5 file: {filename}")
        with timer.timeit_read():
            hdf5_data = ID22_bliss_parser(filename, entries=entries, exclude_entries=processed)

    print(f"Processing {len(hdf5_data)} entries: {list(hdf5_data)}")

    for entry in hdf5_data:
        if entry in processed:
            logger.warning("Skip entry '%s' (already processed)", entry)
            continue

        roicol = hdf5_data[entry]["roicol"]
        arm = hdf5_data[entry]["arm"]
        mon = hdf5_data[entry]["mon"]
        if len(roicol)!=len(arm) or len(arm)!=len(mon):
            kept_points = min(len(roicol), len(arm), len(mon))
            roicol = roicol[:kept_points]
            arm = arm[:kept_points]
            mon = mon[:kept_points]
            logger.warning(f"Some arrays have different length, was the scan interrupted ? shrinking scan size: {kept_points} !")
        dtth = step or (abs(numpy.median(arm[1:] - arm[:-1])))
        if range:
            tth_min = range[0] if numpy.isfinite(range[0]) else  arm.min() + psi.min()
            tth_max = range[0] if numpy.isfinite(range[1]) else  arm.max() + psi.max()
        else:
            tth_min = arm.min() + psi.min()
            tth_max = arm.max() + psi.max()

        print(f"Rebin data from {filename}::{entry}")
        with timer.timeit_rebin():
            res = mma.integrate(roicol,
                                arm,
                                mon,
                                tth_min, tth_max, dtth=dtth,
                                iter_max=iter,
                                roi_min=startp,
                                roi_max=endp,
                                phi_max=phi,
                                width=width or param.get("wg", 0.0),
                                dtthw=delta2theta)

        if debug:
            numpy.savez("dump", res)

        if output:
            print(f"Save to {output}::{entry}")
            with timer.timeit_write():
                save_rebin(output, beamline="id22", name="id22rebin", topas=param, res=res, start_time=timer.start_time, entry=entry)
        yield res


def rebin_file(**kwargs):
    for _ in rebin_result_generator(**kwargs):
        pass


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


def main():
    options = vars(parse())
    filenames = options.pop("args")
    timer = Timer()
    for filename in filenames:
        rebin_file(filename=filename, timer=timer, **options)
    timer.print()


if __name__ == "__main__":
    main()
