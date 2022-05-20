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
from ..file_io import topas_parser, ID22_bliss_parser, save_rebin, get_isotime


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
    optional.add_argument("--entry", type=str, default=None,
                           help="Entry name (aka scan name) in the input HDF5 file to process. It should be a `fscan`. "
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


def rebin(options):
    start_time = get_isotime()
    t_start = time.perf_counter()
    rt_read = 0.0
    rt_rebin = 0.0
    rt_write = 0.0
    
    print(f"Load topas refinement file: {options.pars}")
    param = topas_parser(options.pars)
    # Ensure all units are consitent. Here lengths are in milimeters.
    L = param["L1"]
    L2 = param["L2"]
    pixel = options.pixel

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
    if options.device and OclMultiAnalyzer:
        mma = OclMultiAnalyzer(L, L2, pixel, center, tha, thd, psi, rollx, rolly, device=options.device.split(","))
        print(f"Using device {mma.ctx.devices[0]}")
    else:
        mma = MultiAnalyzer(L, L2, pixel, center, tha, thd, psi, rollx, rolly)
        print("Using Cython+OpenMP")
    for infile in options.args:
        print(f"Read ROI-collection from  HDF5 file: {infile}")

        output = options.output or os.path.splitext(infile)[0] + "_rebin.h5"
        if os.path.exists(output):
            logger.warning(f"Output file {output} exist, removing!")
            os.unlink(output)

        t_start_reading = time.perf_counter()
        hdf5_data = ID22_bliss_parser(infile, entry=options.entry)
        t_end_reading = time.perf_counter()
        rt_read += t_end_reading - t_start_reading
        logger.info(f"HDF5 read time: {rt_read:.3f}s for {len(hdf5_data)} entries")

        for entry in hdf5_data:
            roicol = hdf5_data[entry]["roicol"]
            arm = hdf5_data[entry]["arm"]
            mon = hdf5_data[entry]["mon"]
            if len(roicol)!=len(arm) or len(arm)!=len(mon):
                kept_points = min(len(roicol), len(arm), len(mon))
                roicol = roicol[:kept_points]
                arm = arm[:kept_points]
                mon = mon[:kept_points]
                logger.warning(f"Some arrays have different length, was the scan interupted ? shrinking scan size: {kept_points} !")
            dtth = options.step or (abs(numpy.median(arm[1:] - arm[:-1])))
            if options.range:
                tth_min = options.range[0] if numpy.isfinite(options.range[0]) else  arm.min() + psi.min()
                tth_max = options.range[0] if numpy.isfinite(options.range[1]) else  arm.max() + psi.max()
            else:
                tth_min = arm.min() + psi.min()
                tth_max = arm.max() + psi.max()
            print(f"Rebin data from {infile}::{entry}")
            t_start_rebinning = time.perf_counter()
            res = mma.integrate(roicol,
                                arm,
                                mon,
                                tth_min, tth_max, dtth=dtth,
                                iter_max=options.iter,
                                roi_min=options.startp,
                                roi_max=options.endp,
                                phi_max=options.phi,
                                width=options.width or param.get("wg", 0.0),
                                dtthw=options.delta2theta)
            t_end_rebinning = time.perf_counter()
            logger.info(f"Rebinning time: {t_end_rebinning - t_start_rebinning:.3f}s")
            numpy.savez("dump", res)
            print(f"Save to {output}::{entry}")
            t_start_saving = time.perf_counter()
            save_rebin(output, beamline="id22", name="id22rebin", topas=param, res=res, start_time=start_time, entry=entry)
            t_end_saving = time.perf_counter()
            logger.info(f"HDF5 write time: {t_end_saving - t_start_saving:.3f}s")
            
            rt_rebin += t_end_rebinning - t_start_rebinning
            rt_write += t_end_saving - t_start_saving

    print(f"Total execution time: {time.perf_counter()-t_start:.3f}s (of which read:{rt_read:.3f}s regrid:{rt_rebin:.3f} write:{rt_write:.3f}s)")
    return res


def main():
    options = parse()
    rebin(options)


if __name__ == "__main__":
    main()
