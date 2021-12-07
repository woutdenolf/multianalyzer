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
__date__ = "07/12/2021"
__status__ = "development"

from argparse import ArgumentParser
import logging
try:
    logging.basicConfig(level=logging.WARNING, force=True)
except ValueError:
    logging.basicConfig(level=logging.WARNING)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
try:
    import hdf5plugin  # noqa
except ImportError:
    logger.debug("Unable to load hdf5plugin, backtrace:", exc_info=True)

try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
except ImportError:
    logger.debug("No socket opened for debugging. Please install rfoo")


from .._multianalyzer import MultiAnalyzer
from ..file_io import topas_parser, ID22_bliss_parser, Nexus


def main():
    
    description = """Rebin ROI-collection into useable powder diffraction patterns.
    """
    epilog = """ This software is MIT licenced and availanle from https://github.com/kif/multianalyzer"""
    usage = """id22rebin [options] ROIcol.h5"""
    
    from .._version import version
    print(version)
    version = f"id22rebin version {version}" 
    parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
    required = parser.add_argument_group('Required arguments')
    required.add_argument("-p", "--pars", metavar='FILE', type=str,
                          help="Topas refinement file", required=True)
    required.add_argument("args", metavar='FILE', type=str, nargs='1',
                        help="HDF5 file with ROI-collection")
    parser = parser.add_argument_group('Optionnal arguments')
    parser.add_argument("-v", "--version", action='version', version=version)
    parser.add_argument("-d", "--debug",
                        action="store_true", dest="debug", default=False,
                        help="switch to verbose/debug mode")
    parser.add_argument("-w", "--wavelength", type=float, default=None,
                        help="Wavelength of the incident beam (in Å). Default: use the one in `topas` file")
    parser.add_argument("-e", "--energy", type=float, default=None,
                        help="Energy of the incident beam (in keV)")

    parser.add_argument("-e", "--energy", type=float, default=None,
                        help="Energy of the incident beam (in keV)")
    options = parser.parse_args()

    if options.debug:
        # pyFAI.logger.setLevel(logging.DEBUG)
        pass
    
    #
    # pyFAI.benchmark.run(number=options.number,
    #                     repeat=options.repeat,
    #                     memprof=options.memprof,
    #                     max_size=options.size,
    #                     do_1d=options.onedim,
    #                     do_2d=options.twodim,
    #                     devices=devices)
    #
    # if pyFAI.benchmark.pylab is not None:
    #     pyFAI.benchmark.pylab.ion()
    input("Enter to quit")


if __name__ == "__main__":
    main()