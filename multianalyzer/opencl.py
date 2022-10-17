import os
import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict
import numpy
import pyopencl
from pyopencl import array as cla


class OclMultiAnalyzer:
    NUM_CRYSTAL = numpy.int32(13)

    def __init__(self, L, L2, pixel, center, tha, thd, psi, rollx, rolly, device=None):
        """Constructor if the "Multi-analyzer" working on OpenCL 
        
        Nota:
        Internally, all calculation are performed in radians.
        All distances must use the same unit (m, mm or inches)
        
        :param L:  distance from the sample to the analyzer
        :param L2: distance from the analyzer to the detector 
        :param pixel: pixel size
        :param center: position of the center on the detector (in pixel)
        :param tha: acceptance angle of the analyzer crystal(°)
        :param thd: diffraction angle of the analyzer crystal(°) 2x tha
        :param psi: Offset of angles (in 2th) of the analyzer crystals
        :param rollx: mis-orientation of the analyzer along x (°)
        :param rolly: mis-orientation of the analyzer along y (°)
        :param device: 2-tuple with the device
        """
        self.L = numpy.float64(L)
        self.L2 = numpy.float64(L2)
        self.pixel = numpy.float64(pixel)
        self._center = numpy.ascontiguousarray(center, dtype=numpy.float64)
        self._tha = numpy.deg2rad(tha)
        if thd:
            self._thd = numpy.deg2rad(thd)
        else:
            self._thd = 2.0 * self._tha

        assert len(psi) == self.NUM_CRYSTAL, "psi has the right size"
        assert len(rollx) == self.NUM_CRYSTAL, "rollx has the right size"
        assert len(rolly) == self.NUM_CRYSTAL, "rolly has the right size"

        self._psi = numpy.deg2rad(psi, dtype=numpy.float64)
        self._rollx = numpy.deg2rad(rollx, dtype=numpy.float64)
        self._rolly = numpy.deg2rad(rolly, dtype=numpy.float64)

        if device:
            self.ctx = pyopencl.create_some_context(answers=[str(i) for i in device])
        else:
            self.ctx = pyopencl.create_some_context(interactive=True)

        self.queue = pyopencl.CommandQueue(self.ctx)
        self.kernel_arguments = OrderedDict()
        self.buffers = {}
        self.kernel_arguments = {}
        self.prg = None
        self.allocate_buffers()
        self.set_kernel_arguments()
        self.compile_kernel()

        # Used only during multi-pass integration
        self.tth_b = None
        self.shape = None
        self.arm = None
        self.mon = None

    def allocate_buffers(self):
        self.buffers["roicoll"] = None
        self.buffers["monitor"] = None
        self.buffers["arm"] = None
        self.buffers["center"] = cla.to_device(self.queue, self._center)
        self.buffers["psi"] = cla.to_device(self.queue, self._psi)
        self.buffers["rollx"] = cla.to_device(self.queue, self._rollx)
        self.buffers["rolly"] = cla.to_device(self.queue, self._rolly)
        self.buffers["out_signal"] = None
        self.buffers["out_norm"] = None

    def set_kernel_arguments(self):
        self.kernel_arguments["integrate"] = OrderedDict([("roicoll", self.buffers["roicoll"]),
                                                          ("monitor", self.buffers["monitor"]),
                                                          ("arm", self.buffers["arm"]),
                                                          ("num_crystal", self.NUM_CRYSTAL),
                                                          ("num_frame", None),
                                                          ("num_row", numpy.uint32(512)),
                                                          ("num_col", numpy.uint32(1)),
                                                          ("columnorder", numpy.uint8(0)),
                                                          ("num_bin", None),
                                                          ("L", self.L),
                                                          ("L2", self.L2),
                                                          ("pixel", self.pixel),
                                                          ("center", self.buffers["center"].data),
                                                          ("tha", self._tha),
                                                          ("thd", self._thd),
                                                          ("psi", self.buffers["psi"].data),
                                                          ("rollx", self.buffers["rollx"].data),
                                                          ("rolly", self.buffers["rolly"].data),
                                                          ("resolution", None),
                                                          ("niter", 250),
                                                          ("phi_max", None),
                                                          ("roi_min", None),
                                                          ("roi_max", None),
                                                          ("tth_min", None),
                                                          ('tth_max', None),
                                                          ("dtth", None),
                                                          ("width", numpy.int32(0)),
                                                          ("dtthw", None),
                                                          ("out_signal", self.buffers["out_signal"]),
                                                          ("out_norm", self.buffers["out_norm"]),
                                                          ("do_debug", numpy.uint8(0)),
                                                          ("cycles", None),
                                                          ("local", None)])

    def compile_kernel(self):
        with open(os.path.join(os.path.dirname(__file__), "multianalyzer.cl"), "r") as r:
            src = r.read()
        self.prg = pyopencl.Program(self.ctx, src).build()

    def get_max_size(self):
        "return maximum allocation size for a single array"
        return self.ctx.devices[0].max_mem_alloc_size

    def set_shape(self, columnorder, nframes, num_col, num_row, **kw):
        if columnorder == 0:
            shape = (nframes, num_col, self.NUM_CRYSTAL, num_row)
        elif columnorder == 1:
            shape = (nframes, self.NUM_CRYSTAL, num_col, num_row)
        elif columnorder == 2:
            shape = (nframes, self.NUM_CRYSTAL, num_row, num_col)
        self.shape = shape
        return shape

    def integrate(self,
                  roicollection,
                  arm,
                  mon,
                  tth_min,
                  tth_max,
                  dtth,
                  num_row=512,
                  num_col=31,
                  columnorder=0,  # // 0: (column=31, channel=13, row=512), 1: (channel=13, column=31, row=512), 2: (channel=13, row=512, column=31)
                  phi_max=90.,
                  roi_min=0,
                  roi_max=512,
                  roi_step=1,
                  iter_max=250,
                  resolution=1e-3,
                  width=1,
                  dtthw=None
                  ):
        """Performess the integration of the ROIstack recorded at given angles on t
        
        :param roi_stack: stack of (nframes,NUM_CRYSTAL*numROI) with the recorded signal
        :param arm: 2theta position of the arm (in degrees)
        :param tth_min: start position of the histograms (in degrees)
        :param tth_max: End positon of the histogram (in degrees)
        :param dtth: bin size for the histogram (in degrees)
        :param phi_max: discard data with |phi| larger than this value (in degree)
        :param iter_max: maximum number of iteration in the 2theta convergence
        :param resolution: precision of the 2theta convergence in fraction of dtth
        :param width: width of the sample, same unit as pixels
        :param dtthw: Minimum precision expected for ROI being `width` appart, by default dtth
        :return: center of bins, histogram of signal and histogram of normalization, cycles per data-point
        """

        if roi_step and roi_step != 1:
            logger.warning("only roi_step=1 is supported in OpenCL")
        dtthw = dtthw or dtth

        do_debug = logger.getEffectiveLevel() <= logging.DEBUG
        nframes = arm.shape[0]
        shape = self.set_shape(columnorder, nframes, num_col, num_row)
        roicoll = numpy.ascontiguousarray(roicollection, dtype=numpy.int32).reshape(shape)
        mon = numpy.ascontiguousarray(mon, dtype=numpy.int32)
        tth_max += 0.5 * dtth
        tth_b = numpy.arange(tth_min, tth_max + (0.5 - numpy.finfo("float64").eps) * dtth, dtth)
        tth_min -= 0.5 * dtth
        nbin = tth_b.size
        assert mon.shape[0] == arm.shape[0], "monitor array shape matches the one from arm array "

        arm = numpy.deg2rad(arm)
        try:
            max_frames = min(int(int(self.ctx.devices[0].max_mem_alloc_size) / (numpy.dtype(numpy.int32).itemsize * self.NUM_CRYSTAL * num_row)),
                             nframes)
        except:
            max_frames = None
        logger.info(f"Allocate `out_norm` on device for {4*self.NUM_CRYSTAL*nbin/1e6}MB")
        self.buffers["out_norm"] = cla.empty(self.queue, (self.NUM_CRYSTAL, nbin), dtype=numpy.int32)
        logger.info(f"Allocate `out_signal` on device for {4*self.NUM_CRYSTAL*nbin*num_col/1e6}MB")
        self.buffers["out_signal"] = cla.empty(self.queue, (self.NUM_CRYSTAL, nbin, num_col), dtype=numpy.int32)
        evt = self.prg.memset(self.queue, (nbin, self.NUM_CRYSTAL), None,
                              numpy.uint32(self.NUM_CRYSTAL),
                              numpy.uint32(nbin),
                              numpy.uint32(num_col),
                              self.buffers["out_signal"].data,
                              self.buffers["out_norm"].data)
        if max_frames:
            shape = self.set_shape(columnorder, max_frames, num_col, num_row)
            logger.info(f"Allocate partial `roicoll` on device for {numpy.dtype(numpy.int32).itemsize*self.NUM_CRYSTAL*num_row*num_col*max_frames/1e6}MB")
            self.buffers["roicoll"] = cla.empty(self.queue, shape, dtype=numpy.int32)
            logger.info(f"Allocate partial  `mon` on device for {numpy.dtype(numpy.int32).itemsize*max_frames/1e6}MB")
            self.buffers["monitor"] = cla.empty(self.queue, (max_frames), dtype=numpy.int32)
            logger.info(f"Allocate partial  `arm` on device for {numpy.dtype(numpy.float64).itemsize*max_frames/1e6}MB")
            self.buffers["arm"] = cla.empty(self.queue, (max_frames), dtype=numpy.float64)
        else:
            logger.info(f"Allocate complete `roicoll` on device for {roicoll.nbytes/1e6}MB")
            self.buffers["roicoll"] = cla.to_device(self.queue, roicoll)
            logger.info(f"Allocate complete `mon` on device for {mon.nbytes/1e6}MB")
            self.buffers["monitor"] = cla.to_device(self.queue, mon)
            logger.info(f"Allocate complete `arm` on device for {arm.nbytes/1e6}MB")
            self.buffers["arm"] = cla.to_device(self.queue, arm)
        kwags = self.kernel_arguments["integrate"]
        kwags["roicoll"] = self.buffers["roicoll"].data
        kwags["monitor"] = self.buffers["monitor"].data
        kwags["arm"] = self.buffers["arm"].data
        kwags["out_norm"] = self.buffers["out_norm"].data
        kwags["out_signal"] = self.buffers["out_signal"].data
        kwags["num_frame"] = numpy.uint32(max_frames if max_frames else nframes)
        kwags["num_row"] = numpy.uint32(num_row)
        kwags["num_col"] = numpy.uint32(num_col)
        kwags["columnorder"] = numpy.uint8(columnorder)
        kwags["num_bin"] = numpy.uint32(nbin)
        kwags["resolution"] = numpy.deg2rad(resolution * dtth)
        kwags["niter"] = numpy.int32(iter_max)
        kwags["phi_max"] = numpy.deg2rad(phi_max)
        kwags["tth_min"] = numpy.deg2rad(tth_min)
        kwags['tth_max'] = numpy.deg2rad(tth_max)
        kwags["dtth"] = numpy.deg2rad(dtth)
        kwags["roi_min"] = numpy.uint32(max(roi_min, 0))
        kwags["roi_max"] = numpy.uint32(min(roi_max, num_row))
        kwags["local"] = pyopencl.LocalMemory(8 * num_row)
        kwags["width"] = numpy.int32(0.5 * width / self.pixel)
        kwags["dtthw"] = numpy.deg2rad(dtthw)
        if do_debug:
            logger.info(f"Allocate `cycles` on device for {self.NUM_CRYSTAL*num_row*nframes/1e6}MB")

            if max_frames:
                self.buffers["cycles"] = cla.empty(self.queue, (self.NUM_CRYSTAL, num_row, max_frames), dtype=numpy.uint8)
            else:
                self.buffers["cycles"] = cla.empty(self.queue, (self.NUM_CRYSTAL, num_row, nframes), dtype=numpy.uint8)
            cycles = numpy.zeros((self.NUM_CRYSTAL, num_row, nframes), dtype=numpy.uint8)
        else:
            self.buffers["cycles"] = cla.empty(self.queue, (1, 1, 1), dtype=numpy.uint8)
        kwags["do_debug"] = numpy.int32(do_debug)
        kwags["cycles"] = self.buffers["cycles"].data

        if do_debug:
            log = ["Parameters of the `integrate` kernel:"]
            i = 0
            for k, v in kwags.items():
                i += 1
                log.append(f"#{i}\t{k}: {v}")
            logger.debug("\n".join(log))
        if max_frames:
            for start in range(0, nframes, max_frames):
                stop = start + max_frames
                if stop < nframes:
                    sub_roicol = roicoll[start:stop,:,:]
                    sub_arm = arm[start:stop]
                    sub_mon = mon[start:stop]
                else:
                    stop = nframes
                    sub_roicol = numpy.empty(shape, dtype=numpy.int32)
                    sub_roicol[:stop - start, ...] = roicoll[start:stop, ...]
                    sub_arm = numpy.empty((max_frames), dtype=numpy.float64)
                    sub_arm[:stop - start] = arm[start:stop]
                    sub_mon = numpy.empty((max_frames), dtype=numpy.int32)
                    sub_mon[:stop - start] = mon[start:stop]
                self.buffers["roicoll"].set(sub_roicol)
                self.buffers["monitor"].set(sub_mon)
                self.buffers["arm"].set(sub_arm)
                evt = self.prg.integrate(self.queue, (num_row, stop - start, self.NUM_CRYSTAL), (num_row, 1, 1), *kwags.values())
                if do_debug:
                    cycles[:,:, start:stop] = self.buffers["cycles"].get()[:,:,:stop - start]
        else:
            evt = self.prg.integrate(self.queue, (num_row, nframes, self.NUM_CRYSTAL), (num_row, 1, 1), *kwags.values())
            if do_debug:
                cycles = self.buffers["cycles"].get()
        evt.wait()
        if do_debug:
            return tth_b, self.buffers["out_signal"].get(), self.buffers["out_norm"].get(), cycles
        else:
            return tth_b, self.buffers["out_signal"].get(), self.buffers["out_norm"].get()

#-----------------------------------------------
#    Multi pass implementation
#-----------------------------------------------
    def init_integrate(self,
                  max_frames,
                  arm,
                  mon,
                  tth_min,
                  tth_max,
                  dtth,
                  num_row=512,
                  num_col=31,
                  columnorder=0,  # // 0: (column=31, channel=13, row=512), 1: (channel=13, column=31, row=512), 2: (channel=13, row=512, column=31)
                  phi_max=90.,
                  roi_min=0,
                  roi_max=512,
                  roi_step=1,
                  iter_max=250,
                  resolution=1e-3,
                  width=1,
                  dtthw=None
                  ):
        """Initializes the integrator for the rebinning of several small chunks of data.
        
        :param max_frames: number of frames (max) per block
        :param arm: 2theta position of the arm (in degrees)
        :param tth_min: start position of the histograms (in degrees)
        :param tth_max: End positon of the histogram (in degrees)
        :param dtth: bin size for the histogram (in degrees)
        :param phi_max: discard data with |phi| larger than this value (in degree)
        :param iter_max: maximum number of iteration in the 2theta convergence
        :param resolution: precision of the 2theta convergence in fraction of dtth
        :param width: width of the sample, same unit as pixels
        :param dtthw: Minimum precision expected for ROI being `width` appart, by default dtth
        :return: nothing.
        """
        shape = self.set_shape(columnorder, max_frames, num_col, num_row)

        if roi_step and roi_step != 1:
            logger.warning("only roi_step=1 is supported in OpenCL")
        dtthw = dtthw or dtth
        self.mon = mon = numpy.ascontiguousarray(mon, dtype=numpy.int32)
        tth_max += 0.5 * dtth
        self.tth_b = tth_b = numpy.arange(tth_min, tth_max + (0.5 - numpy.finfo("float64").eps) * dtth, dtth)
        tth_min -= 0.5 * dtth
        nbin = tth_b.size
        assert mon.shape[0] == arm.shape[0], "monitor array shape matches the one from arm array "

        self.arm = arm = numpy.deg2rad(arm)
        logger.info(f"Allocate `out_norm` on device for {4*self.NUM_CRYSTAL*nbin/1e6:.3f} MB")
        self.buffers["out_norm"] = cla.empty(self.queue, (self.NUM_CRYSTAL, nbin), dtype=numpy.int32)
        logger.info(f"Allocate `out_signal` on device for {4*self.NUM_CRYSTAL*nbin*num_col/1e6:.3f} MB")
        self.buffers["out_signal"] = cla.empty(self.queue, (self.NUM_CRYSTAL, nbin, num_col), dtype=numpy.int32)
        evt = self.prg.memset(self.queue, (nbin, self.NUM_CRYSTAL), None,
                              numpy.uint32(self.NUM_CRYSTAL),
                              numpy.uint32(nbin),
                              numpy.uint32(num_col),
                              self.buffers["out_signal"].data,
                              self.buffers["out_norm"].data)
        logger.info(f"Allocate partial `roicoll` on device for {numpy.dtype(numpy.int32).itemsize*self.NUM_CRYSTAL*num_row*num_col*max_frames/1e6:.3f} MB")
        self.buffers["roicoll"] = cla.empty(self.queue, shape, dtype=numpy.int32)
        logger.info(f"Allocate partial  `mon` on device for {numpy.dtype(numpy.int32).itemsize*max_frames/1e6:.3f} MB")
        self.buffers["monitor"] = cla.empty(self.queue, (max_frames), dtype=numpy.int32)
        logger.info(f"Allocate partial  `arm` on device for {numpy.dtype(numpy.float64).itemsize*max_frames/1e6:.3f} MB")
        self.buffers["arm"] = cla.empty(self.queue, (max_frames), dtype=numpy.float64)
        kwags = self.kernel_arguments["integrate"]
        kwags["roicoll"] = self.buffers["roicoll"].data
        kwags["monitor"] = self.buffers["monitor"].data
        kwags["arm"] = self.buffers["arm"].data
        kwags["out_norm"] = self.buffers["out_norm"].data
        kwags["out_signal"] = self.buffers["out_signal"].data
        kwags["num_frame"] = numpy.uint32(max_frames)
        kwags["num_row"] = numpy.uint32(num_row)
        kwags["num_col"] = numpy.uint32(num_col)
        kwags["columnorder"] = numpy.uint8(columnorder)
        kwags["num_bin"] = numpy.uint32(nbin)
        kwags["resolution"] = numpy.deg2rad(resolution * dtth)
        kwags["niter"] = numpy.int32(iter_max)
        kwags["phi_max"] = numpy.deg2rad(phi_max)
        kwags["tth_min"] = numpy.deg2rad(tth_min)
        kwags['tth_max'] = numpy.deg2rad(tth_max)
        kwags["dtth"] = numpy.deg2rad(dtth)
        kwags["roi_min"] = numpy.uint32(max(roi_min, 0))
        kwags["roi_max"] = numpy.uint32(min(roi_max, num_row))
        kwags["local"] = pyopencl.LocalMemory(8 * num_row)
        kwags["width"] = numpy.int32(0.5 * width / self.pixel)
        kwags["dtthw"] = numpy.deg2rad(dtthw)
        self.buffers["cycles"] = cla.empty(self.queue, (1, 1, 1), dtype=numpy.uint8)
        kwags["do_debug"] = numpy.int32(0)
        kwags["cycles"] = self.buffers["cycles"].data
        evt.wait()

    def partial_integate(self, roicol_description, roicol_data):
        start = roicol_description.start
        stop = roicol_description.stop
        kwags = self.kernel_arguments["integrate"]
        num_row = kwags["num_row"]
        num_col = kwags["num_col"]
        max_frames = kwags["num_frame"]
        roicol = numpy.ascontiguousarray(roicol_data, numpy.int32).reshape((-1,) + self.shape[1:])
        if stop - start == max_frames:
            sub_arm = self.arm[start:stop]
            sub_mon = self.mon[start:stop]
            sub_roicol = roicol
        else:
            sub_roicol = numpy.empty(self.shape, dtype=numpy.int32)
            sub_roicol[:stop - start, ...] = roicol[:stop - start, ...]
            sub_arm = numpy.empty((max_frames), dtype=numpy.float64)
            sub_arm[:stop - start] = self.arm[start:stop]
            sub_mon = numpy.empty((max_frames), dtype=numpy.int32)
            sub_mon[:stop - start] = self.mon[start:stop]

        self.buffers["roicoll"].set(sub_roicol)
        self.buffers["monitor"].set(sub_mon)
        self.buffers["arm"].set(sub_arm)

        kwags["num_frame"] = numpy.uint32(stop - start)
        num_row = int(kwags["num_row"])
        logger.debug("Process frames %i to %i out of %i", start, stop, len(self.arm))
        # for k, v in kwags.items():
        #     print(k, v)
        evt = self.prg.integrate(self.queue, (num_row, stop - start, self.NUM_CRYSTAL), (num_row, 1, 1), *kwags.values())
        evt.wait()

    def finish_integrate(self):
        return self.tth_b, self.buffers["out_signal"].get(), self.buffers["out_norm"].get()

    def reset(self):
        "reset the integrator and zeros out all arrays"
        self.tth_b = None
        self.arm = None
        self.mon = None
        self.shape = None
