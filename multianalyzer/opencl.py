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
        self.allocate_buffers()
        self.set_kernel_arguments()

    def allocate_buffers(self):
        self.buffers["roicoll"] = None
        self.buffers["monitor"] = None
        self.buffers["arm"] = None
        self.buffers["center"] = cla.to_device(self.queue, self._psi)
        self.buffers["psi"] = cla.to_device(self.queue, self._center)
        self.buffers["rollx"] = cla.to_device(self.queue, self._rollx)
        self.buffers["rolly"] = cla.to_device(self.queue, self._rolly)
        self.buffers["out_signal"] = None
        self.buffers["out_norm"] = None

    def set_kernel_arguments(self):
        self.kernel_arguments["memset"] = OrderedDict([("num_crystal", self.NUM_CRYSTAL),
                                                       ("num_bin", None),
                                                       ("out_signal", self.buffers["out_signal"]),
                                                       ("out_norm", self.buffers["out_norm"])])
        self.kernel_arguments["integrate"] = OrderedDict([("roicoll", self.buffers["roicoll"]),
                                                          ("monitor", self.buffers["monitor"]),
                                                          ("arm", self.buffers["arm"]),
                                                          ("num_crystal", self.NUM_CRYSTAL),
                                                          ("num_frame", None),
                                                          ("num_roi", numpy.uint32(512)),
                                                          ("num_bin", None),
                                                          ("L", self.L),
                                                          ("L2", self.L2),
                                                          ("pixel", self.pixel),
                                                          ("center", self.buffers["center"].data),
                                                          ("tha", self._tha),
                                                          ("thd", self._thd)
                                                          ("psi", self.buffers["psi"].data),
                                                          ("rollx", self.buffers["rollx"].data),
                                                          ("rolly", self.buffers["rolly"].data),
                                                          ("resolution", None),
                                                          ("niter", 250),
                                                          ("phi_max", None),
                                                          ("tth_min", None),
                                                          ('tth_max', None),
                                                          ("dtth", None),
                                                          ("out_signal", self.buffers["out_signal"]),
                                                          ("out_norm", self.buffers["out_norm"])])

