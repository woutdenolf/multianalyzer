#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
##cython: linetrace=True

__author__ = "Jérôme KIEFFER"
__date__  = "08/10/2021"
__copyright__ = "2021, ESRF, France"
__licence__ = "MIT"

from libc.math cimport pi, sin, cos, atan2, tan, asin, acos, sqrt, isnan, fabs, NAN, floor
from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, \
                         int32_t, uint32_t, int64_t, uint64_t
ctypedef double float64_t
ctypedef float float32_t
from cython.parallel import prange
import numpy


cdef class MultiAnalyzer:
    """This class performes the reduction of roi-collection, taking into account 
    the angle between the different analyzers.

    All equations refer to doi:10.1107/S1600576721005288

    """
    cdef:
        public int NUM_CRYSTAL
        public double dr
        public double L, L2, pixel, _tha, _thd, sin_tha, cot_tha, cos_thd
        public double[::1] _center, _rollx, _rolly, _psi, cos_rx, sin_rx, cos_ry, sin_ry, _Lp, _Ln
    
    def __cinit__(self, L, L2, pixel, center, tha, thd, psi, rollx, rolly):
        "Performes the initialization of the data"
        self.NUM_CRYSTAL = 13
        self.dr = pi/180.
        self._center = numpy.empty(self.NUM_CRYSTAL, dtype=numpy.float64) # position of the center, par analyzer
        self._psi = numpy.empty(self.NUM_CRYSTAL, dtype=numpy.float64)    # Analyzer position on 2th arm
        self._rollx = numpy.empty(self.NUM_CRYSTAL, dtype=numpy.float64)  # mis-alignment, along x
        self._rolly = numpy.empty(self.NUM_CRYSTAL, dtype=numpy.float64)  # mis-alignment, along y
        self.cos_rx = numpy.empty(self.NUM_CRYSTAL, dtype=numpy.float64)
        self.sin_rx = numpy.empty(self.NUM_CRYSTAL, dtype=numpy.float64)
        self.cos_ry = numpy.empty(self.NUM_CRYSTAL, dtype=numpy.float64)
        self.sin_ry = numpy.empty(self.NUM_CRYSTAL, dtype=numpy.float64)
        self._Ln = numpy.empty(self.NUM_CRYSTAL, dtype=numpy.float64)
        self._Lp = numpy.empty(self.NUM_CRYSTAL, dtype=numpy.float64)

    def __dealloc(self):
        self._center = None
        self._psi = None
        self._rollx = None
        self._rolly = None
        self.cos_rx = None
        self.sin_rx = None
        self.cos_ry = None
        self.sin_ry = None
        self._Ln = None
        self._Lp = None

    @property
    def Lp(self):
        """Sample-analyzer distance at secondary diffraction point
        """
        return numpy.asarray(self._Lp)
    
    @property
    def Ln(self):
        "Sample-analyzer distance, crystal per crystal"
        return numpy.asarray(self._Ln)
    
    @property 
    def center(self):
        return numpy.asarray(self._center)
    
    @property 
    def psi(self):
        return numpy.rad2deg(self._psi)

    def __init__(self, L, L2, pixel, center, tha, thd, psi, rollx, rolly):
        """Constructor if the "mono-analyzer". 
        Works only on the middle crystal.
        
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
        """
        self.L = float(L)
        self.L2 = float(L2)
        self.pixel = float(pixel)
        self._center = numpy.ascontiguousarray(center, dtype=numpy.float64)
        self._tha = tha * self.dr
        if thd:
            self._thd = thd * self.dr
        else:
            self._thd = 2.0 * self._tha
        self._psi = numpy.ascontiguousarray(psi, dtype=numpy.float64) * self.dr
        self._rollx = numpy.ascontiguousarray(rollx, dtype=numpy.float64) * self.dr
        self._rolly = numpy.ascontiguousarray(rolly, dtype=numpy.float64) * self.dr
        self.sin_tha = sin(self._tha)
        self.cot_tha = 1/tan(self._tha)
        self.cos_thd = cos(self._thd)
        self.cos_rx = numpy.cos(self._rollx)
        self.sin_rx = numpy.sin(self._rollx)
        self.sin_ry = numpy.sin(self._rolly)
        self.cos_ry = numpy.cos(self._rolly)
        self._Ln = self.calc_Ln()
        self._Lp = self.calc_Lp()
    
    def calc_Ln(self):
        "Implementation of the Eq24"
        return (self.L/self.sin_tha)*(numpy.sin(self._psi) - numpy.sin(numpy.asarray(self._psi)-self._tha))
    
    def calc_Lp(self):
        "Variation on Eq26"
        return self._Ln * (numpy.asarray(self.cos_rx) - numpy.asarray(self.sin_rx)*self.sin_ry*self.cot_tha)
    
    cdef double _calc_zd(self, int idr, int ida) nogil:
        """Calculate the distance to the center along z
        
        :param idr: index of ROI 
        :param ida: index of analyzer <self.NUM_CRYSTAL
        :return: algebraic distance to the center of the detector
        """
        return self.pixel * (idr - self._center[ida]) # -> in unit

    cdef double _init_phi(self, double zd, double tth) nogil:
        """Approximative value of the azimuthal angle
        
        :param zd: distance to the center of the detector
        :param tth: 2theta position in rad.
        :return: an approximation of the azimuthal angle phi in rad.
        """
        return atan2(zd, (self.L+self.L2)*sin(tth))

    cdef double _calc_phi(self, int ida, double zd, double L3, double tth) nogil:
        """Implementation of Eq29 
        :param ida: index of analyzer
        :param zd: height of ROI
        :param L3: distance from analyzer (L'2) to detector
        :param tth: scattering angle 2theta
        :return: azimuthal angle phi, in radian, of ours !
        """
        cdef:
            double Lp = self._Lp[ida]
            double sin_tha = self.sin_tha
            double sin_rx  = self.sin_rx[ida]
            double cos_ry  = self.sin_ry[ida]
            double num, den
        num = zd + 2.0*L3*sin_tha*sin_rx*cos_ry
        den = (Lp+L3)*sin(tth)
        return asin(num/den)
    
    cdef double _calc_L3(self, int ida, double arm, double tth, double phi) nogil:
        """Implementation Eq28.
        
        Calculate the total distance from analyzer (at diffraction point) to detector
        
        :param ida: index analyzer
        :param arm: 2theta position of the arm (in radian)
        :param tth: diffracted angle radial (2theta)
        :param phi: diffracted angle azimuthal
        :return: analyzer-detector distance.
        """
        cdef:
            double cos_phi = cos(phi)
            double cos_tth = cos(tth)
            double sin_tth = sin(tth)

            double arm_d = arm - self._thd
            double cos_arm_d = cos(arm_d)
            double sin_arm_d = sin(arm_d)

            double arm_a_n = arm + self._psi[ida] - self._tha
            double cos_arm_a_n = cos(arm_a_n)
            double sin_arm_a_n = sin(arm_a_n)
            
            double sin_tha = self.sin_tha
            double  cos_rx = self.cos_rx[ida]
            double  sin_rx = self.sin_rx[ida]
            double  sin_ry = self.sin_ry[ida]
            double num, den
            
        num = self.L*self.cos_thd + self.L2 - self._Lp[ida]*(cos_arm_d*cos_tth + sin_arm_d*sin_tth*cos_phi) 
        den = cos_arm_d*(cos_tth         + 2.0 * sin_tha*(sin_arm_a_n*cos_rx        + cos_arm_a_n*sin_rx*sin_ry))\
            + sin_arm_d*(sin_tth*cos_phi + 2.0 * sin_tha*(sin_arm_a_n*sin_rx*sin_ry - cos_arm_a_n*cos_rx))
        return num/den

    cdef double _calc_tth(self, int ida, double arm, double phi) nogil:
        """Calculate the 2th from Eq 31: 
        
        resolution by development of cos(a-b)
        
        Nota: all angles are in radians
        
        :param arm: angular arm position
        :param phi: associated azimuthal angle
        :return: Corrected position in 2theta
        """
        cdef:
            double sin_tha = self.sin_tha
            double cos_rx = self.cos_rx[ida]
            double sin_rx = self.sin_rx[ida]
            double sin_ry = self.sin_ry[ida]
            double cos_ry = self.cos_ry[ida]
            
            double arm_a_n = arm +self._psi[ida] - self._tha
            double cos_arm_a = cos(arm_a_n)
            double sin_arm_a = sin(arm_a_n)
            double sin_phi = sin(phi)
            double cos_phi = cos(phi)
            double X, Y, Z
        X = sin_arm_a*cos_rx + cos_arm_a*sin_rx*sin_ry
        Y = (sin_arm_a*sin_rx*sin_ry - cos_arm_a*cos_rx) * cos_phi - sin_rx*cos_ry*sin_phi
        Z = -sin_tha
        # Nota technically this should be +/- 
        return atan2(Y, X) + acos(Z/sqrt(X*X+Y*Y))
    
    cdef double _refine(self, int idr, int ida, 
                        double arm, double resolution=1e-8, int niter=100, 
                        double phi_max=pi) nogil:
        """Refine with the angles in radians
        
        :param idr: index of ROI
        :param ida: index of analyzer
        
        :return scattering angle 2theta in radian, NaN if not converged or |phi|>max_phi
        """
        cdef:
            int i
            double zd, phi, tth_old, tth, L3
            
        zd = self._calc_zd(idr, ida)
        phi = self._init_phi(zd, arm) 
        tth_old = self._calc_tth(ida, arm, phi)
        for i in range(niter):
            L3 = self._calc_L3(ida, arm, tth_old, phi)
            phi = self._calc_phi(ida, zd, L3, tth_old)
            tth = self._calc_tth(ida, arm, phi)
            if fabs(tth-tth_old)>resolution:
                tth_old = tth
                tth = NAN
            else:
                break
        if fabs(phi) > phi_max:
            tth = NAN
        return tth

    def refine(self, int idr, int ida, 
               double arm, double resolution=1e-8, int niter=100, 
               double phi_max=90):
        """Refine the diffraction angle for an arm position any analyzer and any ROI"""
        return self._refine(idr, ida, arm*self.dr, resolution*self.dr, niter, phi_max*self.dr)/self.dr
    
    def integrate(self,
                  roicollection, 
                  double[::1] arm, 
                  double[::1] mon, 
                  double tth_min, 
                  double tth_max, 
                  double dtth, 
                  double phi_max):
        """Performess the integration of the ROIstack recorded at given angles on t
        
        :param roi_stack: stack of (nframes,NUM_CRYSTAL*numROI) with the recorded signal
        :param arm: 2theta position of the arm (in degrees)
        :param tth_min: start position of the histograms
        :param tth_max: End positon of the histogram
        :param dtth: bin size for the histogram
        :param phi_max: discard data with |phi| larger than this value
        :return: center of bins, histogram of signal and histogram of normalization
        """
        cdef:
            int nbin, nroi, frame, ida, idr, value, niter, idx, nframes = arm.shape[0]
            double[:, ::1] norm_b
            int64_t[:, ::1] signal_b
            double a, tth, nrm, resolution
            int32_t[:, :, ::1] roicoll = numpy.ascontiguousarray(roicollection, dtype=numpy.int32).reshape((nframes, self.NUM_CRYSTAL, -1))

        niter = 100
        tth_b = numpy.arange(tth_min, tth_max+dtth, dtth)
        tth_min -= dtth/2.
        tth_max += dtth/2
        nbin = tth_b.size
        norm_b = numpy.zeros((self.NUM_CRYSTAL, nbin), dtype=numpy.float64)
        signal_b = numpy.zeros((self.NUM_CRYSTAL, nbin), dtype=numpy.int64)
        assert mon.shape[0] == arm.shape[0], "mon shape matches arm"        
        nroi = roicoll.shape[2]
        resolution = 0.001*self.dr*dtth
        #switch to radians:
#         print(nframes, self.NUM_CRYSTAL, nroi)
        phi_max *= self.dr
        tth_min *= self.dr
        tth_max *= self.dr
        dtth *= self.dr
        with nogil:
            for ida in prange(self.NUM_CRYSTAL, schedule="dynamic"):
                for frame in range(nframes):
                    a = arm[frame]*self.dr
                    nrm = mon[frame]
                    for idr in range(nroi):
                        value = roicoll[frame, ida, idr]
                        tth = self._refine(idr, ida, a, resolution, niter, phi_max)
                        if (tth>=tth_min) and (tth<tth_max):
                            idx = <int>floor((tth - tth_min)/dtth)
#                             if (idx>=nbin):
#                                 with gil:
#                                     print(idx, tth, (tth - tth_min)/dtth)
                            norm_b[ida, idx] = norm_b[ida, idx] + nrm
                            signal_b[ida, idx] = signal_b[ida, idx] + value
        return numpy.asarray(tth_b), numpy.asarray(signal_b), numpy.asarray(norm_b)
