#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
##cython: linetrace=True

__author__ = "Jérôme KIEFFER"
__date__  = "18/10/2022"
__copyright__ = "2021-2022, ESRF, France"
__licence__ = "MIT"

import logging
logger = logging.getLogger(__name__)
from libc.math cimport pi, sin, cos, atan2, atan, tan, asin, acos, sqrt, isnan, fabs, NAN, floor, copysign, fmin, fmax, isfinite
                       
from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, \
                         int32_t, uint32_t, int64_t, uint64_t
#from libc.stdio cimport printf
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
        public int NUM_CRYSTAL, NUM_ROW, do_debug
        public float64_t dr
        public float64_t L, L2, pixel, _tha, _thd, sin_tha, cot_tha, cos_thd
        public float64_t[::1] _center, _rollx, _rolly, _psi, cos_rx, sin_rx, cos_ry, sin_ry, _Lp, _Ln
        public uint8_t[:, :, ::1] cycles #array to debug the number of cycles spent in refine
        #Arrays used when using multi-pass
        public float64_t[::1] out_tth, arm, mon
        public float64_t[:, ::1] out_norm
        public int32_t[:, :, ::1] out_signal
        public dict scalars
        public tuple shape
        
    def __cinit__(self, L, L2, pixel, center, tha, thd, psi, rollx, rolly):
        "Performes the initialization of the data"
        self.NUM_CRYSTAL = len(center)
        self.NUM_ROW = 512
        self.do_debug = logger.getEffectiveLevel()<=logging.DEBUG
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
        self.cycles = numpy.empty((self.NUM_CRYSTAL, self.NUM_ROW, 1) , dtype=numpy.uint8)
        #Those arrays are used only when working in multipass
        self.scalars = {}
        self.out_tth = None
        self.arm = None
        self.mon = None
        self.shape = None
        self.out_norm = None
        self.out_signal = None
        
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
        self.cycles = None
        #Those arrays are used only when working in multipass
        self.out_tth = None
        self.arm = None
        self.mon = None
        self.shape = None
        self.out_norm = None
        self.out_signal = None
        self.scalars = None
        
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
        
        assert len(psi) == self.NUM_CRYSTAL, "psi has the right size"
        assert len(rollx) == self.NUM_CRYSTAL, "rollx has the right size"
        assert len(rolly) == self.NUM_CRYSTAL, "rolly has the right size"
        
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
        #return self.L*(tan(self._tha/2.0)*numpy.sin(self._psi) + numpy.cos(self._psi))
    
    def calc_Lp(self):
        "Variation on Eq26"
        return self._Ln * (numpy.asarray(self.cos_rx) - numpy.asarray(self.sin_rx)*self.sin_ry*self.cot_tha)
        # return numpy.asarray(self._Ln) * (numpy.asarray(self.cos_rx)*self.sin_tha - numpy.asarray(self.sin_rx)*numpy.asarray(self.sin_ry)*self.cos_tha) / self.sin_tha
    
    cdef float64_t _calc_zd(self, int idr, int ida) nogil:
        """Calculate the distance to the center along z
        
        :param idr: index of ROI 
        :param ida: index of analyzer <self.NUM_CRYSTAL
        :return: algebraic distance to the center of the detector
        """
        return self.pixel * (idr - self._center[ida]) # -> in unit

    cdef float64_t _init_phi(self, float64_t zd, float64_t tth) nogil:
        """Approximative value of the azimuthal angle
        
        :param zd: distance to the center of the detector
        :param tth: 2theta position in rad.
        :return: an approximation of the azimuthal angle phi in rad.
        """
        return atan2(zd, (self.L+self.L2)*fabs(sin(tth)))

    cdef float64_t _init_sin_phi(self, float64_t zd, float64_t sin_tth) nogil:
        """Approximative value of the sine of the azimuthal angle
        
        :param zd: distance to the center of the detector
        :param sin_tth: sine of 2theta position
        :return: an approximation of the sine of azimuthal angle phi.
        """
        cdef: 
            float64_t L, tan_phi #, tan2_phi
        if sin_tth == 0.0:
            return 0.0
        else:
            L = self.L + self.L2
            tan_phi = zd / (L*fabs(sin_tth))
            return fmin(0.95, fmax(tan_phi, -.95))
            # tan2_phi = tan_phi * tan_phi 
            # return copysign(sqrt(tan2_phi/(1.0+tan2_phi)), zd) 

    cdef float64_t _calc_phi(self, int ida, float64_t zd, float64_t L3, float64_t tth) nogil:
        """Implementation of Eq29 
        :param ida: index of analyzer
        :param zd: height of ROI
        :param L3: distance from analyzer (L'2) to detector
        :param tth: scattering angle 2theta
        :return: azimuthal angle phi, in radian, of cours !
        """
        cdef:
            float64_t Lp = self._Lp[ida]
            float64_t sin_tha = self.sin_tha
            float64_t sin_rx  = self.sin_rx[ida]
            float64_t cos_ry  = self.sin_ry[ida]
            float64_t num, den, ratio, res
        num = zd + 2.0*L3*sin_tha*sin_rx*cos_ry
        den = (Lp+L3)*sin(tth)
        ratio = num/den
        res = asin(ratio) if fabs(ratio)<1.0 else copysign(0.5*pi, ratio)
        return res

    cdef float64_t _calc_sin_phi(self, int ida, float64_t zd, float64_t L3, float64_t sin_tth) nogil:
        """Implementation of Eq29, alternative implementation based on sin 
        :param ida: index of analyzer
        :param zd: height of ROI
        :param L3: distance from analyzer (L'2) to detector
        :param tth: scattering angle 2theta
        :return: sine of the azimuthal angle phi
        """
        cdef:
            float64_t Lp = self._Lp[ida]
            float64_t sin_tha = self.sin_tha
            float64_t sin_rx  = self.sin_rx[ida]
            float64_t cos_ry  = self.sin_ry[ida]
            float64_t num, den, ratio
        num = zd + 2.0*L3*sin_tha*sin_rx*cos_ry
        den = (Lp+L3) * sin_tth
        ratio = num/den
        return fmin(1.0, fmax(ratio, -1.0))

    
    cdef float64_t _calc_L3(self, int ida, float64_t arm, float64_t tth, float64_t phi) nogil:
        """Implementation Eq28.
        
        Calculate the total distance from analyzer (at diffraction point) to detector
        
        :param ida: index analyzer
        :param arm: 2theta position of the arm (in radian)
        :param tth: diffracted angle radial (2theta)
        :param phi: diffracted angle azimuthal
        :return: analyzer-detector distance.
        """
        cdef:
            float64_t cos_phi = cos(phi)
            float64_t cos_tth = cos(tth)
            float64_t sin_tth = sin(tth)

            float64_t arm_d = arm - self._thd
            float64_t cos_arm_d = cos(arm_d)
            float64_t sin_arm_d = sin(arm_d)

            float64_t arm_a_n = arm + self._psi[ida] - self._tha
            float64_t cos_arm_a_n = cos(arm_a_n)
            float64_t sin_arm_a_n = sin(arm_a_n)
            
            float64_t sin_tha = self.sin_tha
            float64_t  cos_rx = self.cos_rx[ida]
            float64_t  sin_rx = self.sin_rx[ida]
            float64_t  sin_ry = self.sin_ry[ida]
            float64_t num, den
            
        num = self.L*self.cos_thd + self.L2 - self._Lp[ida]*(cos_arm_d*cos_tth + sin_arm_d*sin_tth*cos_phi) 
        den = cos_arm_d*(cos_tth         + 2.0 * sin_tha*(sin_arm_a_n*cos_rx        + cos_arm_a_n*sin_rx*sin_ry))\
            + sin_arm_d*(sin_tth*cos_phi + 2.0 * sin_tha*(sin_arm_a_n*sin_rx*sin_ry - cos_arm_a_n*cos_rx))
        return num/den
    
    cdef float64_t _calc_L3_v2(self, int ida, float64_t sin_arm_d, float64_t cos_arm_d, float64_t sin_arm_a_n, float64_t cos_arm_a_n, 
                            float64_t sin_tth, float64_t cos_tth, float64_t sin_phi, float64_t cos_phi) nogil:
        """Implementation Eq28.
        
        Calculate the total distance from analyzer (at diffraction point) to detector
        
        :param ida: index analyzer
        :param arm: 2theta position of the arm (in radian)
        :param sin_tth, cos_tth: sine and cosine of diffracted angle radial (2theta)
        :param sin_phi, cos_phi: sine and cosine of diffracted angle azimuthal (phi)
        :return: analyzer-detector distance.
        """
        cdef:
            
            float64_t sin_tha = self.sin_tha
            float64_t  cos_rx = self.cos_rx[ida]
            float64_t  sin_rx = self.sin_rx[ida]
            float64_t  sin_ry = self.sin_ry[ida]
            float64_t num, den
            
        num = self.L*self.cos_thd + self.L2 - self._Lp[ida]*(cos_arm_d*cos_tth + sin_arm_d*sin_tth*cos_phi) 
        den = cos_arm_d*(cos_tth         + 2.0 * sin_tha*(sin_arm_a_n*cos_rx        + cos_arm_a_n*sin_rx*sin_ry))\
            + sin_arm_d*(sin_tth*cos_phi + 2.0 * sin_tha*(sin_arm_a_n*sin_rx*sin_ry - cos_arm_a_n*cos_rx))
        return num/den

    cdef float64_t _calc_tth(self, int ida, float64_t arm, float64_t phi) nogil:
        """Calculate the 2th from Eq 31: 
        
        resolution by development of cos(a-b)
        
        Nota: all angles are in radians
        
        :param arm: angular arm position
        :param phi: associated azimuthal angle
        :return: Corrected position in 2theta
        """
        cdef:
            float64_t sin_tha = self.sin_tha
            float64_t cos_rx = self.cos_rx[ida]
            float64_t sin_rx = self.sin_rx[ida]
            float64_t sin_ry = self.sin_ry[ida]
            float64_t cos_ry = self.cos_ry[ida]
            float64_t arm_n = arm + self._psi[ida]
            float64_t arm_a_n = arm_n - self._tha
            float64_t cos_arm_a = cos(arm_a_n)
            float64_t sin_arm_a = sin(arm_a_n)
            float64_t sin_phi = sin(phi)
            float64_t cos_phi = cos(phi)
            float64_t X, Y, Z, X2, Y2, Z2, D, D2, S1, S2, S
            
        # Solve `X cos(2th) + Y sin (2th) = Z` for 2th real 
        X = sin_arm_a*cos_rx + cos_arm_a*sin_rx*sin_ry
        Y = (sin_arm_a*sin_rx*sin_ry - cos_arm_a*cos_rx) * cos_phi - sin_rx*cos_ry*sin_phi
        Z = -sin_tha
        
        Y2 = Y*Y
        X2 = X*X
        D2 = X2 + Y2
        Z2 = Z*Z
        #Solution from mathematica:
        #cdef float64_t C, XZ, Z2, G
        # XZ = X*Z 
        # X2 = X*X
        # Z2 = Z*Z
        # Y2 = Y*Y
        # D2 = X2+Y2
        # D4 = sqrt(Y2*(D2-Z2))
        # G = Z*Y2 
        # S1 = atan2(XZ - D4, G + X*sqrt(Y2*(D2-Z2))/Y)
        # S2 = atan2(XZ + D4, G - X*sqrt(Y2*(D2-Z2))/Y)
        
        # Solution from wolfram alpha:
        # cdef float64_t D3, XpZ = X+Z
        # if XpZ:
        #     D3 = sqrt(D2 - Z2)
        #     S1 = 2.0 * atan2(Y - D3, XpZ)
        #     S2 = 2.0 * atan2(Y + D3, XpZ)
        # else:
        #     S1 = pi
        #     S2 = -2.0*atan2(X, Y)
        
        #Solution from cos(a-b) = cos(a)cos(b)-sin(a)sin(b) 
        D = sqrt(X*X+Y*Y)
        if Z > D:
            C = 0.0
        elif Z < -D:
            C = pi
        else:
            C = acos(Z/D)
        G = atan2(Y, X)
        S1 = G + C
        S2 = G - C
        
        #return the solution closest to the arm position
        S = S1 if fabs(arm_n-S1)<fabs(arm_n-S2) else S2
        return S

    cdef float64_t _calc_tth_v2(self, int ida, float64_t arm_n, float64_t sin_arm_a, float64_t cos_arm_a, float64_t sin_phi, float64_t cos_phi) nogil:
        """Calculate the 2th from Eq 31: 
        
        resolution by development of cos(a-b)
        
        Nota: all angles are in radians
        
        :param arm: angular arm position
        :param sin_phi: sine of azimuthal angle
        :param cos_phi: cosine of azimuthal angle
        :return: Corrected position in 2theta
        """
        cdef:
            float64_t sin_tha = self.sin_tha
            float64_t cos_rx = self.cos_rx[ida]
            float64_t sin_rx = self.sin_rx[ida]
            float64_t sin_ry = self.sin_ry[ida]
            float64_t cos_ry = self.cos_ry[ida]
            # float64_t arm_n = arm + self._psi[ida]
            # float64_t arm_a_n = arm_n - self._tha
            # float64_t cos_arm_a = cos(arm_a_n)
            # float64_t sin_arm_a = sin(arm_a_n)
            float64_t X, Y, Z, X2, Y2, D, D2, S1, S2, S
            
        # Solve `X cos(2th) + Y sin (2th) = Z` for 2th real 
        X = sin_arm_a*cos_rx + cos_arm_a*sin_rx*sin_ry
        Y = (sin_arm_a*sin_rx*sin_ry - cos_arm_a*cos_rx) * cos_phi - sin_rx*cos_ry*sin_phi
        Z = -sin_tha
        
        Y2 = Y*Y
        X2 = X*X
        D2 = X2 + Y2
        
        #Solution from cos(a-b) = cos(a)cos(b)-sin(a)sin(b) 
        # D = sqrt(D2)
        # if Z > D:
        #     C = 0.0
        # elif Z < -D:
        #     C = pi
        # else:
        #     C = acos(Z/D)
        # G = atan2(Y, X)
        # S1 = G + C
        # S2 = G - C
    
        #Solution from wolfram alpha    
        #===========================
        # solve X cos(t) + Y sin(t) = Z for t
        # t = 2 π n + π and Z = -X and n element Z
        # t = 2 (π n + tan^(-1)((Y - sqrt(X^2 + Y^2 - Z^2))/(X + Z))) and X + Z!=0 and X^2 + X Z + Y^2!=Y sqrt(X^2 + Y^2 - Z^2) and n element Z
        # t = 2 (π n + tan^(-1)((sqrt(X^2 + Y^2 - Z^2) + Y)/(X + Z))) and X + Z!=0 and Y (sqrt(X^2 + Y^2 - Z^2) + Y) + X^2 + X Z!=0 and n element Z
        # t = 2 π n - 2 tan^(-1)(X/Y) and Y!=0 and X^2 + Y^2!=0 and Z = -X and n element Z
        cdef float64_t D3, XpZ=X+Z, Z2=Z*Z, XZ=X*Z, D2XZ, D3Y
        
        S1 = S2 = 132.456
          
        if XpZ != 0.0:
            D3 = sqrt(D2 - Z2) if D2>Z2 else 0.0
            D2XZ = D2+XZ 
            D3Y = Y*D3
            if D2XZ != D3Y:
                S1 = 2.0 * atan((Y - D3) / XpZ)
            if D2XZ + D3Y != 0.0:
                S2 = 2.0 * atan((Y + D3) / XpZ)
        else:
            S1 = pi
            if (Y != 0.0) and (D2 != 0.0):
                S2 = -2.0*atan(X/Y)
        
        #return the solution closest to the arm position
        S = S1 if fabs(arm_n-S1)<fabs(arm_n-S2) else S2
        return S


    cdef float64_t _refine(self, int idr, int ida, 
                        float64_t arm, float64_t resolution=1e-8, int niter=250, 
                        float64_t sin_phi_max=1.0, int idf=-1) nogil:
        """Refine with the angles in radians
        
        :param idr: index of ROI
        :param ida: index of analyzer
        :param arm: position of the arm in 2theta in radians
        :param resolution: refine 2th to this resolution
        :param niter: max number of iterations (recorded in cycle if idf>=0
        :param sin_phi_max: maximal value of sine of phi before discarding point
        :param idf: index of frame to record the number of cycles
        :return scattering angle 2theta in radian, NaN if not converged or |phi|>max_phi
        """
        cdef:
            int i
            float64_t zd, phi, tth_old, tth, L3
            float64_t sin_phi, cos_phi, sin_tth, cos_tth
            float64_t arm_n = arm + self._psi[ida]
            float64_t arm_a_n = arm_n - self._tha
            float64_t arm_d = arm - self._thd

            # float64_t cos_arm_n = cos(arm_n)
            float64_t sin_arm_n = sin(arm_n)

            
            float64_t cos_arm_a = cos(arm_a_n)
            float64_t sin_arm_a = sin(arm_a_n)            
            
            float64_t cos_arm_d = cos(arm_d)
            float64_t sin_arm_d = sin(arm_d)

        zd = self._calc_zd(idr, ida)
        sin_phi = self._init_sin_phi(zd, sin_arm_n)
        cos_phi = sqrt(1.0-sin_phi*sin_phi)
        tth_old = self._calc_tth_v2(ida, arm_n, sin_arm_a, cos_arm_a, sin_phi, cos_phi)
        
        sin_tth = sin(tth_old)
        cos_tth = cos(tth_old)
        
        for i in range(niter):
            L3 = self._calc_L3_v2(ida, sin_arm_d, cos_arm_d, sin_arm_a, cos_arm_a, sin_tth, cos_tth, sin_phi, cos_phi)
            sin_phi = self._calc_sin_phi(ida, zd, L3, sin_tth)
            
            cos_phi = sqrt(1.0-sin_phi*sin_phi)
            tth = self._calc_tth_v2(ida, arm_n, sin_arm_a, cos_arm_a, sin_phi, cos_phi)

            # damp the update of this new value
            # if fabs(tth)<fabs(self._tha):
            #     tth = 0.2*tth + 0.8*tth_old
            
            sin_tth = sin(tth)
            cos_tth = cos(tth)
            
            if fabs(tth-tth_old)>resolution:
                tth_old = tth
                tth = NAN
            else:
                break
        if (fabs(sin_phi) >= sin_phi_max):
            tth = NAN
            i = 250
            
        if self.do_debug:
            if idf>=0:    
                if i+1 == niter:
                    i = 251
                elif not isfinite(L3):
                    i = 252
                elif not isfinite(sin_phi):
                    i = 253
                elif not isfinite(tth):
                    i = 254
                self.cycles[ida, idr, idf] += i 
                

        return tth


    def refine(self, int idr, int ida, 
               float64_t arm, float64_t resolution=1e-8, int niter=250, 
               float64_t phi_max=90):
        """Refine the diffraction angle for an arm position any analyzer and any ROI"""
        return self._refine(idr, ida, arm*self.dr, resolution*self.dr, niter, sin(phi_max*self.dr))/self.dr
    
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
                  float64_t[::1] arm, 
                  float64_t[::1] mon, 
                  float64_t tth_min, 
                  float64_t tth_max, 
                  float64_t dtth, 
                  int num_row = 512,
                  int num_col = 31,
                  int columnorder=0, #// 0: (column=31, channel=13, row=512), 1: (channel=13, column=31, row=512), 2: (channel=13, row=512, column=31)  
                  float64_t phi_max=90.,
                  int roi_min=0,
                  int roi_max=1024,
                  int roi_step=1,
                  int iter_max=250,
                  float64_t resolution=1e-3,
                  width=0,
                  dtthw=None):
        """Performs the integration of the ROIstack recorded at given angles on t
        
        :param roi_stack: stack of (nframes,NUM_CRYSTAL*numROI) with the recorded signal
        :param arm: 2theta position of the arm (in degrees)
        :param tth_min: start position of the histograms (in degrees)
        :param tth_max: End positon of the histogram (in degrees)
        :param dtth: bin size for the histogram (in degrees)
        :param num_row: number of rows, usually 512
        :param num_col: number of columns usially 31,
        :param columnorder: 0: (column=31, channel=13, row=512), 1: (channel=13, column=31, row=512), 2: (channel=13, row=512, column=31)  
        :param phi_max: discard data with |phi| larger than this value (in degree)
        :param roi_min: first row to be considered
        :param roi_max: Last row to be considered (excluded)
        :param roi_step: consider rows stepwise
        :param iter_max: maximum number of iteration in the 2theta convergence
        :param resolution: precision of the 2theta convergence in fraction of dtth
        :param width: unsupported for now, only works on OpenCL
        :param dtthw: unsupported for now, only works on OpenCL  
        :return: center of bins, histogram of signal and histogram of normalization, cycles per data-point
        """
        cdef:
            int nbin, idx_row, frame, ida, idr, value, idx, idx_col, nframes = arm.shape[0]
            float64_t[:, ::1] norm_b
            float64_t a, tth, nrm, sin_phi_max
            int32_t[:, :, ::1] signal_b
            int32_t[:, :, :, ::1] roicoll

        self.set_shape(columnorder, nframes, num_col, num_row)
        roicoll = numpy.ascontiguousarray(roicollection, dtype=numpy.int32).reshape(self.shape)
        
        if dtthw:
            logger.warning("Width/dtthw parameters are not supported in Cython implementation")
        tth_max += 0.5 * dtth
        tth_b = numpy.arange(tth_min, tth_max + (0.5-numpy.finfo("float64").eps) * dtth, dtth)
        tth_min -= 0.5 * dtth
        nbin = tth_b.size
        
        norm_b = numpy.zeros((self.NUM_CRYSTAL, nbin), dtype=numpy.float64)
        signal_b = numpy.zeros((self.NUM_CRYSTAL, nbin, num_col), dtype=numpy.int32)
        assert mon.shape[0] == arm.shape[0], "`monitor` array size matches the one from `arm` array "        
               
        roi_min, roi_max = min(roi_min, roi_max), max(roi_min, roi_max)
        roi_min = max(roi_min, 0)
        roi_max = min(roi_max, num_row)
        # this is a work-around to https://github.com/cython/cython/issues/1106
        roi_step = abs(roi_step)
        num_row = (roi_max-roi_min) // roi_step
        
        if self.do_debug:
            self.cycles = numpy.zeros((self.NUM_CRYSTAL, num_row, nframes), dtype=numpy.uint8)
        sin_phi_max = sin(phi_max*self.dr)
        #switch to radians:
        tth_min *= self.dr
        tth_max *= self.dr
        dtth *= self.dr
        resolution *= dtth
        with nogil:
            for ida in prange(self.NUM_CRYSTAL, schedule="dynamic"):
                for frame in range(nframes):
                    a = arm[frame]*self.dr
                    nrm = mon[frame]
                    idx_row = roi_min - roi_step
                    for idr in range(num_row):
                        idx_row = idx_row + roi_step
                        tth = self._refine(idx_row, ida, a, resolution, iter_max, sin_phi_max, frame)
                        if (tth>=tth_min) and (tth<tth_max):
                            idx = <int>floor((tth - tth_min)/dtth)
                            norm_b[ida, idx] = norm_b[ida, idx] + nrm
                            for idx_col in range(num_col):
                                if columnorder == 0:
                                    value = roicoll[frame, idx_col, ida, idx_row]
                                elif columnorder == 1:
                                    value = roicoll[frame, ida, idx_col, idx_row]
                                elif columnorder == 2:
                                    value = roicoll[frame, ida, idx_row, idx_col]
                                signal_b[ida, idx, idx_col] = signal_b[ida, idx, idx_col] + value
        if self.do_debug:
            return numpy.asarray(tth_b), numpy.asarray(signal_b), numpy.asarray(norm_b), numpy.asarray(self.cycles)
        else:
            return numpy.asarray(tth_b), numpy.asarray(signal_b), numpy.asarray(norm_b)

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
        if dtthw:
            logger.warning("Width/dtthw parameters are not supported in Cython implementation")
        tth_max += 0.5 * dtth
        self.out_tth = numpy.arange(tth_min, tth_max + (0.5-numpy.finfo("float64").eps) * dtth, dtth)
        tth_min -= 0.5 * dtth
        nbin = self.out_tth.size
        self.arm = arm
        self.mon = mon
        self.out_norm = numpy.zeros((self.NUM_CRYSTAL, nbin), dtype=numpy.float64)
        self.out_signal = numpy.zeros((self.NUM_CRYSTAL, nbin, num_col), dtype=numpy.int32)
        self.scalars["num_row"] = num_row
        self.scalars["num_col"] = num_col
        self.scalars["num_frame"] = max_frames
        self.scalars["nbin"] = nbin
        self.scalars["columnorder"]=columnorder
        self.scalars["phi_max"]=phi_max
        self.scalars["roi_min"]=roi_min
        self.scalars["roi_step"]=roi_step
        self.scalars["iter_max"]=iter_max
        #switch to radians:
        self.scalars["tth_max"] = tth_max*self.dr
        self.scalars["tth_min"] = tth_min*self.dr
        self.scalars["dtth"]= dtth*self.dr
        self.scalars["resolution"]=resolution*self.dr
        self.scalars["sin_phi_max"] = sin(phi_max*self.dr)

    def partial_integate(self, roicol_description, roicol_data):
        cdef:
            int32_t[:, :, :, ::1] roicol
            float64_t[::1] arm 
            float64_t[::1] mon 
            int start, stop, num_row, num_col, columnorder, roi_min, roi_max, roi_step, iter_max, idr, idx_row, idx, ida, nbin, value, num_frame, frame, idx_col
            float64_t phi_max, resolution, tth_min, tth_max, dtth
            float64_t a, tth, nrm, sin_phi_max

        start = roicol_description.start
        stop = roicol_description.stop
        num_frame = stop - start
        num_row = self.scalars["num_row"]
        num_col = self.scalars["num_col"]
        max_frames = self.scalars["num_frame"]
        nbin = self.scalars["nbin"]
        tth_max = self.scalars["tth_max"]
        tth_min = self.scalars["tth_min"]
        dtth = self.scalars["dtth"]
        columnorder = self.scalars["columnorder"]
        phi_max = self.scalars["phi_max"]
        roi_min  = self.scalars["roi_min"]
        roi_step  = self.scalars["roi_step"]
        resolution  = self.scalars["resolution"]
        iter_max = self.scalars["iter_max"]
        sin_phi_max = self.scalars["sin_phi_max"]
        roicol = numpy.ascontiguousarray(roicol_data, numpy.int32).reshape((-1,) + self.shape[1:])
        arm = self.arm[start: stop]
        mon = self.mon[start: stop]
        assert self.out_signal.shape[0] == self.NUM_CRYSTAL, "self.out_signal.shape[0] == self.NUM_CRYSTAL" 
        assert self.out_norm.shape[0] == self.NUM_CRYSTAL, "self.out_norm.shape[0] == self.NUM_CRYSTAL"
        assert self.out_signal.shape[1] == nbin, "self.out_signal.shape[1] == nbin" 
        assert self.out_norm.shape[1] == nbin, "self.out_norm.shape[1] == nbin"
        assert self.out_signal.shape[2] == num_col, "self.out_signal.shape[2] == num_col"        
        logger.debug("Process frames %i to %i out of %i", start, stop, len(self.arm))
        with nogil:
            for ida in prange(self.NUM_CRYSTAL, schedule="dynamic"):
        # for ida in range(self.NUM_CRYSTAL):
                for frame in range(num_frame):
                    a = arm[frame]*self.dr
                    nrm = mon[frame]
                    idx_row = roi_min - roi_step
                    for idr in range(num_row):
                        idx_row = idx_row + roi_step
                        tth = self._refine(idx_row, ida, a, resolution, iter_max, sin_phi_max, frame)
                        if (tth>=tth_min) and (tth<tth_max):
                            idx = <int>floor((tth - tth_min)/dtth)
                            self.out_norm[ida, idx] = self.out_norm[ida, idx] + nrm
                            for idx_col in range(num_col):
                                if columnorder == 0:
                                    value = roicol[frame, idx_col, ida, idx_row]
                                elif columnorder == 1:
                                    value = roicol[frame, ida, idx_col, idx_row]
                                elif columnorder == 2:
                                    value = roicol[frame, ida, idx_row, idx_col]
                                if 0<=value<65530:
                                    self.out_signal[ida, idx, idx_col] = self.out_signal[ida, idx, idx_col] + value

    def finish_integrate(self):
        return numpy.asarray(self.out_tth), numpy.asarray(self.out_signal), numpy.asarray(self.out_norm)

    def reset(self):
        "reset the integrator and zeros out all arrays"
        self.out_tth = None
        self.arm = None
        self.mon = None
        self.shape = None
        self.out_norm = None
        self.out_signal = None
        self.scalars = {}
