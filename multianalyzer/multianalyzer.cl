
/*
 * Set of function to calculate tth !
 */

double calc_L3(double L,
               double L2,
               double Lp,
               double sin_tha,
               double cos_thd,
               double cos_rx,
               double sin_rx,
               double sin_ry,
               double sin_arm_d, 
               double cos_arm_d, 
               double sin_arm_a_n, 
               double cos_arm_a_n, 
               double sin_tth, 
               double cos_tth, 
               double sin_phi, 
               double cos_phi)
{//Implementation Eq28.
    
    double num, den;
    
    num = L*cos_thd + L2 - Lp*(cos_arm_d*cos_tth + sin_arm_d*sin_tth*cos_phi); 
    den = cos_arm_d*(cos_tth         + 2.0 * sin_tha*(sin_arm_a_n*cos_rx        + cos_arm_a_n*sin_rx*sin_ry))
        + sin_arm_d*(sin_tth*cos_phi + 2.0 * sin_tha*(sin_arm_a_n*sin_rx*sin_ry - cos_arm_a_n*cos_rx));
    return num/den;
}
        
        

double calc_tth(double sin_tha,
                double cos_rx,
                double sin_rx,
                double sin_ry,
                double cos_ry,
                double arm_n, 
                double sin_arm_a_n, 
                double cos_arm_a_n, 
                double sin_phi, 
                double cos_phi)
{ //Search for t in Xcos(t)+Ysin(t)=Z
     
    
    double X = sin_arm_a_n*cos_rx + cos_arm_a_n*sin_rx*sin_ry;
    double Y = (sin_arm_a_n*sin_rx*sin_ry - cos_arm_a_n*cos_rx) * cos_phi - sin_rx*cos_ry*sin_phi;
    double Z = -sin_tha;
    
    double Y2 = Y*Y;
    double X2 = X*X;
    double D2 = X2 + Y2;
    
    // Solution from wolfram alpha    
    double D3, XpZ=X+Z, Z2=Z*Z, XZ=X*Z, D2XZ, D3Y;
    
    double S1 = 132.456, S2=123.456;
      
    if (XpZ != 0.0){
        D3 = sqrt(D2 - Z2);
        D2XZ = D2+XZ;
        D3Y = Y*D3;
        if (D2XZ != D3Y){
            S1 = 2.0 * atan((Y - D3) / XpZ);
        }
        if (D2XZ + D3Y != 0.0){
            S2 = 2.0 * atan((Y + D3) / XpZ);
        }
    }
    else{
        S1 = M_PI_F;
        if ((Y != 0.0) && (D2 != 0.0)){
            S2 = -2.0*atan(X/Y);
        }
    }
    return (fabs(arm_n-S1)<fabs(arm_n-S2))?S1:S2;
}


double calc_sin_phi(double Lp, double sin_tha, double sin_rx, double cos_ry, double zd, double L3, double sin_tth)
{//Implementation of Eq29
 
    double num, den;
    num = zd + 2.0*L3*sin_tha*sin_rx*cos_ry;
    den = (Lp+L3) * sin_tth;
    return clamp(num/den, -1.0, 1.0);
}


double refine_tth(
        double L, 
        double L2, 
        double pixel, 
        global double *d_arm,
        global double *center, 
        double tha, 
        double thd, 
        global double *psi, 
        global double *rollx, 
        global double *rolly,
        double resolution, 
        int niter, 
        double phi_max
){
    size_t idr = get_global_id(0);
    size_t idf = get_global_id(1);
    size_t ida = get_global_id(2);
    
    double sin_tha = sin(tha);
    double cot_tha = 1.0/tan(tha);
    double cos_thd = cos(thd);
    double cos_rx = cos(rollx[ida]);
    double sin_rx = sin(rollx[ida]);
    double sin_ry = sin(rolly[ida]);
    double cos_ry = cos(rolly[ida]);
    
    double arm = d_arm[idf];
    double arm_n = arm + psi[ida];
    double arm_a_n = arm_n - tha;
    double arm_d = arm - thd;

    //double cos_arm_n = cos(arm_n);
    double sin_arm_n = sin(arm_n);
    
    double cos_arm_a_n = cos(arm_a_n);
    double sin_arm_a_n = sin(arm_a_n);           
    
    double cos_arm_d = cos(arm_d);
    double sin_arm_d = sin(arm_d);

    double zd = pixel * (idr - center[ida]);
 
    double sin_phi = 0.0;
    if (sin_arm_n!=0){
        double LL = L + L2;
        double tan_phi = zd / (LL*fabs(sin_arm_n));
        double tan2_phi = tan_phi * tan_phi;
        sin_phi = copysign(sqrt(tan2_phi/(1.0+tan2_phi)), zd);
    }
    double cos_phi = sqrt(1.0-sin_phi*sin_phi);
            
    double tth, tth_old = calc_tth(sin_tha, cos_rx, sin_rx, sin_ry, cos_ry, arm_n, sin_arm_a_n, cos_arm_a_n, sin_phi, cos_phi);
    
    double sin_tth = sin(tth_old);
    double cos_tth = cos(tth_old);
    double L3;
    double Ln = L*(sin(psi[ida]) - sin(psi[ida]-tha))/sin_tha; //Eq24
    double Lp = Ln*(cos_rx - sin_rx*sin_ry*cot_tha); //Eq26
    for (int i=0; i<niter; i++){
        L3 = calc_L3(L, L2, Lp, sin_tha, cos_thd, cos_rx, sin_rx, sin_ry, sin_arm_d, cos_arm_d, sin_arm_a_n, cos_arm_a_n, sin_tth, cos_tth, sin_phi, cos_phi);
        sin_phi = calc_sin_phi( Lp,  sin_tha,  sin_rx,  cos_ry,  zd,  L3,  sin_tth);
        cos_phi = sqrt(1.0-sin_phi*sin_phi);
        tth = calc_tth(sin_tha, cos_rx, sin_rx, sin_ry, cos_ry, arm_n, sin_arm_a_n, cos_arm_a_n, sin_phi, cos_phi);
        sin_tth = sin(tth);
        cos_tth = cos(tth);
        
        if (fabs(tth-tth_old)>resolution){
            tth_old = tth;
            tth = NAN;
        }
        else{
            break;
        }
    }
    return tth;
}
/*
 * Integrate  
 */


kernel void  integrate(
        global int *roicoll,
        global int *monitor,
        global double *d_arm,
        uint num_crystal,
        uint num_frame,
        uint num_roi, 
        uint num_bin,
        double L, 
        double L2, 
        double pixel, 
        global double *center, 
        double tha, 
        double thd, 
        global double *psi, 
        global double *rollx, 
        global double *rolly,
        double resolution, 
        int niter, 
        double phi_max,
        double tth_min,
        double tth_max,
        double dtth,
        global int *out_signal,
        global int *out_norm
        ){
    
    size_t idr = get_global_id(0);
    size_t idf = get_global_id(1);
    size_t ida = get_global_id(2);
    
    if ((idr>=num_roi) || (idf>=num_frame) || (ida>=num_roi)){
        return;
    }
    
    double tth = refine_tth( L, L2, pixel, d_arm, center, tha, thd, psi, rollx, rolly, resolution, niter, phi_max );
    if ((tth>=tth_min) && (tth<tth_max)){
        int nrm = monitor[idf];
        int value = roicoll[idf*num_roi*num_crystal + ida*num_roi + idr];
        int idx = convert_int_rtn((tth - tth_min)/dtth);
        size_t pos = num_bin*ida + idx;
        atomic_add(&out_signal[pos], value);
        atomic_add(&out_norm[pos], nrm);
    }
    
    
}