/*
 * Refine  
 */

kernel void  refine(
        double L, 
        double L2, 
        double pixel, 
        global double *center, 
        double tha, 
        double thd, 
        global double *psi, 
        global double *rollx, 
        global double *rolly
        global double *arm, 
        double resolution, 
        int niter, 
        double phi_max,
        ){
    
    int idr = get_global_id(0);
    int idf = get_global_id(1);
    int ida = get_global_id(2);
    
    double sin_tha = sin(tha);
    double cot_tha = 1.0/tan(tha);
    double cos_thd = cos(thd);
    double cos_rx = cos(rollx[ida]);
    double sin_rx = sin(rollx[ida]);
    double sin_ry = sin(rolly[ida]);
    double cos_ry = cos(rolly[ida]);
    
    double zd = pixel * (idr - center[ida]);
    
    
}