//------------------------------------------------------------------------------
// GB_mx_put_time: put the time back to the global workspace
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

double grbtime = 0, tic [2] = {0,0} ;

void GB_mx_put_time (void)
{

    // create a built-in array with the right size
    mxArray * grbresults_builtin = GB_mx_create_full (1, 2, GrB_FP64) ;

    // copy the time into the built-in array
    double *t = (double *) mxGetData (grbresults_builtin) ;

    t [0] = grbtime ;
    t [1] = 0 ;

    grbtime = 0 ;

    // put the built-in array into the global workspace, overwriting the
    // version that was already there
    mexPutVariable ("global", "GraphBLAS_results", grbresults_builtin) ;
}

