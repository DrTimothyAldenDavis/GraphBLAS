//------------------------------------------------------------------------------
// gbmx_set_double_scalar: set the value of a MATLAB scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The mxArray *mxscalar must have been created with mxCreateDoubleScalar.

#include "gb_interface.h"

void gbmx_set_double_scalar (mxArray *mxscalar, double value)
{ 
    double *p = (double *) mxGetData (mxscalar) ;
    (*p) = value ;
}

