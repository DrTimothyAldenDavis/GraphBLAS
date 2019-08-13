//------------------------------------------------------------------------------
// GB_matlab_helper.h: helper functions for MATLAB interface
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// These functions are only used by the MATLAB interface for
// SuiteSparse:GraphBLAS.

#ifndef GB_MATLAB_HELPER_H
#define GB_MATLAB_HELPER_H

#include "GB.h"

void GB_matlab_helper1      // convert zero-based indices to one-based
(
    double *I_double,       // output array
    const GrB_Index *I,     // input array
    int64_t nvals           // size of input and output arrays
) ;

void GB_matlab_helper2      // fill Xp and Xi for a dense matrix
(
    GrB_Index *Xp,          // size ncols+1
    GrB_Index *Xi,          // size nrows*ncols
    int64_t ncols,
    int64_t nrows
) ;

bool GB_matlab_helper3          // return true if OK, false on error
(
    int64_t *List,              // size len, output array
    double *List_double,        // size len, input array
    int64_t len,
    int64_t *List_max           // also compute the max entry in the list
) ;

int64_t GB_matlab_helper4       // find max (I) + 1
(
    const GrB_Index *I,         // array of size len
    const int64_t len
) ;

#endif

