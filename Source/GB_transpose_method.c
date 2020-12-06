//------------------------------------------------------------------------------
// GB_transpose_method: select method for GB_transpose
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_transpose.h"

// GB_transpose can use choose between a merge-sort-based method that takes
// O(anz*log(anz)) time, or a bucket-sort method that takes O(anz+m+n) time.

bool GB_transpose_method    // true: use sort, false: use bucket
(
    GrB_Matrix A
)
{

    // return true to use GB_builder, false to use bucket method

    int64_t hack = GB_Global_hack_get ( ) ;
    if (hack < 0) return (false) ;      // use bucket
    if (hack > 0) return (true) ;       // use sort

    int64_t anvec = A->nvec ;
    int64_t anz = GB_NNZ (A) ;
    int64_t avlen = A->vlen ;
    int64_t avdim = A->vdim ;

    // select automatically
    double sort_work   = (log2 ((double) anz + 1) * (anz)) ;
    double bucket_work = (double) (anz + avlen + anvec) ;

    return (sort_work < bucket_work) ;
}

