//------------------------------------------------------------------------------
// GB_AxB_dot2_control.c: determine when to use GB_AxB_dot2
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C=A'*B, C<M>=A'*B, or C<!M>=A'*B where C is constructed in bitmap format.
// C must be small and likely very dense.

#include "GB_mxm.h"

bool GB_AxB_dot2_control  // true: use dot2, false: use saxpy
(
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Context Context
)
{

//  int hack = GB_Global_hack_get ( ) ;     // see d2.m test script 
//  if (hack == 10) { GBURBLE ("(dot2:force true) ") ; return (true) ; }
//  if (hack == 11) { GBURBLE ("(dot2:force false) ") ; return (false) ; }

    //--------------------------------------------------------------------------
    // C = A'*B is very efficient if A and/or B are full or bitmap
    //--------------------------------------------------------------------------

    if (GB_IS_FULL (A) || GB_IS_BITMAP (A) ||
        GB_IS_FULL (B) || GB_IS_BITMAP (B))
    { 
        return (true) ;
    }

    //--------------------------------------------------------------------------
    // both A and B are sparse or hyper
    //--------------------------------------------------------------------------

    // Notation: C=A'*B where all 3 matrices are CSC.  This might be C=A*B'
    // where all 3 matrices are CSR, equivalently.  The comments here assume
    // CSC, but this method is CSC/CSR agnostic.

    double anz = GB_NNZ (A) ;       // # of entries in A
    double bnz = GB_NNZ (B) ;       // # of entries in B
    if (anz == 0)
    { 
        // C is empty, so use saxpy and compute it as a sparse empty matrix
        return (false) ;
    }

    if (A->nvec_nonempty < 0) A->nvec_nonempty = GB_nvec_nonempty (A, Context) ;
    if (B->nvec_nonempty < 0) B->nvec_nonempty = GB_nvec_nonempty (B, Context) ;
    double anvec = A->nvec_nonempty ;
    double bnvec = B->nvec_nonempty ;
    double avlen = A->vlen ;
    ASSERT (avlen == B->vlen) ;
    double cnz = (anvec * bnvec) ;  // size of the C bitmap
    double row_degree = anz / avlen ;
    double col_degree = anz / anvec ;
//  GBURBLE ("(rowdeg %g coldeg %g ", row_degree, col_degree) ;
//  GBURBLE ("anz+bnz %g cnz %g) ", anz+bnz, cnz) ;

    if (cnz > anz + bnz)
    { 
        // The C bitmap is too big, use saxpy and construct C as sparse
        GBURBLE ("(C huge: saxpy C=(A')*B) ") ;
        return (false) ;
    }

    if ((anz + bnz > 10000 * cnz) || (cnz <= 100))
    { 
        // The C bitmap is very small compared with A and B, so use dot2
        // and construct C as bitmap
        GBURBLE ("(C tiny: dot) ") ;
        return (true) ;
    }

    // average # of entries in each row and column of A (assuming A is CSC)
    if (row_degree < 0.125 && col_degree > 1200)
    { 
        // If AT=A' is computed, it will have mostly empty vectors (the
        // row_degree of A), so do not transpose it.  If the fraction of
        // populated vectors in AT is very low (< 0.0625 by default), then AT
        // will become hypersparse, and this slows down the saxpy method.  If
        // the vectors (col_degree) have lots of entries, then dot2 is
        // efficient in this case.  If both conditions how, use dot2 and
        // compute C as bitmap.
        GBURBLE ("(A' implicit: dot) ") ;
        return (true) ;
    }

    // if none of the above rules trigger, use saxpy
    GBURBLE ("(saxpy C=(A')*B) ") ;
    return (false) ;
}

