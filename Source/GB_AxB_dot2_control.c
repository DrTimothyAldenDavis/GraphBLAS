//------------------------------------------------------------------------------
// GB_AxB_dot2_control.c: determine when to use GB_AxB_dot2
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C=A'*B, C<M>=A'*B, or C<!M>=A'*B where C is constructed in bitmap format.
// C must be small and likely very dense.

// TODO:: tune this heuristic.  See gbtest99 and k = 1000.
// See also SuiteSparse/MATLAB_Tools/SSMULT/ssmult.c

#include "GB_mxm.h"

bool GB_AxB_dot2_control  // true: use dot2, false: use saxpy
(
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Context Context
)
{

    // TODO:: HACK
    int hack = GB_Global_hack_get ( ) ;
    if (hack == 10) { GBURBLE ("(dot2:force true) ") ; return (true) ; }
    if (hack == 11) { GBURBLE ("(dot2:force false) ") ; return (false) ; }

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

    if (A->nvec_nonempty < 0) A->nvec_nonempty = GB_nvec_nonempty (A, Context) ;
    if (B->nvec_nonempty < 0) B->nvec_nonempty = GB_nvec_nonempty (B, Context) ;
    double anvec = A->nvec_nonempty ;
    double bnvec = B->nvec_nonempty ;
    double avlen = A->vlen ;
    ASSERT (avlen == B->vlen) ;

    if (anz == 0)
    { 
        // C is empty, so use saxpy and compute it as a sparse empty matrix
        return (false) ;
    }

    double cnz = (anvec * bnvec) ;  // size of the C bitmap
    if (anz + bnz > 10000 * cnz || cnz <= 100)
    { 
        // The C bitmap is very small compared with A and B, so use dot2
        // and construct C as bitmap
        GBURBLE ("(C tiny: use dot) ") ;
        return (true) ;
    }

    if (anz + bnz < 10 * cnz)
    { 
        // The C bitmap is too big, use saxpy and construct C as sparse
        GBURBLE ("(C huge: use saxpy) ") ;
        return (false) ;
    }

    // average # of entries in each row and column of A (assuming A is CSC)
    double row_degree = anz / avlen ;
    double col_degree = anz / anvec ;
    if (row_degree < 0.125 && col_degree > 256)
    { 
        // If AT=A' is computed, it will have mostly empty vectors (the
        // row_degree of A), so do not transpose it.  However, the vectors
        // (col_degree) have lots of entries, and dot2 is efficient in this
        // case.  Use dot2 anc compute C as bitmap.
        GBURBLE ("(rowdeg tiny, coldeg big: use dot) ") ;
        return (true) ;
    }

    // if none of the above rules trigger, use saxpy
    GBURBLE ("(punt: saxpy C=(A')*B) ") ;
    return (false) ;
}

