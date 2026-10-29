//------------------------------------------------------------------------------
// GB_add_full_33:  C=A+B; C and B are full, A is bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // Method33: C and B are full; A is bitmap
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(C_nthreads) schedule(static)
    for (p = 0 ; p < cnz ; p++)
    {
        if (Ab [p])
        { 
            // C (i,j) = A (i,j) + B (i,j)
            GB_EWISE_AIJ_OP_BIJ (Cx, p, Ax, p, A_iso, Bx, p, B_iso,
                p % vlen, p / vlen) ;
        }
        else
        {
            // C (i,j) = B (i,j), or alpha + B (i,j)
            GB_EWISE_ALPHA_OP_BIJ (Cx, p, alpha_scalar, Bx, p, B_iso,
                p % vlen, p / vlen) ;
        }
    }
}

