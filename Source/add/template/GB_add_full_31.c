//------------------------------------------------------------------------------
// GB_add_full_31: C=A+B; C and A are full, B is bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // Method31: C and A are full; B is bitmap
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(C_nthreads) schedule(static)
    for (p = 0 ; p < cnz ; p++)
    {
        if (Bb [p])
        { 
            // C (i,j) = A (i,j) + B (i,j)
            GB_EWISE_AIJ_OP_BIJ (Cx, p, Ax, p, A_iso, Bx, p, B_iso,
                p % vlen, p / vlen) ;
        }
        else
        {
            // C (i,j) = A (i,j), or A (i,j) + beta
            GB_EWISE_AIJ_OP_BETA (Cx, p, Ax, p, A_iso, beta_scalar,
                p % vlen, p / vlen) ;
        }
    }
}

