//------------------------------------------------------------------------------
// GB_unaryop_apply_op: Cx=op(cast(Ax)), with typecasting
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// PARALLEL: done.
// all entries in C=op(cast(A)) are computed fully in parallel. 

{

    //--------------------------------------------------------------------------
    // Cx=op(cast(Ax))
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(nthreads)
    for (int64_t p = 0 ; p < anz ; p++)
    {
        // aij = A (i,j)
        GB_GETA (aij, Ax, p) ;
        // cij = op (cast (aij))
        GB_CASTING (x, aij) ;
        GB_OP (GB_CX (p), x) ;
    }
}

