//------------------------------------------------------------------------------
// GB_unaryop_transpose_op: C=op(cast(A')), transpose, apply op and cast
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // C=op(cast(A'))
    //--------------------------------------------------------------------------

    const int64_t *restrict Ai = A->i ;

    GBI_for_each_vector (A)
    {
        GBI_for_each_entry (j, p, pend)
        {
            int64_t q = Cp [Ai [p]]++ ;
            Ci [q] = j ;
            // aij = A (i,j)
            GB_GETA (aij, Ax, p) ;
            // Cx [q] = op (cast (aij))
            GB_CASTING (x, aij) ;
            GB_OP (GB_CX (q), x) ;
        }
    }
}

