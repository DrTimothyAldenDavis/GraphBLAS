//------------------------------------------------------------------------------
// GB_unaryop_transpose: C=op(cast(A')), transpose, typecast, and apply op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// PARALLEL: done, but uses only naslice = nnz(A)/(A->vlen) threads.  This is
// limited since each thread requires O(vlen) space for the rowcount workspace.

{

    //--------------------------------------------------------------------------
    // get A and C
    //--------------------------------------------------------------------------

    const int64_t *restrict Ai = A->i ;

    #if defined ( GB_PHASE_2_OF_2 )
    const GB_ATYPE *restrict Ax = A->x ;
    int64_t  *restrict Cp = C->p ;
    int64_t  *restrict Ci = C->i ;
    GB_CTYPE *restrict Cx = C->x ;
    #endif

    //--------------------------------------------------------------------------
    // C = op (cast (A'))
    //--------------------------------------------------------------------------

    #pragma omp parallel for num_threads(naslice) schedule(dynamic,1)
    for (int taskid = 0 ; taskid < naslice ; taskid++)
    { 
        // get the rowcount for this slice, of size A->vlen
        int64_t *restrict rowcount = Rowcounts [taskid] ;
        for (int64_t Iter_k = A_slice [taskid] ;
                     Iter_k < A_slice [taskid+1] ;
                     Iter_k++)
        {
            GBI_jth_iteration_with_iter (Iter, j, pA, pA_end) ;
            for ( ; pA < pA_end ; pA++)
            { 
                int64_t i = Ai [pA] ;
                #if defined ( GB_PHASE_1_OF_2)
                // count one more entry in C(i,:) for this slice
                rowcount [i]++ ;
                #else
                // insert the entry into C(i,:) for this slice
                int64_t pC = Cp [i] + rowcount [i]++ ;
                Ci [pC] = j ;
                // Cx [pC] = op (cast (Ax [pA]))
                GB_CAST_OP (pC, pA) ;
                #endif
            }
        }
    }
}

