//------------------------------------------------------------------------------
// GB_unop_transpose: C=op(cast(A')), transpose, typecast, and apply op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// This method is parallel, but not highly scalable.  It uses only naslice =
// nnz(A)/(A->vlen) threads.  Each thread requires O(vlen) workspace.

{

    // Ax unused for some uses of this template
    #include "GB_unused.h"

    //--------------------------------------------------------------------------
    // get A and C
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int64_t *GB_RESTRICT Ai = A->i ;

    #if defined ( GB_PHASE_2_OF_2 )
    const GB_ATYPE *GB_RESTRICT Ax = (GB_ATYPE *) A->x ;
    int64_t  *GB_RESTRICT Ci = C->i ;
    GB_CTYPE *GB_RESTRICT Cx = (GB_CTYPE *) C->x ;
    #endif

    //--------------------------------------------------------------------------
    // C = op (cast (A'))
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(naslice) schedule(static)
    for (tid = 0 ; tid < naslice ; tid++)
    {
        // get the rowcount for this slice, of size A->vlen
        int64_t *GB_RESTRICT rowcount = Rowcounts [tid] ;
        for (int64_t k = A_slice [tid] ; k < A_slice [tid+1] ; k++)
        {
            // iterate over the entries in A(:,j)
            int64_t j = (Ah == NULL) ? k : Ah [k] ;
            int64_t pA_start = Ap [k] ;
            int64_t pA_end = Ap [k+1] ;
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            { 
                // get A(i,j) where i = Ai [pA]
                #if defined ( GB_PHASE_1_OF_2)
                // count one more entry in C(i,:) for this slice
                rowcount [Ai [pA]]++ ;
                #else
                // insert the entry into C(i,:) for this slice
                int64_t pC = rowcount [Ai [pA]]++ ;
                Ci [pC] = j ;
                // Cx [pC] = op (Ax [pA])
                GB_CAST_OP (pC, pA) ;
                #endif
            }
        }
    }
}

