//------------------------------------------------------------------------------
// GB_bitmap_assign_C_template: iterate over a bitmap matrix C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The #include'ing file defines a GB_CIJ_WORK macro for the body of the loop,
// which operates on the entry C(iC,jC) at position Cx [pC] and Cb [pC].  The C
// matrix held in bitmap form.  If the mask matrix is also a bitmap matrix or
// full matrix, the GB_GET_MIJ macro can compute the effective value of the
// mask for the C(iC,jC) entry.

#ifndef GB_GET_MIJ
#define GB_GET_MIJ(mij,pM) ;
#endif

{
    switch (assign_kind)
    {

        //----------------------------------------------------------------------
        // row assignment: C<M'>(iC,:), M is a column vector
        //----------------------------------------------------------------------

        case GB_ROW_ASSIGN :
        {
            #include "GB_bitmap_assign_C_row_template.c"
        }
        break ;

        //----------------------------------------------------------------------
        // column assignment: C<M>(:,jC), M is a column vector
        //----------------------------------------------------------------------

        case GB_COL_ASSIGN :
        {
            #include "GB_bitmap_assign_C_col_template.c"
        }
        break ;

        //----------------------------------------------------------------------
        // GrB_assign: C<M>(I,J), M is a matrix the same size as C
        //----------------------------------------------------------------------

        case GB_ASSIGN :
        {
            #include "GB_bitmap_assign_C_all_template.c"
        }
        break ;

        //----------------------------------------------------------------------
        // GxB_subassign: C(I,J)<M>, M is a matrix the same size as C(I,J)
        //----------------------------------------------------------------------

        case GB_SUBASSIGN :
        {

            // iterate over all of C(I,J)
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pC)                                         \
            {                                                               \
                GB_GET_MIJ (mij, (iA + jA * nI)) ; /* mij = Mask (p)    */  \
                GB_CIJ_WORK (mij, pC) ;         /* operate on C(iC,jC)  */  \
            }
            #include "GB_bitmap_assign_IxJ_template.c"
#if 0
            int64_t p ;
            int64_t ij_nvals = nI*nJ ;
            int nthreads = GB_nthreads (ij_nvals, chunk, nthreads_max) ;
            #pragma omp parallel for num_threads(nthreads) schedule(static) \
                reduction(+:cnvals)
            for (p = 0 ; p < ij_nvals ; p++)
            {
                int64_t iC = GB_ijlist (I, p % nI, Ikind, Icolon) ;
                int64_t jC = GB_ijlist (J, p / nI, Jkind, Jcolon) ;
                int64_t pC = iC + jC * cvlen ;
                GB_GET_MIJ (mij, p) ;           // mij = Mask (p)
                GB_CIJ_WORK (mij, pC) ;         // operate on C(iC,jC)
            }
#endif
        }
        break ;

        default:;
    }
}

#undef GB_GET_MIJ
#undef GB_CIJ_WORK

