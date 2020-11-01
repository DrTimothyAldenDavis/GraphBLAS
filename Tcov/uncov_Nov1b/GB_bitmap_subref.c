//------------------------------------------------------------------------------
// GB_bitmap_subref: C = A(I,J) where A is bitmap or full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// C=A(I,J), where A is bitmap or full, symbolic and numeric.
// See GB_subref for details.

#include "GB_subref.h"
#include "GB_subassign_IxJ_slice.h"

#define GB_FREE_ALL             \
{                               \
    GB_Matrix_free (Chandle) ;  \
}

GrB_Info GB_bitmap_subref       // C = A(I,J): either symbolic or numeric
(
    // output
    GrB_Matrix *Chandle,
    // input, not modified
    const bool C_is_csc,        // requested format of C
    const GrB_Matrix A,
    const GrB_Index *I,         // index list for C = A(I,J), or GrB_ALL, etc.
    const int64_t ni,           // length of I, or special
    const GrB_Index *J,         // index list for C = A(I,J), or GrB_ALL, etc.
    const int64_t nj,           // length of J, or special
    const bool symbolic,        // if true, construct C as symbolic
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (Chandle != NULL) ;
    ASSERT_MATRIX_OK (A, "A for C=A(I,J) bitmap subref", GB0) ;
    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    (*Chandle) = NULL ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const int8_t  *GB_RESTRICT Ab = A->b ;
    const GB_void *GB_RESTRICT Ax = (GB_void *) A->x ;
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    const size_t asize = A->type->size ;

    //--------------------------------------------------------------------------
    // check the properties of I and J
    //--------------------------------------------------------------------------

    // C = A(I,J) so I is in range 0:avlen-1 and J is in range 0:avdim-1
    int64_t nI, nJ, Icolon [3], Jcolon [3] ;
    int Ikind, Jkind ;
    GB_ijlength (I, ni, avlen, &nI, &Ikind, Icolon) ;
    GB_ijlength (J, nj, avdim, &nJ, &Jkind, Jcolon) ;

    bool I_unsorted, I_has_dupl, I_contig, J_unsorted, J_has_dupl, J_contig ;
    int64_t imin, imax, jmin, jmax ;

    info = GB_ijproperties (I, ni, nI, avlen, &Ikind, Icolon,
        &I_unsorted, &I_has_dupl, &I_contig, &imin, &imax, Context) ;
    if (info != GrB_SUCCESS)
    {   GB_cov[2833]++ ;
// covered (2833): 2764
        // I invalid
        return (info) ;
    }

    info = GB_ijproperties (J, nj, nJ, avdim, &Jkind, Jcolon,
        &J_unsorted, &J_has_dupl, &J_contig, &jmin, &jmax, Context) ;
    if (info != GrB_SUCCESS)
    {   GB_cov[2834]++ ;
// covered (2834): 3166
        // J invalid
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL ;
    int64_t cnzmax ;
    bool ok = GB_Index_multiply ((GrB_Index *) (&cnzmax), nI, nJ) ;
    if (!ok)
    {
        // problem too large
        return (GrB_OUT_OF_MEMORY) ;
    }
    GrB_Type ctype = symbolic ? GrB_INT64 : A->type ;
    int sparsity = GB_IS_BITMAP (A) ? GxB_BITMAP : GxB_FULL ;
    GB_OK (GB_new_bix (Chandle, // bitmap or full, new header
        ctype, nI, nJ, GB_Ap_null, C_is_csc,
        sparsity, A->hyper_switch, -1, cnzmax, true, Context)) ;
    C = (*Chandle) ;

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    int8_t *GB_RESTRICT Cb = C->b ;

    // In GB_bitmap_assign_IxJ_template, vlen is the vector length of the
    // submatrix C(I,J), but here the template is used to access A(I,J), and so
    // the vector length is A->vlen, not C->vlen.  The pointers pA and pC are
    // swapped in the macros below, since C=A(I,J) is being computed, instead
    // of C(I,J)=A for the bitmap assignment.

    int64_t vlen = avlen ;

    //--------------------------------------------------------------------------
    // C = A(I,J)
    //--------------------------------------------------------------------------

    int64_t cnvals = 0 ;

    if (sparsity == GxB_BITMAP)
    {

        //----------------------------------------------------------------------
        // C = A (I,J) where A and C are both bitmap
        //----------------------------------------------------------------------

        if (symbolic)
        {   GB_cov[2835]++ ;
// NOT COVERED (2835):
GB_GOTCHA ;
            int64_t *GB_RESTRICT Cx = (int64_t *) C->x ;
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pA,pC)                                      \
            {                                                               \
                int8_t ab = Ab [pA] ;                                       \
                Cb [pC] = ab ;                                              \
                Cx [pC] = pA ;                                              \
                cnvals += ab ;                                              \
            }
            #include "GB_bitmap_assign_IxJ_template.c"
        }
        else
        {   GB_cov[2836]++ ;
// covered (2836): 105499
            GB_void *GB_RESTRICT Cx = (GB_void *) C->x ;
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pA,pC)                                      \
            {                                                               \
                int8_t ab = Ab [pA] ;                                       \
                Cb [pC] = ab ;                                              \
                if (ab)                                                     \
                {                                                           \
                    /* Cx [pC] = Ax [pA] */                                 \
                    memcpy (Cx +((pC)*asize), Ax +((pA)*asize), asize) ;    \
                    cnvals++ ;                                              \
                }                                                           \
            }
            #include "GB_bitmap_assign_IxJ_template.c"
        }
        C->nvals = cnvals ;

    }
    else
    {

        //----------------------------------------------------------------------
        // C = A (I,J) where A and C are both full
        //----------------------------------------------------------------------

        if (symbolic)
        {   GB_cov[2837]++ ;
// covered (2837): 952
            int64_t *GB_RESTRICT Cx = (int64_t *) C->x ;
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pA,pC)                                      \
            {                                                               \
                Cx [pC] = pA ;                                              \
            }
            #include "GB_bitmap_assign_IxJ_template.c"
        }
        else
        {   GB_cov[2838]++ ;
// covered (2838): 10766
            GB_void *GB_RESTRICT Cx = (GB_void *) C->x ;
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pA,pC)                                      \
            {                                                               \
                /* Cx [pC] = Ax [pA] */                                     \
                memcpy (Cx +((pC)*asize), Ax +((pA)*asize), asize) ;        \
            }
            #include "GB_bitmap_assign_IxJ_template.c"
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (C, "C output for bitmap subref C=A(I,J)", GB0) ;
    return (GrB_SUCCESS) ;
}

