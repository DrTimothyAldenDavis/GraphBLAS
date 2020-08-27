//------------------------------------------------------------------------------
// GB_bitmap_assign_notM_noaccum:  assign to C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------
// C<!M>(I,J) = A       assign
// C(I,J)<!M> = A       subassign

// C<!M,repl>(I,J) = A       assign
// C(I,J)<!M,repl> = A       subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, hypersparse or sparse (not bitmap or full)
// Mask_comp:   true
// Mask_struct: true or false
// C_replace:   true or false
// accum:       not present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign, row assign, col assign, or subassign

#include "GB_bitmap_assign_methods.h"

#define GB_FREE_ALL \
    GB_ek_slice_free (&pstart_Mslice, &kfirst_Mslice, &klast_Mslice) ;

GrB_Info GB_bitmap_assign_notM_noaccum
(
    // input/output:
    GrB_Matrix C,               // input/output matrix in bitmap format
    // inputs:
    const bool C_replace,       // descriptor for C
    const GrB_Index *I,         // I index list
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,         // J index list
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,         // mask matrix
//  const bool Mask_comp,       // true here, for !M only
    const bool Mask_struct,     // true if M is structural, false if valued
//  const GrB_BinaryOp accum,   // not present
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    const int assign_kind,      // row assign, col assign, assign, or subassign
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBURBLE_BITMAP_ASSIGN ("bit8", M, true, NULL) ;
    ASSERT (GB_IS_BITMAP (C)) ;
    ASSERT (GB_IS_HYPERSPARSE (M) || GB_IS_SPARSE (M)) ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign, !M, noaccum", GB0) ;
    ASSERT_MATRIX_OK (M, "M for bitmap assign, !M, noaccum", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (A, "A for bitmap assign, !M, noaccum", GB0) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C
    GB_SLICE_M
    GB_GET_A

    //--------------------------------------------------------------------------
    // scatter M into the bitmap of C
    //--------------------------------------------------------------------------

    // for each entry mij == 1
    //     Cb (i,j) += 2
    #define GB_MASK_WORK(pC) Cb [pC] += 2 ;
    #include "GB_bitmap_assign_M_template.c"

        // Cb (i,j) = 0:   cij not present, mij zero: can be modified
        // Cb (i,j) = 1:   cij present, mij zero: can be modified,
        //                      but delete if aij not present
        // Cb (i,j) = 2:   cij not present, mij == 1: do not modify
        // Cb (i,j) = 3:   cij present, mij == 1: do not modify

    //--------------------------------------------------------------------------
    // assign A into C
    //--------------------------------------------------------------------------

    if (A == NULL)
    {

        //----------------------------------------------------------------------
        // scalar assignment: C<!M>(I,J) = scalar
        //----------------------------------------------------------------------

        // for all IxJ
        #define GB_IXJ_WORK(pC)             \
        {                                   \
            int8_t cb = Cb [pC] ;           \
            if (cb <= 1)                    \
            {                               \
                /* Cx [pC] = scalar */      \
                GB_ASSIGN_SCALAR (pC) ;     \
                Cb [pC] = 1 ;               \
                cnvals += (cb == 0) ;       \
            }                               \
            else if (C_replace)             \
            {                               \
                /* delete this entry */     \
                Cb [pC] = 0 ;               \
                cnvals -= (cb == 3) ;       \
            }                               \
            else                            \
            {                               \
                /* keep this entry */       \
                Cb [pC] = (cb == 3) ;       \
            }                               \
        }
        #include "GB_bitmap_assign_IxJ_template.c"

    }
    else
    {

        //----------------------------------------------------------------------
        // matrix assignment: C<!M>(I,J) = A
        //----------------------------------------------------------------------

        // for all entries aij in A (A can be hyper, sparse, bitmap, or full)
        //     if Cb(p) == 0       // C(iC,jC) is now present, insert
        //         Cx(p) = aij     //
        //         Cb(p) = 4       // keep it
        //         cnvals++
        //     if Cb(p) == 1       // C(iC,jC) still present, updated
        //         Cx(p) = aij     //
        //         Cb(p) = 4       // keep it
        //     if Cb(p) == 2       // do nothing
        //     if Cb(p) == 3       // do nothing

        #define GB_AIJ_WORK(pC,pA)          \
        {                                   \
            int8_t cb = Cb [pC] ;           \
            if (cb <= 1)                    \
            {                               \
                /* Cx [pC] = Ax [pA] */     \
                GB_ASSIGN_AIJ (pC, pA) ;    \
                Cb [pC] = 4 ;               \
                cnvals += (cb == 0) ;       \
            }                               \
        }
        #include "GB_bitmap_assign_A_template.c"

        //----------------------------------------------------------------------
        // handle entries in IxJ
        //----------------------------------------------------------------------

        if (C_replace)
        {
            // for all IxJ
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pC)                 \
            {                                       \
                int8_t cb = Cb [pC] ;               \
                Cb [pC] = (cb == 4) ;               \
                cnvals -= (cb == 1 || cb == 3) ;    \
            }
            #include "GB_bitmap_assign_IxJ_template.c"
        }
        else
        {
            // for all IxJ
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pC)                 \
            {                                       \
                int8_t cb = Cb [pC] ;               \
                Cb [pC] = (cb == 4 || cb == 3) ;    \
                cnvals -= (cb == 1) ;               \
            }
            #include "GB_bitmap_assign_IxJ_template.c"
        }
    }

    //--------------------------------------------------------------------------
    // handle entries outside of IxJ
    //--------------------------------------------------------------------------

    if (assign_kind == GB_SUBASSIGN)
    {
        // see above.  no more work to do
    }
    else
    {
        if (C_replace)
        {
            // for all entries in C.  Also clears M from C
            #define GB_CIJ_WORK(mij,pC)             \
            {                                       \
                int8_t cb = Cb [pC] ;               \
                Cb [pC] = (cb == 1) ;               \
                cnvals -= (cb == 3) ;               \
            }
            #include "GB_bitmap_assign_C_template.c"
        }
        else
        {
            // clear M from C
            #define GB_MASK_WORK(pC) Cb [pC] = (Cb [pC] % 2) ;
            #include "GB_bitmap_assign_M_template.c"
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "final C for bitmap assign, !M, noaccum", GB0) ;
    return (GrB_SUCCESS) ;
}

