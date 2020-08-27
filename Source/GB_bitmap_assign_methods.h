//------------------------------------------------------------------------------
// GB_bitmap_assign_methods.h: definitions for GB_bitmap_assign* methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#ifndef GB_BITMAP_ASSIGN_METHODS_H
#define GB_BITMAP_ASSIGN_METHODS_H
#include "GB_bitmap_assign.h"
#include "GB_ek_slice.h"
#include "GB_ij.h"
#include "GB_subassign_IxJ_slice.h"

//------------------------------------------------------------------------------
// burble
//------------------------------------------------------------------------------

#if GB_BURBLE
    #define GBURBLE_BITMAP_ASSIGN(method,M,Mask_comp,accum)                 \
        GBURBLE ("Method:" method " ") ;                                    \
        GB_burble_assign (C_replace, Ikind, Jkind, M, Mask_comp,            \
            Mask_struct, accum, A, assign_kind) ;
#else
    #define GBURBLE_BITMAP ;
#endif

//------------------------------------------------------------------------------
// GB_GET_C: get the C matrix
//------------------------------------------------------------------------------

#define GB_GET_C                                                            \
    GrB_Info info ;                                                         \
    /* also get the max # of threads to use */                              \
    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;                    \
    ASSERT_MATRIX_OK (C, "C for bitmap assign", GB0) ;                      \
    int8_t  *GB_RESTRICT Cb = C->b ;                                        \
    GB_void *GB_RESTRICT Cx = (GB_void *) C->x ;                            \
    const size_t csize = C->type->size ;                                    \
    const GB_Type_code ccode = C->type->code ;                              \
    const int64_t cvdim = C->vdim ;                                         \
    const int64_t cvlen = C->vlen ;                                         \
    const int64_t cnzmax = cvlen * cvdim ;                                  \
    int64_t cnvals = C->nvals ;

//------------------------------------------------------------------------------
// GB_GET_M: get the mask matrix M
//------------------------------------------------------------------------------

#define GB_GET_M                                                            \
    ASSERT_MATRIX_OK (M, "M for bitmap assign", GB0) ;                      \
    const int64_t *GB_RESTRICT Mp = M->p ;                                  \
    const int8_t  *GB_RESTRICT Mb = M->b ;                                  \
    const int64_t *GB_RESTRICT Mh = M->h ;                                  \
    const int64_t *GB_RESTRICT Mi = M->i ;                                  \
    const GB_void *GB_RESTRICT Mx = (GB_void *) (Mask_struct ? NULL : (M->x)) ;\
    const size_t msize = M->type->size ;                                    \
    const size_t mvlen = M->vlen ;

//------------------------------------------------------------------------------
// GB_SLICE_M: slice the mask matrix M
//------------------------------------------------------------------------------

#define GB_SLICE_M                                                          \
    GB_GET_M                                                                \
    int64_t mnz = GB_NNZ (M) ;                                              \
    int mthreads = GB_nthreads (mnz + M->nvec, chunk, nthreads_max) ;       \
    int mtasks = (mthreads == 1) ? 1 : (8 * mthreads) ;                     \
    int64_t *pstart_Mslice = NULL ;                                         \
    int64_t *kfirst_Mslice = NULL ;                                         \
    int64_t *klast_Mslice  = NULL ;                                         \
    if (!GB_ek_slice (&pstart_Mslice, &kfirst_Mslice, &klast_Mslice,        \
        M, &mtasks))                                                        \
    {                                                                       \
        /* out of memory */                                                 \
        return (GrB_OUT_OF_MEMORY) ;                                        \
    }

//------------------------------------------------------------------------------
// GB_GET_A: get the A matrix or the scalar
//------------------------------------------------------------------------------

#define GB_GET_A                                                            \
    const int64_t *GB_RESTRICT Ap = NULL ;                                  \
    const int64_t *GB_RESTRICT Ah = NULL ;                                  \
    const int8_t  *GB_RESTRICT Ab = NULL ;                                  \
    const int64_t *GB_RESTRICT Ai = NULL ;                                  \
    const GB_void *GB_RESTRICT Ax = NULL ;                                  \
    size_t asize ;                                                          \
    GB_Type_code acode ;                                                    \
    if (A == NULL)                                                          \
    {                                                                       \
        asize = scalar_type->size ;                                         \
        acode = scalar_type->code ;                                         \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        ASSERT_MATRIX_OK (A, "A for bitmap assign", GB0) ;                  \
        ASSERT (nI == A->vlen && nJ == A->vdim) ;                           \
        asize = A->type->size ;                                             \
        acode = A->type->code ;                                             \
        Ap = A->p ;                                                         \
        Ah = A->h ;                                                         \
        Ab = A->b ;                                                         \
        Ai = A->i ;                                                         \
        Ax = (GB_void *) A->x ;                                             \
    }                                                                       \
    GB_cast_function cast_A_to_C = GB_cast_factory (ccode, acode) ;         \
    GB_void cwork [GB_VLA(csize)] ;                                         \
    if (A == NULL)                                                          \
    {                                                                       \
        cast_A_to_C (cwork, scalar, asize) ;                                \
    }

//------------------------------------------------------------------------------
// GB_GET_ACCUM: get the accumulator op and its related typecasting functions
//------------------------------------------------------------------------------

#define GB_GET_ACCUM                                                        \
    ASSERT_BINARYOP_OK (accum, "accum for bitmap assign", GB0) ;            \
    ASSERT (!GB_OP_IS_POSITIONAL (accum)) ;                                 \
    GxB_binary_function faccum = accum->function ;                          \
    GB_cast_function cast_A_to_Y = GB_cast_factory (accum->ytype->code, acode);\
    GB_cast_function cast_C_to_X = GB_cast_factory (accum->xtype->code, ccode);\
    GB_cast_function cast_Z_to_C = GB_cast_factory (ccode, accum->ztype->code);\
    size_t xsize = accum->xtype->size ;                                     \
    size_t ysize = accum->ytype->size ;                                     \
    size_t zsize = accum->ztype->size ;                                     \
    GB_void ywork [GB_VLA(ysize)] ;                                         \
    if (A == NULL)                                                          \
    {                                                                       \
        cast_A_to_Y (ywork, scalar, asize) ;                                \
    }

//------------------------------------------------------------------------------
// GB_ASSIGN_SCALAR:  Cx [pC] = cwork, already typecasted
//------------------------------------------------------------------------------

#define GB_ASSIGN_SCALAR(pC)                                \
{                                                           \
    memcpy (Cx +(pC)*csize, cwork, csize) ;                 \
}

//------------------------------------------------------------------------------
// GB_ASSIGN_AIJ:  Cx [pC] = Ax [pA], with typecasting as needed
//------------------------------------------------------------------------------

#define GB_ASSIGN_AIJ(pC,pA)                                \
{                                                           \
    cast_A_to_C (Cx +(pC)*csize, Ax +(pA)*asize, csize) ;   \
}

//------------------------------------------------------------------------------
// GB_ACCUM_SCALAR:  Cx [pC] += ywork
//------------------------------------------------------------------------------

#define GB_ACCUM_SCALAR(pC)                                 \
{                                                           \
    GB_void xwork [GB_VLA(xsize)] ;                         \
    cast_C_to_X (xwork, Cx +((pC)*csize), csize) ;          \
    GB_void zwork [GB_VLA(zsize)] ;                         \
    faccum (zwork, xwork, ywork) ;                          \
    cast_Z_to_C (Cx +((pC)*csize), zwork, csize) ;          \
}                                                           \

//------------------------------------------------------------------------------
// GB_ACCUM_AIJ:  Cx [pC] += Ax [pA]
//------------------------------------------------------------------------------

#define GB_ACCUM_AIJ(pC, pA)                                \
{                                                           \
    /* ywork = Ax [pA], with typecasting as needed */       \
    GB_void ywork [GB_VLA(ysize)] ;                         \
    cast_A_to_Y (ywork, Ax +((pA)*asize), asize) ;          \
    /* Cx [pC] += ywork */                                  \
    GB_ACCUM_SCALAR (pC) ;                                  \
}

//------------------------------------------------------------------------------
// prototypes
//------------------------------------------------------------------------------

GrB_Info GB_bitmap_assign_fullM_accum
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
    const GrB_Matrix M,         // mask matrix, which is not NULL here
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // present here
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    const int assign_kind,      // row assign, col assign, assign, or subassign
    GB_Context Context
) ;

GrB_Info GB_bitmap_assign_fullM_noaccum
(
    // input/output:
    GrB_Matrix C,               // input/output matrix in bitmap format
    const bool C_replace,       // descriptor for C
    // inputs:
    const GrB_Index *I,         // I index list
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,         // J index list
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,         // mask matrix, which is present here
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
//  const GrB_BinaryOp accum,   // not present
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    const int assign_kind,      // row assign, col assign, assign, or subassign
    GB_Context Context
) ;

GrB_Info GB_bitmap_assign_M_accum
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
    const GrB_Matrix M,         // mask matrix, which is not NULL here
//  const bool Mask_comp,       // false here
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // present here
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    const int assign_kind,      // row assign, col assign, assign, or subassign
    GB_Context Context
) ;

GrB_Info GB_bitmap_assign_M_noaccum
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
    const GrB_Matrix M,         // mask matrix, which is not NULL here
//  const bool Mask_comp,       // false here
    const bool Mask_struct,     // true if M is structural, false if valued
//  const GrB_BinaryOp accum,   // not present
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    const int assign_kind,      // row assign, col assign, assign, or subassign
    GB_Context Context
) ;

GrB_Info GB_bitmap_assign_noM_accum
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
//  const GrB_Matrix M,         // mask matrix, not present here
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // present
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    const int assign_kind,      // row assign, col assign, assign, or subassign
    GB_Context Context
) ;

GrB_Info GB_bitmap_assign_noM_noaccum
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
//  const GrB_Matrix M,         // mask matrix, not present here
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
//  const GrB_BinaryOp accum,   // not present
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    const int assign_kind,      // row assign, col assign, assign, or subassign
    GB_Context Context
) ;

GrB_Info GB_bitmap_assign_notM_accum
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
    const GrB_BinaryOp accum,   // present
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    const int assign_kind,      // row assign, col assign, assign, or subassign
    GB_Context Context
) ;

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
) ;

#endif

