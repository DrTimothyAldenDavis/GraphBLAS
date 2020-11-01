//------------------------------------------------------------------------------
// GB_AxB_colscale: C = A*D where D is diagonal
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_mxm.h"
#include "GB_binop.h"
#include "GB_apply.h"
#include "GB_ek_slice.h"
#ifndef GBCOMPACT
#include "GB_binop__include.h"
#endif

#define GB_FREE_WORK \
    GB_ek_slice_free (&pstart_slice, &kfirst_slice, &klast_slice) ;

GrB_Info GB_AxB_colscale            // C = A*D, column scale with diagonal D
(
    GrB_Matrix *Chandle,            // output matrix
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix D,             // diagonal input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*D
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (Chandle != NULL) ;
    ASSERT_MATRIX_OK (A, "A for colscale A*D", GB0) ;
    ASSERT_MATRIX_OK (D, "D for colscale A*D", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_ZOMBIES (D)) ;
    ASSERT (!GB_JUMBLED (D)) ;
    ASSERT (!GB_PENDING (D)) ;
    ASSERT_SEMIRING_OK (semiring, "semiring for numeric A*D", GB0) ;
    ASSERT (A->vdim == D->vlen) ;
    ASSERT (GB_is_diagonal (D, Context)) ;

    ASSERT (!GB_IS_BITMAP (A)) ;        // TODO: ok for now
    ASSERT (!GB_IS_BITMAP (D)) ;        // ok: D is not bitmap
    ASSERT (!GB_IS_FULL (D)) ;          // ok: D is not full

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    ASSERT (mult->ztype == semiring->add->op->ztype) ;
    GB_Opcode opcode = mult->opcode ;

    //--------------------------------------------------------------------------
    // copy the pattern of A into C
    //--------------------------------------------------------------------------

    // allocate C->x but do not initialize it
    (*Chandle) = NULL ;
    info = GB_dup (Chandle, A, false, mult->ztype, Context) ;
    if (info != GrB_SUCCESS)
    {   GB_cov[1931]++ ;
// covered (1931): 9505
        // out of memory
        return (info) ;
    }
    GrB_Matrix C = (*Chandle) ;

    //--------------------------------------------------------------------------
    // apply a positional operator: convert C=A*D to C=op(A)
    //--------------------------------------------------------------------------

    if (GB_OPCODE_IS_POSITIONAL (opcode))
    {   GB_cov[1932]++ ;
// NOT COVERED (1932):
GB_GOTCHA ;
        if (flipxy)
        {   GB_cov[1933]++ ;
// NOT COVERED (1933):
GB_GOTCHA ;
            // the multiplicative operator is fmult(y,x), so flip the opcode
            opcode = GB_binop_flip (opcode) ;
        }
        // determine unary operator to compute C=A*D
        GrB_UnaryOp op1 = NULL ;
        if (mult->ztype == GrB_INT64)
        {
            switch (opcode)
            {
                // first_op(A,D) becomes position_op(A)
                case GB_FIRSTI_opcode    : GB_cov[1934]++ ;  op1 = GxB_POSITIONI_INT64  ;
// NOT COVERED (1934):
GB_GOTCHA ;
                    break ;
                case GB_FIRSTJ_opcode    : GB_cov[1935]++ ;  op1 = GxB_POSITIONJ_INT64  ;
// NOT COVERED (1935):
GB_GOTCHA ;
                    break ;
                case GB_FIRSTI1_opcode   : GB_cov[1936]++ ;  op1 = GxB_POSITIONI1_INT64 ;
// NOT COVERED (1936):
GB_GOTCHA ;
                    break ;
                case GB_FIRSTJ1_opcode   : GB_cov[1937]++ ;  op1 = GxB_POSITIONJ1_INT64 ;
// NOT COVERED (1937):
GB_GOTCHA ;
                    break ;
                // second_op(A,D) becomes position_j(A)
                case GB_SECONDI_opcode   : GB_cov[1938]++ ;  
// NOT COVERED (1938):
GB_GOTCHA ;
                case GB_SECONDJ_opcode   : GB_cov[1939]++ ;  op1 = GxB_POSITIONJ_INT64  ;
// NOT COVERED (1939):
GB_GOTCHA ;
                    break ;
                case GB_SECONDI1_opcode  : GB_cov[1940]++ ;  
// NOT COVERED (1940):
                case GB_SECONDJ1_opcode  : GB_cov[1941]++ ;  op1 = GxB_POSITIONJ1_INT64 ;
// NOT COVERED (1941):
GB_GOTCHA ;
                    break ;
                default:  ;
            }
        }
        else
        {
            switch (opcode)
            {
                // first_op(A,D) becomes position_op(A)
                case GB_FIRSTI_opcode    : GB_cov[1942]++ ;  op1 = GxB_POSITIONI_INT32  ;
// NOT COVERED (1942):
GB_GOTCHA ;
                    break ;
                case GB_FIRSTJ_opcode    : GB_cov[1943]++ ;  op1 = GxB_POSITIONJ_INT32  ;
// NOT COVERED (1943):
GB_GOTCHA ;
                    break ;
                case GB_FIRSTI1_opcode   : GB_cov[1944]++ ;  op1 = GxB_POSITIONI1_INT32 ;
// NOT COVERED (1944):
GB_GOTCHA ;
                    break ;
                case GB_FIRSTJ1_opcode   : GB_cov[1945]++ ;  op1 = GxB_POSITIONJ1_INT32 ;
// NOT COVERED (1945):
GB_GOTCHA ;
                    break ;
                // second_op(A,D) becomes position_j(A)
                case GB_SECONDI_opcode   : GB_cov[1946]++ ;  
// NOT COVERED (1946):
GB_GOTCHA ;
                case GB_SECONDJ_opcode   : GB_cov[1947]++ ;  op1 = GxB_POSITIONJ_INT32  ;
// NOT COVERED (1947):
GB_GOTCHA ;
                    break ;
                case GB_SECONDI1_opcode  : GB_cov[1948]++ ;  
// NOT COVERED (1948):
GB_GOTCHA ;
                case GB_SECONDJ1_opcode  : GB_cov[1949]++ ;  op1 = GxB_POSITIONJ1_INT32 ;
// NOT COVERED (1949):
GB_GOTCHA ;
                    break ;
                default:  ;
            }
        }
        info = GB_apply_op (C->x, op1, NULL, NULL, NULL, A, Context) ;
        if (info != GrB_SUCCESS)
        {   GB_cov[1950]++ ;
// NOT COVERED (1950):
GB_GOTCHA ;
            // out of memory
            GB_Matrix_free (Chandle) ;
            return (info) ;
        }
        ASSERT_MATRIX_OK (C, "colscale positional: C = A*D output", GB0) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t anz = GB_NNZ_HELD (A) ;
    int64_t anvec = A->nvec ;
    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (anz + anvec, chunk, nthreads_max) ;
    int ntasks = (nthreads == 1) ? 1 : (32 * nthreads) ;

    //--------------------------------------------------------------------------
    // slice the entries for each task
    //--------------------------------------------------------------------------

    // Task tid does entries pstart_slice [tid] to pstart_slice [tid+1]-1 and
    // vectors kfirst_slice [tid] to klast_slice [tid].  The first and last
    // vectors may be shared with prior slices and subsequent slices.

    int64_t *pstart_slice = NULL, *kfirst_slice = NULL, *klast_slice = NULL ;
    if (!GB_ek_slice (&pstart_slice, &kfirst_slice, &klast_slice, A, &ntasks))
    {   GB_cov[1951]++ ;
// covered (1951): 5703
        // out of memory
        GB_Matrix_free (Chandle) ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // determine if the values are accessed
    //--------------------------------------------------------------------------

    bool op_is_first  = (opcode == GB_FIRST_opcode) ;
    bool op_is_second = (opcode == GB_SECOND_opcode) ;
    bool op_is_pair   = (opcode == GB_PAIR_opcode) ;
    bool A_is_pattern = false ;
    bool D_is_pattern = false ;

    if (flipxy)
    {   GB_cov[1952]++ ;
// NOT COVERED (1952):
GB_GOTCHA ;
        // z = fmult (b,a) will be computed
        A_is_pattern = op_is_first  || op_is_pair ;
        D_is_pattern = op_is_second || op_is_pair ;
        ASSERT (GB_IMPLIES (!A_is_pattern,
            GB_Type_compatible (A->type, mult->ytype))) ;
        ASSERT (GB_IMPLIES (!D_is_pattern,
            GB_Type_compatible (D->type, mult->xtype))) ;
    }
    else
    {   GB_cov[1953]++ ;
// covered (1953): 13847
        // z = fmult (a,b) will be computed
        A_is_pattern = op_is_second || op_is_pair ;
        D_is_pattern = op_is_first  || op_is_pair ;
        ASSERT (GB_IMPLIES (!A_is_pattern,
            GB_Type_compatible (A->type, mult->xtype))) ;
        ASSERT (GB_IMPLIES (!D_is_pattern,
            GB_Type_compatible (D->type, mult->ytype))) ;
    }

    //--------------------------------------------------------------------------
    // C = A*D, column scale, via built-in binary operators
    //--------------------------------------------------------------------------

    bool done = false ;

    #ifndef GBCOMPACT

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_AxD(mult,xname) GB_AxD_ ## mult ## xname

        #define GB_BINOP_WORKER(mult,xname)                                  \
        {                                                                    \
            info = GB_AxD(mult,xname) (C, A, A_is_pattern, D, D_is_pattern,  \
                kfirst_slice, klast_slice, pstart_slice, ntasks, nthreads) ; \
            done = (info != GrB_NO_VALUE) ;                                  \
        }                                                                    \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        GB_Type_code xcode, ycode, zcode ;
        if (GB_binop_builtin (A->type, A_is_pattern, D->type, D_is_pattern,
            mult, flipxy, &opcode, &xcode, &ycode, &zcode))
        {   GB_cov[1954]++ ;
// covered (1954): 13847
            // C=A*D, colscale with built-in operator
            #define GB_BINOP_IS_SEMIRING_MULTIPLIER
            #include "GB_binop_factory.c"
            #undef  GB_BINOP_IS_SEMIRING_MULTIPLIER
        }

    #endif

    //--------------------------------------------------------------------------
    // C = A*D, column scale, with typecasting or user-defined operator
    //--------------------------------------------------------------------------

    if (!done)
    {

        //----------------------------------------------------------------------
        // get operators, functions, workspace, contents of A, D, and C
        //----------------------------------------------------------------------

        GB_BURBLE_MATRIX (C, "(generic C=A*D colscale) ") ;

        GxB_binary_function fmult = mult->function ;

        size_t csize = C->type->size ;
        size_t asize = A_is_pattern ? 0 : A->type->size ;
        size_t dsize = D_is_pattern ? 0 : D->type->size ;

        size_t xsize = mult->xtype->size ;
        size_t ysize = mult->ytype->size ;

        // scalar workspace: because of typecasting, the x/y types need not
        // be the same as the size of the A and D types.
        // flipxy false: aij = (xtype) A(i,j) and djj = (ytype) D(j,j)
        // flipxy true:  aij = (ytype) A(i,j) and djj = (xtype) D(j,j)
        size_t aij_size = flipxy ? ysize : xsize ;
        size_t djj_size = flipxy ? xsize : ysize ;

        GB_void *GB_RESTRICT Cx = (GB_void *) C->x ;

        GB_cast_function cast_A, cast_D ;
        if (flipxy)
        {   GB_cov[1955]++ ;
// NOT COVERED (1955):
GB_GOTCHA ;
            // A is typecasted to y, and D is typecasted to x
            cast_A = A_is_pattern ? NULL :
                     GB_cast_factory (mult->ytype->code, A->type->code) ;
            cast_D = D_is_pattern ? NULL :
                     GB_cast_factory (mult->xtype->code, D->type->code) ;
        }
        else
        {   GB_cov[1956]++ ;
// NOT COVERED (1956):
GB_GOTCHA ;
            // A is typecasted to x, and D is typecasted to y
            cast_A = A_is_pattern ? NULL :
                     GB_cast_factory (mult->xtype->code, A->type->code) ;
            cast_D = D_is_pattern ? NULL :
                     GB_cast_factory (mult->ytype->code, D->type->code) ;
        }

        //----------------------------------------------------------------------
        // C = A*D via function pointers, and typecasting
        //----------------------------------------------------------------------

        // aij = A(i,j), located in Ax [pA]
        #define GB_GETA(aij,Ax,pA)                                          \
            GB_void aij [GB_VLA(aij_size)] ;                                \
            if (!A_is_pattern) cast_A (aij, Ax +((pA)*asize), asize) ;

        // dji = D(j,j), located in Dx [j]
        #define GB_GETB(djj,Dx,j)                                           \
            GB_void djj [GB_VLA(djj_size)] ;                                \
            if (!D_is_pattern) cast_D (djj, Dx +((j)*dsize), dsize) ;

        // address of Cx [p]
        #define GB_CX(p) Cx +((p)*csize)

        #define GB_ATYPE GB_void
        #define GB_BTYPE GB_void
        #define GB_CTYPE GB_void

        // no vectorization
        #define GB_PRAGMA_SIMD_VECTORIZE ;

        if (flipxy)
        {   GB_cov[1957]++ ;
// NOT COVERED (1957):
GB_GOTCHA ;
            #define GB_BINOP(z,x,y,i,j) fmult (z,y,x)
            #include "GB_AxB_colscale_meta.c"
            #undef GB_BINOP
        }
        else
        {   GB_cov[1958]++ ;
// NOT COVERED (1958):
GB_GOTCHA ;
            #define GB_BINOP(z,x,y,i,j) fmult (z,x,y)
            #include "GB_AxB_colscale_meta.c"
            #undef GB_BINOP
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "colscale: C = A*D output", GB0) ;
    ASSERT (*Chandle == C) ;
    GB_FREE_WORK ;
    return (GrB_SUCCESS) ;
}

