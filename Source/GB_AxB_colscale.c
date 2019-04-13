//------------------------------------------------------------------------------
// GB_AxB_colscale: C = A*D, column scale with diagonal matrix D
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_binop__include.h"
#endif

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
    ASSERT_OK (GB_check (A, "A for colscale A*D", GB0)) ;
    ASSERT_OK (GB_check (D, "D for colscale A*D", GB0)) ;
    ASSERT (!GB_PENDING (A)) ; ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_PENDING (D)) ; ASSERT (!GB_ZOMBIES (D)) ;
    ASSERT_OK (GB_check (semiring, "semiring for numeric A*D", GB0)) ;
    ASSERT (A->vdim == D->vlen) ;
    ASSERT (GB_is_diagonal (D, Context)) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;
// fprintf (stderr, "\ncolscale, threads: %d\n", nthreads) ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    ASSERT (mult->ztype == semiring->add->op->ztype) ;

    bool op_is_first  = mult->opcode == GB_FIRST_opcode ;
    bool op_is_second = mult->opcode == GB_SECOND_opcode ;
    bool A_is_pattern = false ;
    bool D_is_pattern = false ;

    if (flipxy)
    { 
        // z = fmult (b,a) will be computed
        A_is_pattern = op_is_first  ;
        D_is_pattern = op_is_second ;
        if (!A_is_pattern) ASSERT (GB_Type_compatible (A->type, mult->ytype)) ;
        if (!D_is_pattern) ASSERT (GB_Type_compatible (D->type, mult->xtype)) ;
    }
    else
    { 
        // z = fmult (a,b) will be computed
        A_is_pattern = op_is_second ;
        D_is_pattern = op_is_first  ;
        if (!A_is_pattern) ASSERT (GB_Type_compatible (A->type, mult->xtype)) ;
        if (!D_is_pattern) ASSERT (GB_Type_compatible (D->type, mult->ytype)) ;
    }

    (*Chandle) = NULL ;

    //--------------------------------------------------------------------------
    // copy the pattern of A into C
    //--------------------------------------------------------------------------

// #if defined ( _OPENMP )
// double t = omp_get_wtime ( ) ;
// #endif

    // allocate but do not initialize C->x
    info = GB_dup (Chandle, A, false, mult->ztype, Context) ;
    if (info != GrB_SUCCESS)
    {
        // out of memory
        return (info) ;
    }

// #if defined ( _OPENMP )
// t = omp_get_wtime ( ) - t ;
// fprintf (stderr, "\ndup time %g ", t) ;
// #endif

    GrB_Matrix C = (*Chandle) ;

    //--------------------------------------------------------------------------
    // C = A*D, column scale
    //--------------------------------------------------------------------------

    bool done = false ;

// #if defined ( _OPENMP )
// t = omp_get_wtime ( ) ;
// #endif

#ifndef GBCOMPACT

    //--------------------------------------------------------------------------
    // define the worker for the switch factory
    //--------------------------------------------------------------------------

    #define GB_AxD(mult,xyname) GB_AxD_ ## mult ## xyname

    #define GB_BINOP_WORKER(mult,xyname)                                      \
    {                                                                         \
        GB_AxD(mult,xyname) (C, A, A_is_pattern, D, D_is_pattern, nthreads) ; \
        done = true ;                                                         \
    }                                                                         \
    break ;

    //--------------------------------------------------------------------------
    // launch the switch factory
    //--------------------------------------------------------------------------

    GB_Opcode opcode ;
    GB_Type_code xycode, zcode ;

    if (GB_binop_builtin (A, A_is_pattern, D, D_is_pattern, mult,
        flipxy, &opcode, &xycode, &zcode))
    { 
        #include "GB_binop_factory.c"
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

        GB_void *restrict Cx = C->x ;

        GB_cast_function cast_A, cast_D ;
        if (flipxy)
        { 
            // A is typecasted to y, and D is typecasted to x
            cast_A = A_is_pattern ? NULL : 
                     GB_cast_factory (mult->ytype->code, A->type->code) ;
            cast_D = D_is_pattern ? NULL : 
                     GB_cast_factory (mult->xtype->code, D->type->code) ;
        }
        else
        { 
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
            GB_void aij [aij_size] ;                                        \
            if (!A_is_pattern) cast_A (aij, Ax +((pA)*asize), asize) ;

        // dji = D(j,j), located in Dx [j]
        #define GB_GETB(djj,Dx,j)                                           \
            GB_void djj [djj_size] ;                                        \
            if (!D_is_pattern) cast_D (djj, Dx +((j)*dsize), dsize) ;

        // C(i,j) = A(i,j) * D(j,j)
        #define GB_BINOP(cij, aij, djj)                                     \
            GB_BINARYOP (cij, aij, djj) ;                                   \

        // address of Cx [p]
        #define GB_CX(p) Cx +((p)*csize)

        #define GB_ATYPE GB_void
        #define GB_BTYPE GB_void
        #define GB_CTYPE GB_void

        if (flipxy)
        { 
            #define GB_BINARYOP(z,x,y) fmult (z,y,x)
            #include "GB_AxB_colscale_meta.c"
            #undef GB_BINARYOP
        }
        else
        { 
            #define GB_BINARYOP(z,x,y) fmult (z,x,y)
            #include "GB_AxB_colscale_meta.c"
            #undef GB_BINARYOP
        }
    }

// #if defined ( _OPENMP )
// t = omp_get_wtime ( ) - t ;
// fprintf (stderr, " colscale time: %g\n", t) ;
// #endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_OK (GB_check (C, "colscale: C = A*D output", GB0)) ;
    ASSERT (*Chandle == C) ;
    return (GrB_SUCCESS) ;
}

