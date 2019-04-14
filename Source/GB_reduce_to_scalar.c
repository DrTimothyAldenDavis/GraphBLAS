//------------------------------------------------------------------------------
// GB_reduce_to_scalar: reduce a matrix to a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// c = accum (c, reduce_to_scalar(A)), reduce entries in a matrix
// to a scalar.  Not user-callable.  Does the work for GrB_*_reduce_TYPE,
// both matrix and vector.  This funciton tolerates zombies and does not
// delete them.  It does not tolerate pending tuples, so if they are present,
// all zombies are deleted and all pending tuples are assembled.

// This function does not need to know if A is hypersparse or not, and its
// result is the same if A is in CSR or CSC format.

// PARALLEL: done, but needs tuning for chunk size.
// A parallel reduction of all entries in A to a scalar

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_reduce__include.h"
#endif

GrB_Info GB_reduce_to_scalar    // s = reduce_to_scalar (A)
(
    void *c,                    // result scalar
    const GrB_Type ctype,       // the type of scalar, c
    const GrB_BinaryOp accum,   // for c = accum(c,s)
    const GrB_Monoid reduce,    // monoid to do the reduction
    const GrB_Matrix A,         // matrix to reduce
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // Zombies are an opaque internal detail of the GrB_Matrix data structure
    // that do not depend on anything outside the matrix.  Thus, Table 2.4 of
    // the GrapBLAS spec, version 1.1.0, does not require their deletion.
    // Pending tuples are different, since they rely on another object outside
    // the matrix: the pending operator, which might be user-defined.  Per
    // Table 2.4, the user can expect that GrB_reduce applies the pending
    // operator, which can then be deleted by the user.  Thus, if the pending
    // operator is user-defined it must be applied here.  Assembling pending
    // tuples requires zombies to be deleted first.  Note that if the pending
    // operator is built-in, then the updates could in principle be skipped,
    // but this could be done only if the reduce monoid is the same as the
    // pending operator.

    GB_WAIT_PENDING (A) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;       // Zombies are tolerated, and not deleted
    GB_RETURN_IF_NULL_OR_FAULTY (reduce) ;
    GB_RETURN_IF_FAULTY (accum) ;
    GB_RETURN_IF_NULL (c) ;

    ASSERT_OK (GB_check (ctype, "type of scalar c", GB0)) ;
    ASSERT_OK (GB_check (reduce, "reduce for reduce_to_scalar", GB0)) ;
    ASSERT_OK_OR_NULL (GB_check (accum, "accum for reduce_to_scalar", GB0)) ;
    ASSERT_OK (GB_check (A, "A for reduce_to_scalar", GB0)) ;

    // check domains and dimensions for c = accum (c,s)
    GrB_Type ztype = reduce->op->ztype ;
    GrB_Info info = GB_compatible (ctype, NULL, NULL, accum, ztype, Context) ;
    if (info != GrB_SUCCESS)
    { 
        return (info) ;
    }

    // s = reduce (s,A) must be compatible
    if (!GB_Type_compatible (A->type, ztype))
    { 
        return (GB_ERROR (GrB_DOMAIN_MISMATCH, (GB_LOG,
            "incompatible type for reduction operator z=%s(x,y):\n"
            "input of type [%s]\n"
            "cannot be typecast to reduction operator of type [%s]",
            reduce->op->name, A->type->name, reduce->op->ztype->name))) ;
    }

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;

    //--------------------------------------------------------------------------
    // scalar workspace
    //--------------------------------------------------------------------------

    int64_t asize = A->type->size ;
    int64_t anz = GB_NNZ (A) ;
    const int64_t *restrict Ai = A->i ;

    // reduce the # of threads if the problem is small
    // TODO find a good chunk size
    #define GB_CHUNK (4*1024)

    nthreads = GB_IMIN (nthreads, anz / GB_CHUNK) ;
    nthreads = GB_IMAX (nthreads, 1) ;

    int64_t zsize = ztype->size ;

    GB_void s [zsize] ;

    //--------------------------------------------------------------------------
    // s = reduce_to_scalar (A)
    //--------------------------------------------------------------------------

    // s = identity
    memcpy (s, reduce->identity, zsize) ;

    // get terminal value, if any
    GB_void *restrict terminal = reduce->terminal ;

    // reduce all the entries in the matrix, but skip any zombies

    if (A->type == ztype)
    {

        //----------------------------------------------------------------------
        // sum up the entries; no casting needed
        //----------------------------------------------------------------------

        // There are 44 common cases of this function for built-in types and
        // operators.  Four associative operators: min, max, plus, and times
        // with 10 types (int*, uint*, float, and double), and four logical
        // operators (or, and, xor, eq) with a boolean type of C.  All 44 are
        // hard-coded below via a switch factory.  If the case is not handled
        // by the switch factory, 'done' remains false.

        bool done = false ;

        // define the worker for the switch factory

        #define GB_red(op,zname) GB_red_scalar_ ## op ## zname

        #define GB_ASSOC_WORKER(op,zname,ztype,terminal)    \
        {                                                   \
            GB_red (op, zname) ((ztype *) s, A, nthreads) ; \
            done = true ;                                   \
        }                                                   \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        #ifndef GBCOMPACT

            // controlled by opcode and typecode
            GB_Opcode opcode = reduce->op->opcode ;
            GB_Type_code typecode = A->type->code ;
            ASSERT (typecode <= GB_UDT_code) ;

            #include "GB_assoc_factory.c"

        #endif

        //----------------------------------------------------------------------
        // generic worker: sum up the entries, no typecasting
        //----------------------------------------------------------------------

        if (!done)
        {
            // the switch factory didn't handle this case
            GxB_binary_function freduce = reduce->op->function ;

            #define GB_ATYPE GB_void

            // workspace for each thread
            #define GB_REDUCE_WORKSPACE(w, nthreads) GB_void w [nthreads*zsize]

            // t = identity
            #define GB_REDUCE_INIT(t)                   \
                GB_void t [zsize] ;                     \
                memcpy (t, reduce->identity, zsize) ;

            // t += Ax [p], no typecasting
            #define GB_REDUCE(t, Ax, p) freduce (t, t, Ax +((p)*asize)) ;

            // w [tid] = t
            #define GB_REDUCE_WRAPUP(w, tid, t)         \
                memcpy (w +(tid*zsize), t, zsize) ;

            // s += w [tid]
            #define GB_REDUCE_W(s, w, tid) freduce (s, s, w +((tid)*zsize)) ;

            // break if terminal value reached
            #define GB_REDUCE_TERMINAL(t)                           \
                if (terminal != NULL)                               \
                {                                                   \
                    if (memcmp (t, terminal, zsize) == 0) break ;   \
                }

            #include "GB_reduce_to_scalar_template.c"
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // generic worker: sum up the entries, with typecasting
        //----------------------------------------------------------------------

        GxB_binary_function freduce = reduce->op->function ;
        GB_cast_function
            cast_A_to_Z = GB_cast_factory (ztype->code, A->type->code) ;

            // t += ((ztype) Ax [p])
            #undef  GB_REDUCE
            #define GB_REDUCE(t, Ax, p)                         \
                GB_void awork [zsize] ;                         \
                cast_A_to_Z (awork, Ax +((p)*asize), zsize) ;   \
                freduce (t, t, awork) ;

            #include "GB_reduce_to_scalar_template.c"
    }

    //--------------------------------------------------------------------------
    // c = s or c = accum (c,s)
    //--------------------------------------------------------------------------

    // This operation does not use GB_accum_mask, since c and s are
    // scalars, not matrices.  There is no scalar mask.

    if (accum == NULL)
    { 
        // c = (ctype) s
        GB_cast_function
            cast_Z_to_C = GB_cast_factory (ctype->code, ztype->code) ;
        cast_Z_to_C (c, s, ctype->size) ;
    }
    else
    { 
        GxB_binary_function faccum = accum->function ;

        GB_cast_function cast_C_to_xaccum, cast_Z_to_yaccum, cast_zaccum_to_C ;
        cast_C_to_xaccum = GB_cast_factory (accum->xtype->code, ctype->code) ;
        cast_Z_to_yaccum = GB_cast_factory (accum->ytype->code, ztype->code) ;
        cast_zaccum_to_C = GB_cast_factory (ctype->code, accum->ztype->code) ;

        // scalar workspace
        char xaccum [accum->xtype->size] ;
        char yaccum [accum->ytype->size] ;
        char zaccum [accum->ztype->size] ;

        // xaccum = (accum->xtype) c
        cast_C_to_xaccum (xaccum, c, ctype->size) ;

        // yaccum = (accum->ytype) s
        cast_Z_to_yaccum (yaccum, s, zsize) ;

        // zaccum = xaccum "+" yaccum
        faccum (zaccum, xaccum, yaccum) ;

        // c = (ctype) zaccum
        cast_zaccum_to_C (c, zaccum, ctype->size) ;
    }

    return (GrB_SUCCESS) ;
}

