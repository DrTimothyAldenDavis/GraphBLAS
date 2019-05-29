//------------------------------------------------------------------------------
// GB_cast_array: typecast an array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Casts an input array Ax to an output array Cx with a different built-in
// type.  Does not handle user-defined types.

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_unaryop__include.h"
#endif

void GB_cast_array              // typecast an array
(
    void *Cx,                   // output array
    const GB_Type_code code1,   // type code for Cx
    const void *Ax,             // input array
    const GB_Type_code code2,   // type code for Ax
    const int64_t anz,          // number of entries in Cx and Ax
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (anz == 0)
    { 
        // no work to do, and the Ax and Cx pointer may be NULL as well
        return ;
    }

    ASSERT (Cx != NULL) ;
    ASSERT (Ax != NULL) ;
    ASSERT (anz > 0) ;
    ASSERT (code1 <= GB_FP64_code) ;
    ASSERT (code2 <= GB_FP64_code) ;
    ASSERT (GB_code_compatible (code1, code2)) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // typecase the array
    //--------------------------------------------------------------------------

    #ifndef GBCOMPACT

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_unop(zname,xname) GB_unop__identity ## zname ## xname

        #define GB_WORKER(ignore1,zname,ztype,xname,xtype)              \
        {                                                               \
            GB_unop (zname,xname) ((ztype *) Cx, (const xtype *) Ax,    \
                anz, nthreads) ;                                        \
            return ;                                                    \
        }                                                               \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        #include "GB_2type_factory.c"

    #else

        //----------------------------------------------------------------------
        // generic worker: typecasting for compact case only
        //----------------------------------------------------------------------

        int64_t csize = GB_code_size (code1, 1) ;
        int64_t asize = GB_code_size (code2, 1) ;
        GB_cast_function cast_A_to_C = GB_cast_factory (code1, code2) ;

        #pragma omp parallel for num_threads(nthreads)
        for (int64_t p = 0 ; p < anz ; p++)
        { 
            // Cx [p] = Ax [p]
            cast_A_to_C (Cx +(p*csize), Ax +(p*asize), asize) ;
        }

    #endif
}

