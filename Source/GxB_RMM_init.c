//------------------------------------------------------------------------------
// GxB_RMM_init: initialize GraphBLAS for use with RMM
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_init, GxB_init, or GxB_RMM_init must called before any other GraphBLAS
// operation.  GrB_finalize must be called as the last GraphBLAS operation.

#include "GB.h"

GrB_Info GxB_RMM_init       // start up GraphBLAS for use with RMM
(
    GrB_Mode mode,          // blocking or non-blocking mode
    // RMM allocate/deallocate memory management functions
    void * (* rmm_allocate_function   ) (size_t *),
    void   (* rmm_deallocate_function ) (void *p, size_t size)
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CONTEXT ("GxB_RMM_init (mode, rmm_allocate_function, "
        "rmm_deallocate_function)") ;
    GB_RETURN_IF_NULL (rmm_allocate_function) ;
    GB_RETURN_IF_NULL (rmm_deallocate_function) ;

    //--------------------------------------------------------------------------
    // initialize GraphBLAS
    //--------------------------------------------------------------------------

    return (GB_init
        (mode,                      // blocking or non-blocking mode
        NULL, NULL, NULL, true,     // do not use ANSI C11 functions
        rmm_allocate_function,      // use RMM
        rmm_deallocate_function,
        Context)) ;
}

