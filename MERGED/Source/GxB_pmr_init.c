//------------------------------------------------------------------------------
// GxB_pmr_init: initialize GraphBLAS for use with a C++ polymorphic allocator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_init, GxB_init, or GxB_pmr_init must called before any other GraphBLAS
// operation.  GrB_finalize must be called as the last GraphBLAS operation.

// GxB_pmr_init allows the use of C-callable functions that can be integrated
// with a C++ std::pmr::polymorphic_allocator as the backend.

// The C signatures of these two functions allow the use of any a
// PMR-compatible allocator for C++17 and later, such as the NVIDIA Rapids
// Memory Manager.  For example:

//      void *my_alloc (size_t *size)
//      {
//          return (malloc (*size)) ;
//      }

//      void my_dealloc (void *p, size_t size)
//      {
//          free (p) ;
//      }

// The first parameter of my_alloc provides the size, but as (size_t *) instead
// of size_t as used by the ANSI C malloc function.  In addition, this size
// parameter may be modified by my_alloc, and returned to GraphBLAS.  For
// example, to ensure all blocks of memory allocated by GraphBLAS are of size
// at least 256, use the following:

//      void *my_alloc (size_t *size)
//      {
//          if (*size < 256) (*size) = 256 ;
//          return (malloc (*size)) ;
//      }

// To free this block of memory, GraphBLAS will call my_dealloc (p, size), with
// this size as modified by the call to my_alloc.  The my_dealloc function can
// rely on this size to be accurate, which is required for a C++ compatible
// method such as the std::pmr::polymorphic_allocator as the backend.

// To use PMR, see the optional GraphBLAS/pmr_wrap library (TODO) which
// provides a C-callable interface to a PMR-compatible allocator for C++17 and
// later, with the C functions pmr_allocate and pmr_deallocate.  GraphBLAS can
// then be initialized with:

//      GxB_pmr_init (GrB_NONBLOCKING, pmr_allocate, pmr_deallocate) ;

// To use the NVIDIA Rapids Memory Manager, simply pass in rmm_allocate and
// rmm_deallocate, located in the optional library GraphBLAS/rmm_wrap, which
// then also requires the RMM library itself in GraphBLAS/rmm.  This method
// is required to use the GPU with GraphBLAS:

//      GxB_pmr_init (GrB_NONBLOCKING, rmm_allocate, rmm_deallocate) ;

//------------------------------------------------------------------------------
// GxB_pmr_init: initialize GraphBLAS for use with a C++ polymorphic allocator
//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_pmr_init       // start up GraphBLAS for use with pmr
(
    GrB_Mode mode,          // blocking or non-blocking mode
    // pmr allocate/deallocate memory management functions
    void * (* pmr_allocate_function   ) (size_t *),
    void   (* pmr_deallocate_function ) (void *p, size_t size)
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CONTEXT ("GxB_pmr_init (mode, pmr_allocate_function, "
        "pmr_deallocate_function)") ;
    GB_RETURN_IF_NULL (pmr_allocate_function) ;
    GB_RETURN_IF_NULL (pmr_deallocate_function) ;

    //--------------------------------------------------------------------------
    // initialize GraphBLAS
    //--------------------------------------------------------------------------

    return (GB_init
        (mode,                      // blocking or non-blocking mode
        NULL, NULL, NULL, true,     // do not use ANSI C11 functions
        pmr_allocate_function,      // use pmr-compatible allocate/deallocate
        pmr_deallocate_function,
        Context)) ;
}

