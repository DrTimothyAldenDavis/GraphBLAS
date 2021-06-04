//------------------------------------------------------------------------------
// GB_iso_expand: expand a scalar into an entire array
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

void GB_iso_expand          // expand an iso scalar into an entire array
(
    void *restrict X,       // output array to expand into
    int64_t n,              // # of entries in X
    void *restrict scalar,  // scalar to expand into X
    size_t size,            // size of the scalar and each entry of X
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // determine how many threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (n, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // copy the value into X
    //--------------------------------------------------------------------------

    int64_t p ;
    switch (size)
    {

        case 1 : // GrB_BOOL, GrB_UINT8, GrB_INT8, and UDT of size 1
        {
            uint8_t a0 = (*((uint8_t *) scalar)) ;
            uint8_t *restrict Z = (uint8_t *) X ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < n ; p++)
            {
                Z [p] = a0 ;
            }
        }
        break ;

        case 2 : // GrB_UINT16, GrB_INT16, and UDT of size 2
        {
            uint16_t a0 = (*((uint16_t *) scalar)) ;
            uint16_t *restrict Z = (uint16_t *) X ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < n ; p++)
            {
                Z [p] = a0 ;
            }
        }
        break ;

        case 4 : // GrB_UINT32, GrB_INT32, GrB_FP32, and UDT of size 4
        {
            uint32_t a0 = (*((uint32_t *) scalar)) ;
            uint32_t *restrict Z = (uint32_t *) X ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < n ; p++)
            {
                Z [p] = a0 ;
            }
        }
        break ;

        case 8 : // GrB_UINT64, GrB_INT64, GrB_FP64, GxB_FC32, and UDT size 8
        {
            uint64_t a0 = (*((uint64_t *) scalar)) ;
            uint64_t *restrict Z = (uint64_t *) X ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < n ; p++)
            {
                Z [p] = a0 ;
            }
        }
        break ;

        case 16 : // GxB_FC64, and UDT size 16
        {
            uint64_t *restrict a = (uint64_t *) scalar ;
            uint64_t *restrict Z = (uint64_t *) X ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < n ; p++)
            {
                Z [2*p  ] = a [0] ;
                Z [2*p+1] = a [1] ;
            }
        }
        break ;

        default : // user-defined types of arbitrary size
        {
            GB_void *restrict Z = (GB_void *) X ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < n ; p++)
            {
                memcpy (Z + p*size, scalar, size) ;
            }
        }
        break ;
    }
}

