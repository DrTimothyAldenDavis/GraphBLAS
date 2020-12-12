//------------------------------------------------------------------------------
// GB_dense_subassign_21: C(:,:) = x where x is a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C(:,:) = x where C is a matrix and x is a scalar.

// C can have any sparsity on input; it is recreated as a full matrix, or left
// as bitmap.  If C is bitmap, it is either left as bitmap, or converted to
// full if allowed by C->sparsity.

// If C is bitmap, GB_subassigner_method does not select this method directly.
// Instead, it selects GB_bitmap_assign, which then just calls this method
// via GB_bitmap_assign_noM_noaccum_whole.

#include "GB_dense.h"
#include "GB_select.h"
#include "GB_Pending.h"

GrB_Info GB_dense_subassign_21      // C(:,:) = x, scalar to matrix assignment
(
    GrB_Matrix C,                   // input/output matrix
    const void *scalar,             // input scalar
    const GrB_Type scalar_type,     // type of the input scalar
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for C(:,:)=x", GB0) ;
    ASSERT (!GB_is_shallow (C)) ;
    ASSERT (scalar != NULL) ;
    // any prior pending tuples are discarded, and all zombies will be killed,
    // so C can be anything on input.
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (GB_PENDING_OK (C)) ;
    ASSERT_TYPE_OK (scalar_type, "scalar_type for C(:,:)=x", GB0) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t cvdim = C->vdim ;
    int64_t cvlen = C->vlen ;
    GrB_Index cnzmax ;
    bool ok = GB_Index_multiply (&cnzmax, cvlen, cvdim) ;
    if (!ok)
    {   GB_cov[3214]++ ;
// covered (3214): 2
        // problem too large
        return (GrB_OUT_OF_MEMORY) ;
    }

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // typecast the scalar into the same type as C
    //--------------------------------------------------------------------------

    int64_t csize = C->type->size ;
    GB_cast_function
        cast_A_to_C = GB_cast_factory (C->type->code, scalar_type->code) ;
    GB_void cwork [GB_VLA(csize)] ;
    cast_A_to_C (cwork, scalar, scalar_type->size) ;

    //--------------------------------------------------------------------------
    // ensure C is full or bitmap
    //--------------------------------------------------------------------------

    // discard any prior pending tuples
    GB_Pending_free (&(C->Pending)) ;

    if (GB_IS_SPARSE (C) || GB_IS_HYPERSPARSE (C))
    {
        // clear prior content and recreate it; use exising header for C.
        GB_phbix_free (C) ;
        int C_sparsity = C->sparsity ;  // save the sparsity control of C
        info = GB_new_bix (&C,  // full, old header
            C->type, cvlen, cvdim, GB_Ap_null, C->is_csc,
            GxB_FULL, true, C->hyper_switch, -1, cnzmax, true, Context) ;
        if (info != GrB_SUCCESS)
        {   GB_cov[3215]++ ;
// covered (3215): 843
            // out of memory
            return (GrB_OUT_OF_MEMORY) ;
        }
        C->magic = GB_MAGIC ;
        C->nvec_nonempty = (cvlen == 0) ? 0 : cvdim ;
        C->sparsity = C_sparsity ;      // restore the sparsity control of C
    }
    else if (GB_IS_BITMAP (C))
    {
        if (C->sparsity & GxB_FULL)
        {   GB_cov[3216]++ ;
// covered (3216): 82
            // C is bitmap but can become full; convert it to full
            GB_FREE (C->b) ;
        }
        else
        {   GB_cov[3217]++ ;
// NOT COVERED (3217):
            // C is bitmap and must remain so
            GB_memset (C->b, 1, cnzmax, nthreads_max) ;
            C->nvals = cnzmax ;
        }
    }

    //--------------------------------------------------------------------------
    // C = x
    //--------------------------------------------------------------------------

    if (!GB_is_nonzero (cwork, csize))
    {   GB_cov[3218]++ ;
// covered (3218): 486

        //----------------------------------------------------------------------
        // set all of C->x to zero
        //----------------------------------------------------------------------

        GB_memset (C->x, 0, cnzmax * csize, nthreads_max) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        int64_t pC ;
        int nthreads = GB_nthreads (cnzmax, chunk, nthreads_max) ;

        // worker for built-in types
        #define GB_WORKER(ctype)                                               \
        {                                                                      \
            ctype *GB_RESTRICT Cx = (ctype *) C->x ;                           \
            ctype x = (*(ctype *) cwork) ;                                     \
            GB_PRAGMA (omp parallel for num_threads(nthreads) schedule(static))\
            for (pC = 0 ; pC < cnzmax ; pC++)                                  \
            {                                                                  \
                Cx [pC] = x ;                                                  \
            }                                                                  \
        }                                                                      \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        switch (C->type->code)
        {
            case GB_BOOL_code    : GB_cov[3219]++ ;  GB_WORKER (bool) ;
// covered (3219): 30
            case GB_INT8_code    : GB_cov[3220]++ ;  GB_WORKER (int8_t) ;
// covered (3220): 104
            case GB_INT16_code   : GB_cov[3221]++ ;  GB_WORKER (int16_t) ;
// covered (3221): 4
            case GB_INT32_code   : GB_cov[3222]++ ;  GB_WORKER (int32_t) ;
// covered (3222): 8
            case GB_INT64_code   : GB_cov[3223]++ ;  GB_WORKER (int64_t) ;
// covered (3223): 4
            case GB_UINT8_code   : GB_cov[3224]++ ;  GB_WORKER (uint8_t) ;
// covered (3224): 4
            case GB_UINT16_code  : GB_cov[3225]++ ;  GB_WORKER (uint16_t) ;
// covered (3225): 106
            case GB_UINT32_code  : GB_cov[3226]++ ;  GB_WORKER (uint32_t) ;
// covered (3226): 4
            case GB_UINT64_code  : GB_cov[3227]++ ;  GB_WORKER (uint64_t) ;
// covered (3227): 102
            case GB_FP32_code    : GB_cov[3228]++ ;  GB_WORKER (float) ;
// covered (3228): 4
            case GB_FP64_code    : GB_cov[3229]++ ;  GB_WORKER (double) ;
// covered (3229): 52
            case GB_FC32_code    : GB_cov[3230]++ ;  GB_WORKER (GxB_FC32_t) ;
// covered (3230): 4
            case GB_FC64_code    : GB_cov[3231]++ ;  GB_WORKER (GxB_FC64_t) ;
// covered (3231): 111
            default:
                {
                    // worker for all user-defined types
                    GB_BURBLE_N (cnzmax, "(generic C(:,:)=x assign) ") ;
                    GB_void *GB_RESTRICT Cx = (GB_void *) C->x ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(static)
                    for (pC = 0 ; pC < cnzmax ; pC++)
                    {   GB_cov[3232]++ ;
// covered (3232): 13
                        memcpy (Cx +((pC)*csize), cwork, csize) ;
                    }
                }
                break ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C(:,:)=x output", GB0) ;
    ASSERT (GB_IS_FULL (C) || GB_IS_BITMAP (C)) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    return (GrB_SUCCESS) ;
}

