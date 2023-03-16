//------------------------------------------------------------------------------
// GB_subassign_05d: C(:,:)<M> = scalar where C is as-if-full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed (now).

// Method 05d: C(:,:)<M> = scalar ; no S, C is dense

// M:           present
// Mask_comp:   false
// C_replace:   false
// accum:       NULL
// A:           scalar
// S:           none

// C can have any sparsity structure, but it must be entirely dense with
// all entries present.

#include "GB_subassign_shared_definitions.h"
#include "GB_subassign_methods.h"
#include "GB_subassign_dense.h"
#include "GB_unused.h"
#ifndef GBCUDA_DEV
#include "GB_as__include.h"
#endif

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_WERK_POP (M_ek_slicing, int64_t) ;   \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_WORKSPACE

GrB_Info GB_subassign_05d
(
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const bool Mask_struct,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_aliased (C, M)) ;   // NO ALIAS of C==M

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_WERK_DECLARE (M_ek_slicing, int64_t) ;

    ASSERT_MATRIX_OK (C, "C for subassign method_05d", GB0) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    ASSERT (GB_as_if_full (C)) ;

    ASSERT_MATRIX_OK (M, "M for subassign method_05d", GB0) ;
    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_PENDING (M)) ;

    GB_ENSURE_FULL (C) ;    // convert C to full, if sparsity control allows it
    if (C->iso)
    { 
        // work has already been done by GB_assign_prep
        return (GrB_SUCCESS) ;
    }

    const GB_Type_code ccode = C->type->code ;
    const size_t csize = C->type->size ;
    GB_GET_SCALAR ;

    //--------------------------------------------------------------------------
    // Method 05d: C(:,:)<M> = scalar ; no S; C is dense
    //--------------------------------------------------------------------------

    // Time: Optimal:  the method must iterate over all entries in M,
    // and the time is O(nnz(M)).

    //--------------------------------------------------------------------------
    // Parallel: slice M into equal-sized chunks
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // slice the entries for each task
    //--------------------------------------------------------------------------

    int M_ntasks, M_nthreads ;
    GB_SLICE_MATRIX (M, 8, chunk) ;

    //--------------------------------------------------------------------------
    // via the factory kernel
    //--------------------------------------------------------------------------

    info = GrB_NO_VALUE ;

    #ifndef GBCUDA_DEV

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_sub05d(cname) GB (_subassign_05d_ ## cname)
        #define GB_WORKER(cname)                                    \
        {                                                           \
            info = GB_sub05d (cname) (C, M, Mask_struct, cwork,     \
                M_ek_slicing, M_ntasks, M_nthreads) ;               \
        }                                                           \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        // C<M> = x
        switch (ccode)
        {
            case GB_BOOL_code   : GB_WORKER (_bool  )
            case GB_INT8_code   : GB_WORKER (_int8  )
            case GB_INT16_code  : GB_WORKER (_int16 )
            case GB_INT32_code  : GB_WORKER (_int32 )
            case GB_INT64_code  : GB_WORKER (_int64 )
            case GB_UINT8_code  : GB_WORKER (_uint8 )
            case GB_UINT16_code : GB_WORKER (_uint16)
            case GB_UINT32_code : GB_WORKER (_uint32)
            case GB_UINT64_code : GB_WORKER (_uint64)
            case GB_FP32_code   : GB_WORKER (_fp32  )
            case GB_FP64_code   : GB_WORKER (_fp64  )
            case GB_FC32_code   : GB_WORKER (_fc32  )
            case GB_FC64_code   : GB_WORKER (_fc64  )
            default: ;
        }

    #endif

    //--------------------------------------------------------------------------
    // via the JIT kernel
    //--------------------------------------------------------------------------

    #if GB_JIT_ENABLED
    // JIT TODO: type: subassign 05d
    #endif

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 

        //----------------------------------------------------------------------
        // GB_subassign_05d: C(:,:)<M> = scalar where C is as-if-full
        //----------------------------------------------------------------------

        // get operators, functions, workspace, contents of A and C

        #include "GB_generic.h"
        GB_BURBLE_MATRIX (M, "(generic C(:,:)<M>=x assign) ") ;

        const size_t csize = C->type->size ;

        // Cx [pC] = cwork
        #undef  GB_COPY_scalar_to_C
        #define GB_COPY_scalar_to_C(Cx,pC,cwork) \
            memcpy (Cx + ((pC)*csize), cwork, csize)

        #include "GB_subassign_05d_template.c"
        info = GrB_SUCCESS ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    if (info == GrB_SUCCESS)
    {
        ASSERT_MATRIX_OK (C, "C output for subassign method_05d", GB0) ;
    }
    return (info) ;
}

