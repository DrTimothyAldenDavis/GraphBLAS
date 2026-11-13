//------------------------------------------------------------------------------
// GB_cuda_accum_mask_branch: determine if GPU should be used for accum/mask
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_cuda.hpp"

bool GB_cuda_accum_mask_branch  // true if CUDA is to be used
(
    const GrB_Matrix C,         // input/output matrix for results
    const GrB_Matrix M,         // optional mask for C, unused if NULL
    const GrB_BinaryOp accum,   // optional accum for Z=accum(C,results)
    const GrB_Matrix T,         // results of computation, freed when done
    const bool C_replace,       // if true, clear C first
    const bool Mask_struct      // if true, use the only structure of M
)
{

    //--------------------------------------------------------------------------
    // temporary checks; CUDA should handle these in the future
    //--------------------------------------------------------------------------

    // The CUDA kernel currently only
    // handles the case when all matrices are sparse/hypersparse.
    // FIXME: CUDA needs to handle all bitmap/full matrices.

    if ((M != NULL && (GB_IS_BITMAP (M) || GB_IS_FULL (M)))
        || GB_IS_BITMAP (T) || GB_IS_FULL (T))
    {
        // M (if present) and T must be sparse or hypersparse, for now
        // FIXME: handle the bitmap/full cases for C, M, and T
        return (false) ;
    }

    // if C_in_ignored is true, the input value of C is ignored
    bool C_in_ignored = (accum == NULL) && (M == NULL || C_replace) ;

    if (!C_in_ignored)
    {
        if (GB_IS_BITMAP (C) || GB_IS_FULL (C))
        {
            // unless C is ignored on input, it cannot be bitmap or full,
            // for now
            // FIXME: handle the bitmap/full cases for C, M, and T
            return (false) ;
        }
        int64_t cnz = GB_nnz (C) ; // includes live entries and zombies
        int64_t cnpending = GB_Pending_n (C) ; // # pending tuples in C:
        int64_t tnz = GB_nnz (T) ;
        if ((tnz + cnpending) / 16 <= cnz)
        {
            // T is a very small update to C; use the CPU instead for now,
            // so that C can acquire more pending tuples via GB_subassign.
            // FIXME: allow CUDA to create pending tuples itself, for C
            return (false) ;
        }
    }

    //--------------------------------------------------------------------------

    if (GB_ngpus_to_use (1) == 0) return (false) ;

    int data_arena = C->data_arena ;
    int header_arena = GB_arena (C->header_mem) ;
    int dev = data_arena - GxB_NARENAS ;

    // FIXME: in the future, the matrices can have header arenas that are not
    // in the data_arena of this particular GPU device.  The headers just need
    // to be copied over, and back again if they are modified on the GPU.  We
    // do not yet do this, so the tests below currently require "data arena ==
    // header arena"

    bool use_cuda =
        && (dev >= 0 && dev <= GB_Global_gpu_count_get ( )) // data on GPU
        && (data_arena == header_arena)                 // header on same GPU
        && (data_arena == T->data_arena)                // T on same GPU
        && (data_arena == GB_arena (T->header_mem))
        && (GB_jitifyer_get_control ( ) >= GxB_JIT_RUN) // JIT is running
        && (op->hash != UINT64_MAX)                     // op is jitable
        && GB_cuda_type_branch (T->type)                // types OK for CUDA
        && GB_cuda_type_branch (C->type)
        && GB_shallow_arenas_ok (T) ;       // shallow data on same GPU

    if (accum != NULL)
    {
        use_cuda = use_cuda
        && GB_cuda_type_branch (op->xtype)
        && GB_cuda_type_branch (op->ytype)
        && GB_cuda_type_branch (op->ztype) ;
    }

    if (M != NULL)
    {
        use_cuda = use_cuda
        && (data_arena == M->data_arena)                // M on same GPU
        && (data_arena == GB_arena (M->header_mem))
        && (Mask_struct || GB_cuda_type_branch (M->type))
        && GB_shallow_arenas_ok (M) ;
    }

    return (use_cuda) ;
}

