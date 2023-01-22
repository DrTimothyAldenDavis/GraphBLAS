//------------------------------------------------------------------------------
// GB_reduce_to_scalar_cuda_branch: when to use GPU for scalar reduction
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Decide branch direction for GPU use for the reduction to scalar

#include "GB_cuda.h"

bool GB_reduce_to_scalar_cuda_branch    // return true to use the GPU
(
    const GrB_Monoid monoid,        // monoid to do the reduction
    const GrB_Matrix A              // input matrix
)
{

    // work to do
    double work = GB_nnz_held (A) ;

    int ngpus_to_use = GB_ngpus_to_use (work) ;
    GBURBLE (" work:%g gpus:%d ", work, ngpus_to_use) ;

    GB_Opcode opcode = monoid->op->opcode ;

    if (ngpus_to_use > 0
        // do it on the CPU if the monoid operator is user-defined:
        // FIXME: handle user-defined operators
        && (opcode != GB_USER_binop_code)
        // the ANY monoid takes O(1) time; do it on the CPU:
        && (opcode != GB_ANY_binop_code)
        // FIXME: handle user-defined types:
        && (A->type->code != GB_UDT_code)
        // A iso takes O(log(nvals(A))) time; do it on the CPU:
        && !A->iso
    )
    {
        // FIXME: gpu_id = GB_Context_gpu_id_get ( ) ;
        // cudaSetDevice (gpu_id) ;
        return true;
    }
    else
    { 
        return false;
    }
}

