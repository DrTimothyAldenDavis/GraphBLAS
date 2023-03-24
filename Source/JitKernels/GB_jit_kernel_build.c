//------------------------------------------------------------------------------
// GB_jit_kernel_build.c: kernel for GB_build
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

GrB_Info GB_jit_kernel
(
    GB_void *restrict Tx_void,
    int64_t *restrict Ti,
    const GB_void *restrict Sx_void,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

GrB_Info GB_jit_kernel
(
    GB_void *restrict Tx_void,
    int64_t  *restrict Ti,
    const GB_void *restrict Sx_void,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
)
{ 
    GB_T_TYPE *restrict Tx = (GB_T_TYPE *) Tx_void ;
    const GB_S_TYPE *restrict Sx = (GB_S_TYPE *) Sx_void ;
    #include "GB_bld_template.c"
    return (GrB_SUCCESS) ;
}


