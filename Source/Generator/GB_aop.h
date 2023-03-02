
// SPDX-License-Identifier: Apache-2.0
GrB_Info GB (_Cdense_accumB)
(
    GrB_Matrix C,
    const GrB_Matrix B,
    const int64_t *B_ek_slicing,
    const int B_ntasks,
    const int B_nthreads
) ;

GrB_Info GB (_Cdense_accumb)
(
    GrB_Matrix C,
    const GB_void *p_bwork,
    const int nthreads
) ;

