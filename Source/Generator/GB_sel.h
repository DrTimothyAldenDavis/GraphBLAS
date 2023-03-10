
// SPDX-License-Identifier: Apache-2.0
m4_divert(if_phase1)
void GB (_sel_phase1)
(
    int64_t *restrict Cp,
    int64_t *restrict Wfirst,
    int64_t *restrict Wlast,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

m4_divert(if_phase2)
void GB (_sel_phase2)
(
    int64_t *restrict Ci,
    GB_void *restrict Cx_out,
    const int64_t *restrict Cp,
    const int64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
) ;

m4_divert(if_bitmap)
void GB (_sel_bitmap)
(
    int8_t *Cb,
    int64_t *cnvals_handle,
    GrB_Matrix A,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int nthreads
) ;

