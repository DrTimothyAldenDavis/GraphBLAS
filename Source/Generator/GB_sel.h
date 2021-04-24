// SPDX-License-Identifier: Apache-2.0
if_phase1
void GB (_sel_phase1)
(
    int64_t *GB_RESTRICT Zp,
    int64_t *GB_RESTRICT Cp,
    int64_t *GB_RESTRICT Wfirst,
    int64_t *GB_RESTRICT Wlast,
    const GrB_Matrix A,
    const bool flipij,
    const int64_t ithunk,
    const GB_atype *GB_RESTRICT xthunk,
    const GxB_select_function user_select,
    const int64_t *A_ek_slicing, const int A_ntasks, const int A_nthreads
) ;
endif_phase1

void GB (_sel_phase2)
(
    int64_t *GB_RESTRICT Ci,
    GB_atype *GB_RESTRICT Cx,
    const int64_t *GB_RESTRICT Zp,
    const int64_t *GB_RESTRICT Cp,
    const int64_t *GB_RESTRICT Cp_kfirst,
    const GrB_Matrix A,
    const bool flipij,
    const int64_t ithunk,
    const GB_atype *GB_RESTRICT xthunk,
    const GxB_select_function user_select,
    const int64_t *A_ek_slicing, const int A_ntasks, const int A_nthreads
) ;

if_bitmap
void GB (_sel_bitmap)
(
    int8_t *Cb,
    GB_atype *GB_RESTRICT Cx,
    int64_t *cnvals_handle,
    GrB_Matrix A,
    const bool flipij,
    const int64_t ithunk,
    const GB_atype *GB_RESTRICT xthunk,
    const GxB_select_function user_select,
    const int nthreads
) ;
endif_bitmap
