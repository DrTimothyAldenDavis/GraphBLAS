// SPDX-License-Identifier: Apache-2.0
GrB_Info GB_Adot2B
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_comp, const bool Mask_struct,
    const GrB_Matrix A, bool A_is_pattern, int64_t *GB_RESTRICT A_slice,
    const GrB_Matrix B, bool B_is_pattern, int64_t *GB_RESTRICT B_slice,
    int nthreads, int naslice, int nbslice
) ;

GrB_Info GB_Adot3B
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_struct,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    const GB_task_struct *GB_RESTRICT TaskList,
    const int ntasks,
    const int nthreads
) ;

GrB_Info GB_AsaxpyB
(
    GrB_Matrix C,
    const GrB_Matrix M, const bool Mask_comp, const bool Mask_struct,
    const bool M_dense_in_place,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    const int saxpy_method,
    // for saxpy3 method only:
    GB_saxpy3task_struct *GB_RESTRICT SaxpyTasks,
    int ntasks, int nfine,
    // for saxpy3 and saxpy4 methods only:
    int nthreads,
    const int do_sort,
    // for saxpy4 method only:
    int8_t  *GB_RESTRICT Wf,
    int64_t **Wi_handle,
    size_t Wi_size,
    GB_void *GB_RESTRICT Wx,
    int64_t *GB_RESTRICT kfirst_Bslice,
    int64_t *GB_RESTRICT klast_Bslice,
    int64_t *GB_RESTRICT pstart_Bslice,
    GB_Context Context
) ;

GrB_Info GB_Adot4B
(
    GrB_Matrix C,
    const GrB_Matrix A, bool A_is_pattern,
    int64_t *GB_RESTRICT A_slice, int naslice,
    const GrB_Matrix B, bool B_is_pattern,
    int64_t *GB_RESTRICT B_slice, int nbslice,
    const int nthreads
) ;

