
GrB_Info GB_Adot2B
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix *Aslice, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    int64_t *GB_RESTRICT B_slice,
    int64_t *GB_RESTRICT *C_counts,
    int nthreads, int naslice, int nbslice
) ;

GrB_Info GB_Adot3B
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    const GB_task_struct *GB_RESTRICT TaskList,
    const int ntasks,
    const int nthreads
) ;

GrB_Info GB_Asaxpy3B
(
    GrB_Matrix *Chandle,
    const GrB_Matrix A, bool A_is_pattern,
    const GrB_Matrix B, bool B_is_pattern,
    GB_saxpy3task_struct *GB_RESTRICT *TaskList_handle,
    void *Work [3], size_t Worksize [3],
    const int ntasks,
    const int nfine,
    const int nthreads,
    GB_Context Context
) ;

