
#include "GB_sort.h"
#include "gb_matlab.h"
#include "simple_timer.h"

// C = A*B using the hash method

int64_t myhash
(

    int64_t **Cp_handle,
    int64_t **Ci_handle,
    double  **Cx_handle,

    int64_t *restrict Ap,
    int64_t *restrict Ai,
    double  *restrict Ax,
    int64_t anrows,
    int64_t ancols,

    int64_t *restrict Bp,
    int64_t *restrict Bi,
    double  *restrict Bx,
    int64_t bnrows,
    int64_t bncols,

    int64_t *nonempty_result
) ;

void qsort_1b_double    // sort array A of size 2-by-n, using 1 key (A [0][])
(
    int64_t *GB_RESTRICT A_0,      // size n array
    double *GB_RESTRICT A_1,      // size n array
    const size_t xsize,         // size of entries in A_1
    const int64_t n
) ;

