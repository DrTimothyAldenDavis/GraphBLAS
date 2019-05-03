//------------------------------------------------------------------------------
// GB_red__include.h: definitions for GB_red__*.c
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txargt for license.

// This file has been automatically generated from Generator/GB_red.h


void GB_red_scalar__min_int8
(
    int8_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__min_int8
(
    int8_t *restrict Tx,
    int64_t  *restrict Ti,
    const int8_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__min_int16
(
    int16_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__min_int16
(
    int16_t *restrict Tx,
    int64_t  *restrict Ti,
    const int16_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__min_int32
(
    int32_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__min_int32
(
    int32_t *restrict Tx,
    int64_t  *restrict Ti,
    const int32_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__min_int64
(
    int64_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__min_int64
(
    int64_t *restrict Tx,
    int64_t  *restrict Ti,
    const int64_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__min_uint8
(
    uint8_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__min_uint8
(
    uint8_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint8_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__min_uint16
(
    uint16_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__min_uint16
(
    uint16_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint16_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__min_uint32
(
    uint32_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__min_uint32
(
    uint32_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint32_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__min_uint64
(
    uint64_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__min_uint64
(
    uint64_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint64_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__min_fp32
(
    float *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__min_fp32
(
    float *restrict Tx,
    int64_t  *restrict Ti,
    const float *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__min_fp64
(
    double *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__min_fp64
(
    double *restrict Tx,
    int64_t  *restrict Ti,
    const double *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__max_int8
(
    int8_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__max_int8
(
    int8_t *restrict Tx,
    int64_t  *restrict Ti,
    const int8_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__max_int16
(
    int16_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__max_int16
(
    int16_t *restrict Tx,
    int64_t  *restrict Ti,
    const int16_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__max_int32
(
    int32_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__max_int32
(
    int32_t *restrict Tx,
    int64_t  *restrict Ti,
    const int32_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__max_int64
(
    int64_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__max_int64
(
    int64_t *restrict Tx,
    int64_t  *restrict Ti,
    const int64_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__max_uint8
(
    uint8_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__max_uint8
(
    uint8_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint8_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__max_uint16
(
    uint16_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__max_uint16
(
    uint16_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint16_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__max_uint32
(
    uint32_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__max_uint32
(
    uint32_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint32_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__max_uint64
(
    uint64_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__max_uint64
(
    uint64_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint64_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__max_fp32
(
    float *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__max_fp32
(
    float *restrict Tx,
    int64_t  *restrict Ti,
    const float *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__max_fp64
(
    double *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__max_fp64
(
    double *restrict Tx,
    int64_t  *restrict Ti,
    const double *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__plus_int8
(
    int8_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__plus_int8
(
    int8_t *restrict Tx,
    int64_t  *restrict Ti,
    const int8_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__plus_int16
(
    int16_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__plus_int16
(
    int16_t *restrict Tx,
    int64_t  *restrict Ti,
    const int16_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__plus_int32
(
    int32_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__plus_int32
(
    int32_t *restrict Tx,
    int64_t  *restrict Ti,
    const int32_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__plus_int64
(
    int64_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__plus_int64
(
    int64_t *restrict Tx,
    int64_t  *restrict Ti,
    const int64_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__plus_uint8
(
    uint8_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__plus_uint8
(
    uint8_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint8_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__plus_uint16
(
    uint16_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__plus_uint16
(
    uint16_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint16_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__plus_uint32
(
    uint32_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__plus_uint32
(
    uint32_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint32_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__plus_uint64
(
    uint64_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__plus_uint64
(
    uint64_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint64_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__plus_fp32
(
    float *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__plus_fp32
(
    float *restrict Tx,
    int64_t  *restrict Ti,
    const float *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__plus_fp64
(
    double *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__plus_fp64
(
    double *restrict Tx,
    int64_t  *restrict Ti,
    const double *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__times_int8
(
    int8_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__times_int8
(
    int8_t *restrict Tx,
    int64_t  *restrict Ti,
    const int8_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__times_int16
(
    int16_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__times_int16
(
    int16_t *restrict Tx,
    int64_t  *restrict Ti,
    const int16_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__times_int32
(
    int32_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__times_int32
(
    int32_t *restrict Tx,
    int64_t  *restrict Ti,
    const int32_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__times_int64
(
    int64_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__times_int64
(
    int64_t *restrict Tx,
    int64_t  *restrict Ti,
    const int64_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__times_uint8
(
    uint8_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__times_uint8
(
    uint8_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint8_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__times_uint16
(
    uint16_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__times_uint16
(
    uint16_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint16_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__times_uint32
(
    uint32_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__times_uint32
(
    uint32_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint32_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__times_uint64
(
    uint64_t *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__times_uint64
(
    uint64_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint64_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__times_fp32
(
    float *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__times_fp32
(
    float *restrict Tx,
    int64_t  *restrict Ti,
    const float *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__times_fp64
(
    double *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__times_fp64
(
    double *restrict Tx,
    int64_t  *restrict Ti,
    const double *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__lor_bool
(
    bool *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__lor_bool
(
    bool *restrict Tx,
    int64_t  *restrict Ti,
    const bool *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__land_bool
(
    bool *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__land_bool
(
    bool *restrict Tx,
    int64_t  *restrict Ti,
    const bool *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__lxor_bool
(
    bool *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__lxor_bool
(
    bool *restrict Tx,
    int64_t  *restrict Ti,
    const bool *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;


void GB_red_scalar__eq_bool
(
    bool *result,
    const GrB_Matrix A,
    int nthreads
) ;


void GB_bild__eq_bool
(
    bool *restrict Tx,
    int64_t  *restrict Ti,
    const bool *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    bool *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__first_bool
(
    bool *restrict Tx,
    int64_t  *restrict Ti,
    const bool *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    int8_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__first_int8
(
    int8_t *restrict Tx,
    int64_t  *restrict Ti,
    const int8_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    int16_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__first_int16
(
    int16_t *restrict Tx,
    int64_t  *restrict Ti,
    const int16_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    int32_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__first_int32
(
    int32_t *restrict Tx,
    int64_t  *restrict Ti,
    const int32_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    int64_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__first_int64
(
    int64_t *restrict Tx,
    int64_t  *restrict Ti,
    const int64_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    uint8_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__first_uint8
(
    uint8_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint8_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    uint16_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__first_uint16
(
    uint16_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint16_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    uint32_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__first_uint32
(
    uint32_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint32_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    uint64_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__first_uint64
(
    uint64_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint64_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    float *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__first_fp32
(
    float *restrict Tx,
    int64_t  *restrict Ti,
    const float *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    double *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__first_fp64
(
    double *restrict Tx,
    int64_t  *restrict Ti,
    const double *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    bool *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__second_bool
(
    bool *restrict Tx,
    int64_t  *restrict Ti,
    const bool *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    int8_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__second_int8
(
    int8_t *restrict Tx,
    int64_t  *restrict Ti,
    const int8_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    int16_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__second_int16
(
    int16_t *restrict Tx,
    int64_t  *restrict Ti,
    const int16_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    int32_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__second_int32
(
    int32_t *restrict Tx,
    int64_t  *restrict Ti,
    const int32_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    int64_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__second_int64
(
    int64_t *restrict Tx,
    int64_t  *restrict Ti,
    const int64_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    uint8_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__second_uint8
(
    uint8_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint8_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    uint16_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__second_uint16
(
    uint16_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint16_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    uint32_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__second_uint32
(
    uint32_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint32_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    uint64_t *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__second_uint64
(
    uint64_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint64_t *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    float *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__second_fp32
(
    float *restrict Tx,
    int64_t  *restrict Ti,
    const float *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

#if 0
void GB_red_scalar__(none)
(
    double *result,
    const GrB_Matrix A,
    int nthreads
) ;
#endif

void GB_bild__second_fp64
(
    double *restrict Tx,
    int64_t  *restrict Ti,
    const double *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

