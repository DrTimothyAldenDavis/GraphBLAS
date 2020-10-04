
#ifndef GB_AXB_SAXPY_H
#define GB_AXB_SAXPY_H
#include "GB.h"

GrB_Info GB_AxB_saxpy               // C = A*B using Gustavson/Hash/Bitmap
(
    GrB_Matrix *Chandle,            // output matrix (if not done in-place)
    const GrB_Matrix M,             // optional mask matrix
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *mask_applied,             // if true, then mask was applied
    const GrB_Desc_Value AxB_method,
    GB_Context Context
) ;

int GB_AxB_saxpy_sparsity           // return the sparsity structure for C
(
    // input:
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // if true, use !M
    const GrB_Matrix A,             // input A matrix
    const GrB_Matrix B              // input B matrix
) ;

#endif

