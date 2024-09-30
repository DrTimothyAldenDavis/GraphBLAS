//------------------------------------------------------------------------------
// GB_AxB_saxbit_generic_idx_flipped.c: C=A*B, C bitmap/full, flipped idx mult
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap only.
// multiply op is index flipped

#define GB_AXB_SAXPY_GENERIC_METHOD GB_AxB_saxbit_generic_idx_flipped 
#define GB_GENERIC_C_IS_SPARSE_OR_HYPERSPARSE  0
#define GB_GENERIC_OP_IS_POSITIONAL            0
#define GB_GENERIC_FLIPXY                      0
#define GB_GENERIC_NOFLIPXY                    0
#define GB_GENERIC_IDX_FLIPXY                  1
#define GB_GENERIC_IDX_NOFLIPXY                0
#define GB_GENERIC_OP_IS_INT64                 0
#define GB_GENERIC_OP_IS_FIRSTI                0
#define GB_GENERIC_OP_IS_FIRSTJ                0
#define GB_GENERIC_OP_IS_FIRST                 0
#define GB_GENERIC_OP_IS_SECOND                0

#include "mxm/factory/GB_AxB_saxpy_generic_method.c"

