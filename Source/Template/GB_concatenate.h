//------------------------------------------------------------------------------
// GB_concatenate.h: definitions for GxB_concatenate
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CONCATENATE_H
#define GB_CONCATENATE_H

#define GB_TILE(Tiles,i,j) (*(Tiles + (i) * coltiles + (j)))

GrB_Info GB_concat_full             // concatenate into a full matrix
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix *Tiles,        // 2D array of size rowtiles-by-coltiles,
    const GrB_Index rowtiles,       // in row-major form
    const GrB_Index coltiles,
    const int64_t *GB_RESTRICT Tile_rows,  // size rowtiles+1
    const int64_t *GB_RESTRICT Tile_cols,  // size coltiles+1
    GB_Context Context
) ;

#endif

