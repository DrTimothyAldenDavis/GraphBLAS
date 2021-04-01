//------------------------------------------------------------------------------
// GxB_concatenate: concatenate an array of matrices into a single matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_WORK                        \
    GB_WERK_POP (Tile_cols, int64_t) ;      \
    GB_WERK_POP (Tile_rows, int64_t) ;

#define GB_FREE_ALL                         \
    GB_FREE_WORK ;

#include "GB_concatenate.h"

GrB_Info GxB_concatenate            // concatenate a 2D array of matrices
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix *Tiles,        // 2D array of size rowtiles-by-coltiles,
    const GrB_Index rowtiles,       // in row-major order
    const GrB_Index coltiles,
    const GrB_Descriptor desc       // unused, except threading control
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GxB_concatenate (C, Tiles, rowtiles, coltiles, desc)") ;
    GB_BURBLE_START ("GxB_concatenate") ;
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    if (rowtiles <= 0 || coltiles <= 0)
    {
        GB_ERROR (GrB_INVALID_VALUE, "rowtiles and coltiles must be > 0") ;
    }
    GB_RETURN_IF_NULL (Tiles) ;
    for (int64_t k = 0 ; k < rowtiles*coltiles ; k++)
    {
        GrB_Matrix A = Tiles [k] ;
        GB_RETURN_IF_NULL_OR_FAULTY (A) ;
        GB_MATRIX_WAIT (A) ;
    }

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // check the sizes and types of each tile
    //--------------------------------------------------------------------------

    bool C_is_full = true ;

    GB_WERK_DECLARE (Tile_rows, int64_t) ;
    GB_WERK_DECLARE (Tile_cols, int64_t) ;
    GB_WERK_PUSH (Tile_rows, rowtiles+1, int64_t) ;
    GB_WERK_PUSH (Tile_cols, coltiles+1, int64_t) ;

    for (int64_t i = 0 ; i < rowtiles ; i++)
    {
        GrB_Matrix A = GB_TILE (Tiles, i, 0) ;
        int64_t nrows = GB_NROWS (A) ;
        Tile_rows [i] = nrows ;
    }

    for (int64_t j = 0 ; j < coltiles ; j++)
    {
        GrB_Matrix A = GB_TILE (Tiles, 0, j) ;
        int64_t ncols = GB_NCOLS (A) ;
        Tile_cols [j] = ncols ;
        cncols += ncols ;
    }

    // replace Tile_rows and Tile_cols with their cumulative sum
    GB_cumsum (Tile_rows, rowtiles, NULL, 1, Context) ;
    GB_cumsum (Tile_cols, coltiles, NULL, 1, Context) ;
    int64_t cnrows = Tile_rows [rowtiles] ;
    int64_t cncols = Tile_cols [coltiles] ;

    if (cnrows != GB_NROWS (C) || cncols != GB_NCOLS (C))
    {
        GB_FREE_ALL ;
        GB_ERROR (GrB_DIMENSION_MISMATCH,
            "C is %ld-by-%ld but Tiles{:,:} is %ld-by-%ld\n",
            GB_NROWS (C), GB_NCOLS (C), cnrows, cncols) ;
    }

    int64_t cnz = 0 ;
    int64_t k = 0 ;
    for (int64_t i = 0 ; i < rowtiles ; i++)
    {
        for (int64_t j = 0 ; j < coltiles ; j++)
        {
            GrB_Matrix A = GB_TILE (Tiles, i, j) ;
            // C is full only if all A(i,j) are entirely dense 
            C_is_full = C_is_full && GB_is_dense (A) ;
            int64_t nrows = GB_NROWS (A) ;
            int64_t ncols = GB_NCOLS (A) ;
            cnz += GB_NNZ (A) ;
            if (GB_IS_HYPERSPARSE (A))
            {
                k += A->nvec ;
            }
            else
            {
                k += GB_IMAX (nrows, ncols) ;
            }
            if (!GB_Type_compatible (C->type, A->type))
            {
                GB_FREE_ALL ;
                int64_t offset = GB_Global_print_one_based_get ( ) ? 1 : 0 ;
                GB_ERROR (GrB_DOMAIN_MISMATCH,
                    "Input matrix Tiles{%ld,%ld} of type [%s]\n"
                    "cannot be typecast to output of type [%s]\n",
                    i+offset, j+offset, A->type->name, C->type->name) ;
            }
            if (Tile_rows [i] != nrows)
            {
                GB_FREE_ALL ;
                int64_t offset = GB_Global_print_one_based_get ( ) ? 1 : 0 ;
                GB_ERROR (GrB_DIMENSION_MISMATCH,
                    "Input matrix Tiles{%ld,%ld} is %ld-by-%ld; its row\n"
                    "dimension must match all other matrices Tiles{%ld,:},"
                    " which is %ld\n", i+offset, j+offset, nrows, ncols,
                    i+offset, Tile_rows [i]) ;
            }
            if (Tile_cols [i] != ncols)
            {
                GB_FREE_ALL ;
                int64_t offset = GB_Global_print_one_based_get ( ) ? 1 : 0 ;
                GB_ERROR (GrB_DIMENSION_MISMATCH,
                    "Input matrix Tiles{%ld,%ld} is %ld-by-%ld; its column\n"
                    "dimension must match all other matrices Tiles{:,%ld},"
                    " which is %ld\n", i+offset, j+offset, nrows, ncols,
                    j+offset, Tile_cols [j]) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // C = concatenate (Tiles)
    //--------------------------------------------------------------------------

    if (C_is_full)
    {
        // construct C as full
        info = GB_concat_full (C, Tiles, rowtiles, coltiles,
            Tile_rows, Tile_cols, Context) ;
    }
    else if (GB_convert_sparse_to_bitmap_test (C->bitmap_switch, cnz, cnrows,
        cncols))
    {
        // construct C as bitmap
        info = GB_concat_bitmap (C, cnz, Tiles, rowtiles, coltiles,
            Tile_rows, Tile_cols, Context) ;
    }
    else if (GB_convert_sparse_to_hyper_test (C->hyper_switch, k, C->vdim))
    {
        // construct C as hypersparse
        info = GB_concat_hyper (C, cnz, Tiles, rowtiles, coltiles,
            Tile_rows, Tile_cols, Context) ;
    }
    else
    {
        // construct C as sparse
        info = GB_concat_sparse (C, Tiles, rowtiles, coltiles,
            Tile_rows, Tile_cols, Context) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK ;
    GB_BURBLE_END ;
    return (info) ;
}

