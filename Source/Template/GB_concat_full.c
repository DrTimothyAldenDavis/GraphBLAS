//------------------------------------------------------------------------------
// GB_concat_full: concatenate an array of matrices into a full matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_FREE_WORK        \
    GB_Matrix_free (&T) ;

#define GB_FREE_ALL         \
    GB_FREE_WORK ;          \
    GB_phbix_free (C) ;

#include "GB_concatenate.h"

GrB_Info GB_concat_full             // concatenate into a full matrix
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix *Tiles,        // 2D array of size rowtiles-by-coltiles,
    const GrB_Index rowtiles,       // in row-major form
    const GrB_Index coltiles,
    const int64_t *GB_RESTRICT Tile_rows,  // size rowtiles+1
    const int64_t *GB_RESTRICT Tile_cols,  // size coltiles+1
    GB_Context Context
)
{ 

    //--------------------------------------------------------------------------
    // allocate C as a full matrix
    //--------------------------------------------------------------------------

    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = GB_clear_static_header (&T_header) ;
    GrB_Matrix A = NULL ;
    GrB_Type ctype = C->type ;
    int64_t cvlen = C->vlen ;
    int64_t cvdim = C->vdim ;
    bool csc = C->is_csc ;
    size_t csize = ctype->size ;
    GrB_Type_code ccode = ctype->code ;
    if (!GB_IS_FULL (C))
    {
        GB_phbix_free (C) ;
        GB_OK (GB_bix_alloc (C, cvlen * cvdim, false, false, false, true,
            Context) ;
    }
    ASSERT (GB_IS_FULL (C)) ;
    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // concatenate all matrices into C
    //--------------------------------------------------------------------------

    for (int64_t outer = 0 ; outer < csc ? coltiles : rowtiles ; outer++)
    {
        for (int64_t inner = 0 ; inner < csc ? rowtiles : coltiles ; inner++)
        {

            //------------------------------------------------------------------
            // get the tile A; transpose and typecast, if needed
            //------------------------------------------------------------------

            A = csc ? GB_TILE (Tiles, inner, outer)
                    : GB_TILE (Tiles, outer, inner) ;
            if (csc != A->is_csc)
            {
                // T = (ctype) A', not in-place
                GB_OK (GB_transpose (&T, ctype, csc, A,
                    NULL, NULL, NULL, false, Context)) ;
                A = T ;
                GB_MATRIX_WAIT (A) ;
            }
            ASSERT (C->is_csc == A->is_csc) ;
            ASSERT (GB_is_dense (A)) ;
            ASSERT (!GB_ANY_PENDING_WORK (A)) ;
            GrB_Type_code tcode = A->type->code ;

            //------------------------------------------------------------------
            // determine where to place the tile in C
            //------------------------------------------------------------------

            // The tile A appears in vectors cvstart:cvend-1 of C, and indices
            // cistart:ciend-1.

            int64_t cvstart, cvend, cistart, ciend ;
            if (csc)
            {
                // C and A are held by column
                // Tiles is row-major and accessed in column order
                cvstart = Tile_cols [outer] ;
                cvend   = Tile_cols [outer+1] ;
                cistart = Tile_rows [inner] ;
                ciend   = Tile_rows [inner+1] ;
            }
            else
            {
                // C and A are held by row
                // Tiles is row-major and accessed in row order
                cvstart = Tile_rows [outer] ;
                cvend   = Tile_rows [outer+1] ;
                cistart = Tile_cols [inner] ;
                ciend   = Tile_cols [inner+1] ;
            }

            int64_t vdim = cvend - cvstart ;
            int64_t vlen = ciend - cistart ;
            ASSERT (vdim == A->vdim) ;
            ASSERT (vlen == A->vlen) ;
            int64_t anz = vdim * vlen ;
            int A_nthreads = GB_nthreads (anz, chunk, nthreads_max) ;

            //------------------------------------------------------------------
            // copy the tile A into C
            //------------------------------------------------------------------

            if (ccode == tcode)
            {
                // no typecasting needed
                switch (csize)
                {
                    #define GB_COPY(pC,pA) Cx [pC] = Ax [pA]

                    case 1 : // uint8, int8, bool, or 1-byte user-defined
                        #define GB_CTYPE uint8_t
                        #include "GB_concat_full_template.c"
                        break ;

                    case 2 : // uint16, int16, or 2-byte user-defined
                        #define GB_CTYPE uint16_t
                        #include "GB_concat_full_template.c"
                        break ;

                    case 4 : // uint32, int32, float, or 4-byte user-defined
                        #define GB_CTYPE uint32_t
                        #include "GB_concat_full_template.c"
                        break ;

                    case 8 : // uint64, int64, double, float complex,
                             // or 8-byte user defined
                        #define GB_CTYPE uint64_t
                        #include "GB_concat_full_template.c"
                        break ;

                    case 16 : // double complex or 16-byte user-defined
                        #define GB_CTYPE uint64_t
                        #undef  GB_COPY
                        #define GB_COPY(pC,pA)                      \
                            Cx [2*pC  ] = Ax [2*pA  ] ;          \
                            Cx [2*pC+1] = Ax [2*pA+1] ;
                        #include "GB_concat_full_template.c"
                        break ;

                    default : // user-defined of a different size
                        #define GB_CTYPE GB_void
                        #undef  GB_COPY
                        #define GB_COPY(pC,pA)                      \
                            memcpy (Cx + pC, Ax + pA, csize) ;
                        #include "GB_concat_full_template.c"
                        break ;
                }
            }
            else
            {
                // with typecasting (not for user-defined types)
                GB_cast_function cast_T_to_C = GB_cast_factory (ccode, tcode) ;
                #undef  GB_COPY
                #define GB_COPY(pC,pA)  \
                    cast_T_to_C (Cx + (pC)*csize, Ax + (pA)*csize, csize) ;
                #include "GB_concat_full_template.c"
            }

            GB_FREE_WORK ;
        }
    }

    C->magic = GB_MAGIC ;
    return (GrB_SUCCESS) ;
}
