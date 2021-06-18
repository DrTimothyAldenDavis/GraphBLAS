//------------------------------------------------------------------------------
// GB_mex_about3: still more basic tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Test lots of random stuff.  The function otherwise serves no purpose.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_about3"

int myprintf (const char *restrict format, ...) ;

int myprintf (const char *restrict format, ...)
{
    printf ("[[myprintf:") ;
    va_list ap ;
    va_start (ap, format) ;
    vprintf (format, ap) ;
    va_end (ap) ;
    printf ("]]") ;
}

int myflush (void) ;

int myflush (void)
{
    printf ("myflush\n") ;
    fflush (stdout) ;
}

typedef int (* printf_func_t) (const char *restrict format, ...) ;
typedef int (* flush_func_t)  (void) ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    GrB_Matrix C = NULL ;
    GrB_Vector w = NULL ;
    GrB_Type myint = NULL, myblob = NULL ;
    GB_void *Null = NULL ;
    char *err ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    bool malloc_debug = GB_mx_get_global (true) ;
    FILE *f = fopen ("errlog4.txt", "w") ;
    int expected = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // GxB_set/get for printf and flush
    //--------------------------------------------------------------------------

    OK (GxB_Global_Option_set (GxB_BURBLE, true)) ;
    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;

    printf ("\nBurble with standard printf/flush:\n") ;
    GrB_Index nvals ;
    OK (GrB_Matrix_nvals (&nvals, C)) ;
    CHECK (nvals == 0) ;

    OK (GxB_Global_Option_set (GxB_PRINTF, myprintf)) ;
    OK (GxB_Global_Option_set (GxB_FLUSH, myflush)) ;

    printf_func_t mypr ;
    OK (GxB_Global_Option_get (GxB_PRINTF, &mypr)) ;
    CHECK (mypr == myprintf) ;

    flush_func_t myfl ;
    OK (GxB_Global_Option_get (GxB_FLUSH, &myfl)) ;
    CHECK (myfl == myflush) ;

    printf ("\nBurble with myprintf/myflush:\n") ;
    OK (GrB_Matrix_nvals (&nvals, C)) ;
    CHECK (nvals == 0) ;
    OK (GxB_Global_Option_set (GxB_BURBLE, false)) ;

    OK (GxB_Global_Option_set (GxB_PRINTF, printf)) ;
    OK (GxB_Global_Option_set (GxB_FLUSH, NULL)) ;

    //--------------------------------------------------------------------------
    // test GxB_set/get for free_pool_limit
    //--------------------------------------------------------------------------

    int64_t free_pool_limit [64] ;
    OK (GxB_Global_Option_set (GxB_MEMORY_POOL, NULL)) ;
    OK (GxB_Global_Option_get (GxB_MEMORY_POOL, free_pool_limit)) ;
    printf ("\ndefault memory pool limits:\n") ;
    for (int k = 0 ; k < 64 ; k++)
    {
        if (free_pool_limit [k] > 0)
        {
            printf ("pool %2d: limit %ld\n", k, free_pool_limit [k]) ;
        }
    }
    for (int k = 0 ; k < 64 ; k++)
    {
        free_pool_limit [k] = k ;
    }
    OK (GxB_Global_Option_set (GxB_MEMORY_POOL, free_pool_limit)) ;
    OK (GxB_Global_Option_get (GxB_MEMORY_POOL, free_pool_limit)) ;
    for (int k = 0 ; k < 3 ; k++)
    {
        CHECK (free_pool_limit [k] == 0) ;
    }
    for (int k = 3 ; k < 64 ; k++)
    {
        CHECK (free_pool_limit [k] == k) ;
    }
    for (int k = 0 ; k < 64 ; k++)
    {
        free_pool_limit [k] = 0 ;
    }
    OK (GxB_Global_Option_set (GxB_MEMORY_POOL, free_pool_limit)) ;
    OK (GxB_Global_Option_get (GxB_MEMORY_POOL, free_pool_limit)) ;
    for (int k = 0 ; k < 64 ; k++)
    {
        CHECK (free_pool_limit [k] == 0) ;
    }

    //--------------------------------------------------------------------------
    // GrB_reduce with invalid binary op
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (&w, GrB_FP32, 10)) ;
    info = GrB_Matrix_reduce_BinaryOp (w, NULL, NULL, GrB_LT_FP32, C, NULL) ;
    CHECK (info == GrB_DOMAIN_MISMATCH) ;
    const char *s ;
    OK (GrB_error (&s, w)) ;
    printf ("expected error: [%s]\n", s) ;
    GrB_Vector_free_(&w) ;

    //--------------------------------------------------------------------------
    // GB_nnz_held, GB_is_shallow
    //--------------------------------------------------------------------------

    CHECK (GB_nnz_held (NULL) == 0) ;
    CHECK (!GB_is_shallow (NULL)) ;

    //--------------------------------------------------------------------------
    // invalid iso matrix
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_fprint (C, "C ok", GxB_COMPLETE, NULL)) ;
    void *save = C->x ;
    C->x = NULL ;
    C->iso = true ;
    expected = GrB_INVALID_OBJECT ;
    ERR (GxB_Matrix_fprint (C, "C iso invald", GxB_COMPLETE, NULL)) ;
    C->x = save ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // empty scalar for iso build
    //--------------------------------------------------------------------------

    GrB_Index I [4] = { 1, 2, 3, 4 } ;
    GxB_Scalar scalar = NULL ;
    OK (GxB_Scalar_new (&scalar, GrB_FP32)) ;
    OK (GxB_Scalar_fprint (scalar, "scalar init", GxB_COMPLETE, NULL)) ;
    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Vector_new (&w, GrB_FP32, 10)) ;
    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Matrix_build_Scalar (C, I, I, scalar, 4)) ;
    OK (GrB_error (&s, C)) ;
    printf ("expected error: [%s]\n", s) ;

    ERR (GxB_Vector_build_Scalar (w, I, scalar, 4)) ;
    OK (GrB_error (&s, w)) ;
    printf ("expected error: [%s]\n", s) ;

    //--------------------------------------------------------------------------
    // build error handling
    //--------------------------------------------------------------------------

    GrB_Matrix_free_(&C) ;
    OK (GrB_Type_new (&myint, sizeof (int))) ;
    OK (GrB_Matrix_new (&C, myint, 10, 10)) ;
    OK (GxB_Scalar_setElement_FP32 (scalar, 3.0)) ;
    OK (GxB_Scalar_fprint (scalar, "scalar set", GxB_COMPLETE, NULL)) ;
    OK (GxB_Matrix_Option_set ((GrB_Matrix) scalar, GxB_SPARSITY_CONTROL,
        GxB_SPARSE)) ;
    scalar->jumbled = true ;
    OK (GxB_Scalar_wait (&scalar)) ;

    OK (GxB_Scalar_fprint (scalar, "scalar", GxB_COMPLETE, NULL)) ;

    expected = GrB_DOMAIN_MISMATCH ;
    ERR (GrB_Matrix_build_UINT64 (C, I, I, I, 4, GrB_PLUS_UINT64)) ;
    OK (GrB_error (&s, C)) ;
    printf ("expected error: [%s]\n", s) ;

    ERR (GxB_Matrix_build_Scalar (C, I, I, scalar, 4)) ;
    OK (GrB_error (&s, C)) ;
    printf ("expected error: [%s]\n", s) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // import/export
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    for (int k = 0 ; k < 8 ; k++)
    {
        OK (GrB_Matrix_setElement_FP32 (C, 1, k, k+1)) ;
    }

    GrB_Type type ;
    GrB_Index nrows, ncols, Ap_size, Ai_size, Ax_size, Ah_size, nvec ;
    GrB_Index *Ap = NULL, *Ai = NULL, *Ah = NULL ;
    float *Ax = NULL ;
    bool iso, jumbled ;
    OK (GrB_Matrix_wait (&C)) ;
    OK (GxB_Matrix_fprint (C, "C to export", GxB_COMPLETE, NULL)) ;

    // export as CSC
    OK (GxB_Matrix_export_CSC (&C, &type, &nrows, &ncols, &Ap, &Ai, &Ax,
        &Ap_size, &Ai_size, &Ax_size, &iso, &jumbled, NULL)) ;

    // import as CSC
    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai, &Ax,
        0, Ai_size, Ax_size, iso, jumbled, NULL)) ;
    ERR (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai, &Ax,
        Ap_size, 0, Ax_size, iso, jumbled, NULL)) ;
    ERR (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai, &Ax,
        Ap_size, Ai_size, 0, iso, jumbled, NULL)) ;
    ERR (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai, &Null,
        Ap_size, Ai_size, 0, true, jumbled, NULL)) ;
    OK (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai, &Ax,
        Ap_size, Ai_size, Ax_size, iso, jumbled, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C imported sparse", GxB_COMPLETE, NULL)) ;

    // export as HyperCSC
    OK (GxB_Matrix_export_HyperCSC (&C, &type, &nrows, &ncols,
        &Ap, &Ah, &Ai, &Ax,
        &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso, &nvec, &jumbled, NULL)) ;

    // import as HyperCSC
    ERR (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols,
        &Ap, &Ah, &Ai, &Ax,
        0, Ah_size, Ai_size, Ax_size, iso, nvec, jumbled, NULL)) ;
    ERR (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols,
        &Ap, &Ah, &Ai, &Ax,
        Ap_size, 0, Ai_size, Ax_size, iso, nvec, jumbled, NULL)) ;
    ERR (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols,
        &Ap, &Ah, &Ai, &Ax,
        Ap_size, Ah_size, 0, Ax_size, iso, nvec, jumbled, NULL)) ;
    ERR (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols,
        &Ap, &Ah, &Ai, &Ax,
        Ap_size, Ah_size, Ai_size, 0, iso, nvec, jumbled, NULL)) ;
    OK (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols,
        &Ap, &Ah, &Ai, &Ax,
        Ap_size, Ah_size, Ai_size, Ax_size, iso, nvec, jumbled, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C imported hyper", GxB_COMPLETE, NULL)) ;
    GrB_Matrix_free_(&C) ;

    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Matrix_assign_FP32 (C, NULL, NULL, 1, GrB_ALL, 10, GrB_ALL, 10,
        NULL)) ;

    // export as CSC, non-iso
    OK (GxB_Matrix_export_CSC (&C, &type, &nrows, &ncols, &Ap, &Ai, &Ax,
        &Ap_size, &Ai_size, &Ax_size, NULL, &jumbled, NULL)) ;

    OK (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai, &Ax,
        Ap_size, Ai_size, Ax_size, false, jumbled, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C imported non-iso", GxB_COMPLETE, NULL)) ;
    OK (GrB_Matrix_free_(&C)) ;

    //--------------------------------------------------------------------------
    // split a user-defined full matrix
    //--------------------------------------------------------------------------

    typedef struct
    {
        int64_t blob [4] ;
    }
    myblob_struct ;

    OK (GrB_Type_new (&myblob, sizeof (myblob_struct))) ;
    OK (GxB_Type_fprint (myblob, "myblob", GxB_COMPLETE, NULL)) ;
    myblob_struct blob_scalar ;
    OK (GrB_Matrix_new (&C, myblob, 4, 4)) ;

    for (int i = 0 ; i < 4 ; i++)
    {
        for (int j = 0 ; j < 4 ; j++)
        {
            blob_scalar.blob [0] = i ;
            blob_scalar.blob [1] = j ;
            blob_scalar.blob [2] = 32 ;
            blob_scalar.blob [3] = 99 ;
            OK (GrB_Matrix_setElement_UDT (C, &blob_scalar, i, j)) ;
        }
    }
    OK (GrB_Matrix_wait (&C)) ;
    OK (GxB_Matrix_fprint (C, "C blob", GxB_COMPLETE, NULL)) ;

    for (int i = 0 ; i < 4 ; i++)
    {
        for (int j = 0 ; j < 4 ; j++)
        {
            OK (GrB_Matrix_extractElement_UDT (&blob_scalar, C, i, j)) ;
            CHECK (blob_scalar.blob [0] == i) ;
            CHECK (blob_scalar.blob [1] == j) ;
            CHECK (blob_scalar.blob [2] == 32) ;
            CHECK (blob_scalar.blob [3] == 99) ;
            printf ("C(%d,%d) = [%d, %d, %d, %d]\n", i, j,
                blob_scalar.blob [0], blob_scalar.blob [1],
                blob_scalar.blob [2], blob_scalar.blob [3]) ;
        }
    }

    GrB_Matrix Tiles [4]  = { NULL, NULL, NULL, NULL} ;
    GrB_Index Tile_nrows [2] = { 2, 2 } ;
    GrB_Index Tile_ncols [2] = { 2, 2 } ;
    OK (GxB_Matrix_split (Tiles, 2, 2, Tile_nrows, Tile_ncols, C, NULL)) ;

    for (int k = 0 ; k < 4 ; k++)
    {
        printf ("\n================ Tile %d\n", k) ;
        OK (GxB_Matrix_fprint (Tiles [k], "Tile", GxB_COMPLETE, NULL)) ;
        int istart = (k == 0 || k == 1) ? 0 : 2 ;
        int jstart = (k == 0 || k == 2) ? 0 : 2 ;
        for (int i = 0 ; i < 2 ; i++)
        {
            for (int j = 0 ; j < 2 ; j++)
            {
                OK (GrB_Matrix_extractElement_UDT (&blob_scalar,
                    Tiles [k], i, j)) ;
                printf ("Tile(%d,%d) = [%d, %d, %d, %d]\n", i, j,
                    blob_scalar.blob [0], blob_scalar.blob [1],
                    blob_scalar.blob [2], blob_scalar.blob [3]) ;

                CHECK (blob_scalar.blob [0] == i + istart) ;
                CHECK (blob_scalar.blob [1] == j + jstart) ;
                CHECK (blob_scalar.blob [2] == 32) ;
                CHECK (blob_scalar.blob [3] == 99) ;
            }
        }
        OK (GrB_Matrix_free_(& (Tiles [k]))) ;
    }

    // create an iso matrix
    OK (GrB_Matrix_assign_UDT (C, NULL, NULL, (void *) &blob_scalar,
        GrB_ALL, 10, GrB_ALL, 10, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C blob iso", GxB_COMPLETE, NULL)) ;

    // export as FullC, non-iso
    OK (GxB_Matrix_export_FullC (&C, &type, &nrows, &ncols, &Ax, &Ax_size,
        NULL, NULL)) ;

    // import as FullC, non-iso
    OK (GxB_Matrix_import_FullC (&C, type, nrows, ncols, &Ax, Ax_size,
        false, NULL)) ;

    OK (GxB_Matrix_fprint (C, "C blob iso imported", GxB_COMPLETE, NULL)) ;

    for (int i = 0 ; i < 4 ; i++)
    {
        for (int j = 0 ; j < 4 ; j++)
        {
            OK (GrB_Matrix_extractElement_UDT (&blob_scalar, C, i, j)) ;
            printf ("C(%d,%d) = [%d, %d, %d, %d]\n", i, j,
                blob_scalar.blob [0], blob_scalar.blob [1],
                blob_scalar.blob [2], blob_scalar.blob [3]) ;
        }
    }

    // change to iso sparse, and test GB_ix_realloc
    OK (GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GrB_Matrix_assign_UDT (C, NULL, NULL, (void *) &blob_scalar,
        GrB_ALL, 10, GrB_ALL, 10, NULL)) ;
    OK (GB_ix_realloc (C, 32, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C blob sparse non-iso", GxB_COMPLETE, NULL)) ;

    // test wait on jumbled matrix (non-iso)
    OK (GrB_Matrix_setElement_UDT (C, &blob_scalar, 0, 0)) ;
    blob_scalar.blob [0] = 1007 ;
    OK (GrB_Matrix_setElement_UDT (C, &blob_scalar, 1, 1)) ;
    C->jumbled = true ;
    OK (GxB_Matrix_fprint (C, "C blob jumbled", GxB_COMPLETE, NULL)) ;
    OK (GrB_Matrix_wait (&C)) ;
    OK (GxB_Matrix_fprint (C, "C blob wait", GxB_COMPLETE, NULL)) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // setElement
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Matrix_assign_FP32 (C, NULL, NULL, 1, GrB_ALL, 10, GrB_ALL, 10,
        NULL)) ;
    OK (GxB_Matrix_fprint (C, "C iso full", GxB_COMPLETE, NULL)) ;
    OK (GrB_Matrix_setElement_FP32 (C, 2, 0, 0)) ;
    OK (GxB_Matrix_fprint (C, "C non-iso full", GxB_COMPLETE, NULL)) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GxB_Scalar_free_(&scalar) ;
    GrB_Vector_free_(&w) ;
    GrB_Matrix_free_(&C) ;
    GrB_Type_free_(&myint) ;
    GrB_Type_free_(&myblob) ;
    GB_mx_put_global (true) ;   
    fclose (f) ;
    printf ("\nGB_mex_about3: all tests passed\n\n") ;
}

