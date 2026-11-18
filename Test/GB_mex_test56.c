//------------------------------------------------------------------------------
// GB_mex_test56: per-matrix integer hints persist across operations
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GxB_ROWINDEX_INTEGER_HINT, GxB_COLINDEX_INTEGER_HINT, and
// GxB_OFFSET_INTEGER_HINT can be set on each matrix (0 means the global
// setting is used).  The hint is stored in the matrix, and it should remain
// set when a later operation computes a new result in that matrix.  The hint
// is a request, but most operations honor it, so the integer sizes of the
// result are checked as well, except for GrB_Matrix_clear,
// GxB_Matrix_pack_*, and GxB_load_Matrix_from_Container.

// For each operation below, a matrix A is created with its row index and
// offset hints set to h (32 or 64), and its column index hint set to the other
// value, g.  The global hints are all set to g.  The operation then computes a
// new result in A (or in a copy C of A), and the test checks that the hints of
// the result are unchanged.  The row and column index hints differ, so that a
// swap of the two is detected.

// As of v10.6.0 (dev2), the hints are reset to 0 by GrB_assign (C = B, with no
// mask), GxB_Matrix_reshape, GxB_Matrix_concat, GxB_Matrix_diag,
// GxB_Matrix_pack_*, GrB_Matrix_dup (on the new matrix), and GxB_Matrix_sort
// (if A is already held in the orientation being sorted).  Each of these
// creates the new content of C with GB_new or GB_new_bix on the existing
// header of C, which sets the hints to 0.  Otherwise, GxB_Matrix_sort changes
// the format of C to the orientation being sorted (with GB_transpose_cast),
// without swapping the row and column index hints of C.

// As of v10.6.0 (dev2), the hints are kept but not honored by:
//
//  GrB_Matrix_setElement into an empty or cleared matrix, then GrB_wait:
//      GB_wait.c builds the pending tuples with 32-bit integers.
//  GrB_apply, GrB_assign (C<B,struct> = B), and GxB_Matrix_sort: the result
//      takes the integer sizes of its input.
//  GxB_FORMAT: the integer sizes of the row and column indices are swapped,
//      relative to the hints.
//
// The integer sizes are not checked for these cases, which may be by design:
//
//  GrB_Matrix_clear: uses the global hints (GB_WHERE_1).
//  GxB_Matrix_pack_*: the packed arrays are 64-bit.
//  GxB_load_Matrix_from_Container: the integer size of the row indices of a
//      sparse CSR matrix is not kept; there is no A->h array.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL                            \
{                                           \
    GrB_Matrix_free (&A) ;                  \
    GrB_Matrix_free (&B) ;                  \
    GrB_Matrix_free (&C) ;                  \
    GrB_Matrix_free (&S) ;                  \
    for (int t = 0 ; t < 4 ; t++)           \
    {                                       \
        GrB_Matrix_free (&(Tiles [t])) ;    \
    }                                       \
    GrB_Vector_free (&v) ;                  \
    GrB_Scalar_free (&s) ;                  \
    GxB_Container_free (&Container) ;       \
    if (blob != NULL) mxFree (blob) ;       \
    blob = NULL ;                           \
}

// A, B, and C are N-by-N; S is NS-by-NS, so kron (S,S) is N-by-N
#define N  100
#define NZ 500
#define NS 10

// a few tuples for GrB_Matrix_build
static GrB_Index Ilist [3] = { 0, 4, 9 } ;
static GrB_Index Jlist [3] = { 2, 4, 7 } ;
static double    Xlist [3] = { 1, 2, 3 } ;

// the operations to test
#define NOPS 31
static const char *op_name [NOPS] =
{
    "GrB_Matrix_build",
    "GrB_Matrix_setElement into an empty matrix",
    "GrB_Matrix_setElement into a non-empty matrix",
    "GrB_Matrix_removeElement",
    "GrB_Matrix_clear",
    "GrB_Matrix_clear then GrB_Matrix_setElement",
    "GrB_mxm",
    "GrB_mxm with accum",
    "GrB_eWiseAdd",
    "GrB_eWiseMult",
    "GrB_select",
    "GrB_transpose",
    "GrB_apply",
    "GrB_extract",
    "GrB_kronecker",
    "GrB_assign (C<B,struct> = B)",
    "GrB_assign (C = B)",
    "GrB_assign (C = B) with replace",
    "GrB_assign (C(I,J) = scalar)",
    "GrB_Matrix_resize",
    "GxB_Matrix_reshape",
    "GxB_Matrix_sort",
    "GxB_Matrix_concat",
    "GxB_Matrix_diag",
    "GxB_Matrix_build_Scalar",
    "GxB_SPARSITY_CONTROL: to bitmap and back",
    "GxB_FORMAT: change the format",
    "GxB_unload_Matrix_into_Container and GxB_load_Matrix_from_Container",
    "GxB_Matrix_unpack_CSC/CSR and GxB_Matrix_pack_CSC/CSR",
    "GrB_Matrix_dup",
    "GxB_Matrix_serialize and GxB_Matrix_deserialize",
} ;

//------------------------------------------------------------------------------
// new_matrix: create an empty matrix with integer hints of h, g, and h
//------------------------------------------------------------------------------

static void new_matrix
(
    GrB_Matrix *A_handle,
    bool by_col,
    bool hyper,
    int h,              // row index and offset hint
    int g               // column index hint
)
{
    GrB_Info info ;
    GrB_Matrix A = NULL ;
    OK (GrB_Matrix_new (&A, GrB_FP64, N, N)) ;
    OK (GrB_Matrix_set_INT32 (A, by_col ? GxB_BY_COL : GxB_BY_ROW,
        GxB_FORMAT)) ;
    OK (GrB_Matrix_set_INT32 (A, hyper ? GxB_HYPERSPARSE : GxB_SPARSE,
        GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_set_INT32 (A, h, GxB_ROWINDEX_INTEGER_HINT)) ;
    OK (GrB_Matrix_set_INT32 (A, g, GxB_COLINDEX_INTEGER_HINT)) ;
    OK (GrB_Matrix_set_INT32 (A, h, GxB_OFFSET_INTEGER_HINT)) ;
    (*A_handle) = A ;
}

//------------------------------------------------------------------------------
// fill: add a few entries to A
//------------------------------------------------------------------------------

static void fill (GrB_Matrix A)
{
    GrB_Info info ;
    OK (GrB_Matrix_build_FP64 (A, Ilist, Jlist, Xlist, 3, NULL)) ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
}

//------------------------------------------------------------------------------
// check_hints: finish all work on A and check its hints
//------------------------------------------------------------------------------

static bool check_hints
(
    GrB_Matrix A,
    int h,              // expected row index and offset hint
    int g,              // expected column index hint
    bool check_bits,    // if true, also check the integer sizes
    const char *what
)
{
    GrB_Info info ;
    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    int32_t rh, ch, ph ;
    OK (GrB_Matrix_get_INT32 (A, &rh, GxB_ROWINDEX_INTEGER_HINT)) ;
    OK (GrB_Matrix_get_INT32 (A, &ch, GxB_COLINDEX_INTEGER_HINT)) ;
    OK (GrB_Matrix_get_INT32 (A, &ph, GxB_OFFSET_INTEGER_HINT)) ;
    bool ok1 = (rh == h) && (ch == g) && (ph == h) ;
    if (!ok1)
    {
        printf ("%d : %s: expected hints row %d col %d offset %d, got row %d "
            "col %d offset %d\n", ok1, what, h, g, h, rh, ch, ph) ;
    }
    // CHECK (ok1) ;

    if (!check_bits) return (ok1) ;

    // check that the hints are honored: that the row index and offset
    // integers of A have h bits, and the column index integers have g bits
    int32_t rb, cb, pb ;
    OK (GrB_Matrix_get_INT32 (A, &rb, GxB_ROWINDEX_INTEGER_BITS)) ;
    OK (GrB_Matrix_get_INT32 (A, &cb, GxB_COLINDEX_INTEGER_BITS)) ;
    OK (GrB_Matrix_get_INT32 (A, &pb, GxB_OFFSET_INTEGER_BITS)) ;
    bool ok2 = (rb == h) && (cb == g) && (pb == h) ;
    if (!ok2)
    {
        printf ("%d : %s: expected integers of row %d col %d offset %d bits, "
            "got row %d col %d offset %d\n", ok2, what, h, g, h, rb, cb, pb) ;
    }
    return (ok1 && ok2) ;
    // CHECK (ok) ;
}

//------------------------------------------------------------------------------
// GB_mex_test56 mexFunction
//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;
    GrB_Matrix A = NULL, B = NULL, C = NULL, S = NULL ;
    GrB_Matrix Tiles [4] = { NULL, NULL, NULL, NULL } ;
    GrB_Vector v = NULL ;
    GrB_Scalar s = NULL ;
    GxB_Container Container = NULL ;
    void *blob = NULL ;
    GrB_Index blob_size = 0 ;
    int ntests = 0 ;
    int npass = 0 ;

    // save the global hints
    int32_t global_r, global_c, global_p ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &global_r,
        GxB_ROWINDEX_INTEGER_HINT)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &global_c,
        GxB_COLINDEX_INTEGER_HINT)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &global_p,
        GxB_OFFSET_INTEGER_HINT)) ;

    // tile sizes for GxB_Matrix_split and GxB_Matrix_concat
    GrB_Index Tile_size [2] = { N/2, N/2 } ;

    // an iso value for GxB_Matrix_build_Scalar
    OK (GrB_Scalar_new (&s, GrB_FP64)) ;
    OK (GrB_Scalar_setElement_FP64 (s, 2)) ;

    //--------------------------------------------------------------------------
    // test each operation
    //--------------------------------------------------------------------------

    for (int h = 32 ; h <= 64 ; h += 32)
    for (int by_col = 0 ; by_col <= 1 ; by_col++)
    for (int hyper = 0 ; hyper <= 1 ; hyper++)
    {

        // the global hints, and the column index hint of A, are g
        int g = (h == 32) ? 64 : 32 ;
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, g, GxB_ROWINDEX_INTEGER_HINT)) ;
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, g, GxB_COLINDEX_INTEGER_HINT)) ;
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, g, GxB_OFFSET_INTEGER_HINT)) ;

        // B: an N-by-N random matrix, and S: an NS-by-NS random matrix
        simple_rand_seed (1) ;
        OK (GB_mx_random_matrix (&B, false, false, N, N, NZ, 1, false)) ;
        OK (GrB_Matrix_set_INT32 (B, by_col ? GxB_BY_COL : GxB_BY_ROW,
            GxB_FORMAT)) ;
        OK (GrB_Matrix_wait (B, GrB_MATERIALIZE)) ;
        OK (GB_mx_random_matrix (&S, false, false, NS, NS, 2*NS, 1, false)) ;

        for (int k = 0 ; k < NOPS ; k++)
        {
            new_matrix (&A, by_col, hyper, h, g) ;
            GrB_Matrix R = A ;      // the matrix to check
            switch (k)
            {
                case 0:     // build
                    OK (GrB_Matrix_build_FP64 (A, Ilist, Jlist, Xlist, 3,
                        NULL)) ;
                    break ;
                case 1:     // setElement into an empty matrix
                    OK (GrB_Matrix_setElement_FP64 (A, 1, 3, 4)) ;
                    break ;
                case 2:     // setElement into a non-empty matrix
                    fill (A) ;
                    OK (GrB_Matrix_setElement_FP64 (A, 1, 3, 4)) ;
                    break ;
                case 3:     // removeElement
                    fill (A) ;
                    OK (GrB_Matrix_removeElement (A, 4, 4)) ;
                    break ;
                case 4:     // clear
                    fill (A) ;
                    OK (GrB_Matrix_clear (A)) ;
                    break ;
                case 5:     // clear, then setElement
                    fill (A) ;
                    OK (GrB_Matrix_clear (A)) ;
                    OK (GrB_Matrix_setElement_FP64 (A, 1, 3, 4)) ;
                    break ;
                case 6:     // A = B*B
                    OK (GrB_mxm (A, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64,
                        B, B, NULL)) ;
                    break ;
                case 7:     // A += B*B
                    fill (A) ;
                    OK (GrB_mxm (A, NULL, GrB_PLUS_FP64,
                        GrB_PLUS_TIMES_SEMIRING_FP64, B, B, NULL)) ;
                    break ;
                case 8:     // A = B+B
                    OK (GrB_Matrix_eWiseAdd_BinaryOp (A, NULL, NULL,
                        GrB_PLUS_FP64, B, B, NULL)) ;
                    break ;
                case 9:     // A = B.*B
                    OK (GrB_Matrix_eWiseMult_BinaryOp (A, NULL, NULL,
                        GrB_TIMES_FP64, B, B, NULL)) ;
                    break ;
                case 10:    // A = tril (B)
                    OK (GrB_Matrix_select_INT64 (A, NULL, NULL, GrB_TRIL, B,
                        0, NULL)) ;
                    break ;
                case 11:    // A = B'
                    OK (GrB_transpose (A, NULL, NULL, B, NULL)) ;
                    break ;
                case 12:    // A = -B
                    OK (GrB_Matrix_apply (A, NULL, NULL, GrB_AINV_FP64, B,
                        NULL)) ;
                    break ;
                case 13:    // A = B (:,:)
                    OK (GrB_Matrix_extract (A, NULL, NULL, B, GrB_ALL, 0,
                        GrB_ALL, 0, NULL)) ;
                    break ;
                case 14:    // A = kron (S,S)
                    OK (GrB_Matrix_kronecker_BinaryOp (A, NULL, NULL,
                        GrB_TIMES_FP64, S, S, NULL)) ;
                    break ;
                case 15:    // A<B,struct> = B
                    OK (GrB_Matrix_assign (A, B, NULL, B, GrB_ALL, 0,
                        GrB_ALL, 0, GrB_DESC_S)) ;
                    break ;
                case 16:    // A = B
                    OK (GrB_Matrix_assign (A, NULL, NULL, B, GrB_ALL, 0,
                        GrB_ALL, 0, NULL)) ;
                    break ;
                case 17:    // A = B, with replace
                    OK (GrB_Matrix_assign (A, NULL, NULL, B, GrB_ALL, 0,
                        GrB_ALL, 0, GrB_DESC_R)) ;
                    break ;
                case 18:    // A(I,J) = scalar
                    fill (A) ;
                    OK (GrB_Matrix_assign_FP64 (A, NULL, NULL, 3, Ilist, 3,
                        Jlist, 3, NULL)) ;
                    break ;
                case 19:    // resize A
                    fill (A) ;
                    OK (GrB_Matrix_resize (A, 2*N, N/2)) ;
                    break ;
                case 20:    // reshape A
                    fill (A) ;
                    OK (GxB_Matrix_reshape (A, true, N/2, 2*N, NULL)) ;
                    break ;
                case 21:    // A = sort (B)
                    OK (GxB_Matrix_sort (A, NULL, GrB_LT_FP64, B, NULL)) ;
                    break ;
                case 22:    // A = concat (split (B))
                    OK (GxB_Matrix_split (Tiles, 2, 2, Tile_size, Tile_size,
                        B, NULL)) ;
                    OK (GxB_Matrix_concat (A, Tiles, 2, 2, NULL)) ;
                    for (int t = 0 ; t < 4 ; t++)
                    {
                        OK (GrB_Matrix_free (&(Tiles [t]))) ;
                    }
                    break ;
                case 23:    // A = diag (v)
                    OK (GrB_Vector_new (&v, GrB_FP64, N)) ;
                    OK (GrB_Vector_setElement_FP64 (v, 1, 3)) ;
                    OK (GxB_Matrix_diag (A, v, 0, NULL)) ;
                    OK (GrB_Vector_free (&v)) ;
                    break ;
                case 24:    // build an iso matrix
                    OK (GxB_Matrix_build_Scalar (A, Ilist, Jlist, s, 3)) ;
                    break ;
                case 25:    // convert A to bitmap and back
                    fill (A) ;
                    OK (GrB_Matrix_set_INT32 (A, GxB_BITMAP,
                        GxB_SPARSITY_CONTROL)) ;
                    OK (GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
                    OK (GrB_Matrix_set_INT32 (A,
                        hyper ? GxB_HYPERSPARSE : GxB_SPARSE,
                        GxB_SPARSITY_CONTROL)) ;
                    break ;
                case 26:    // change the format of A
                    fill (A) ;
                    OK (GrB_Matrix_set_INT32 (A,
                        by_col ? GxB_BY_ROW : GxB_BY_COL, GxB_FORMAT)) ;
                    break ;
                case 27:    // unload A into a Container and load it back
                    fill (A) ;
                    OK (GxB_Container_new (&Container)) ;
                    OK (GxB_unload_Matrix_into_Container (A, Container,
                        NULL)) ;
                    OK (GxB_load_Matrix_from_Container (A, Container,
                        NULL)) ;
                    OK (GxB_Container_free (&Container)) ;
                    break ;
                case 28:    // unpack A and pack it back
                {
                    uint64_t *Ap = NULL, *Ai = NULL ;
                    void *Ax = NULL ;
                    uint64_t Ap_size, Ai_size, Ax_size ;
                    bool iso, jumbled ;
                    fill (A) ;
                    OK (GrB_Matrix_set_INT32 (A, GxB_SPARSE,
                        GxB_SPARSITY_CONTROL)) ;
                    if (by_col)
                    {
                        OK (GxB_Matrix_unpack_CSC (A, &Ap, &Ai, &Ax,
                            &Ap_size, &Ai_size, &Ax_size, &iso, &jumbled,
                            NULL)) ;
                        OK (GxB_Matrix_pack_CSC (A, &Ap, &Ai, &Ax,
                            Ap_size, Ai_size, Ax_size, iso, jumbled, NULL)) ;
                    }
                    else
                    {
                        OK (GxB_Matrix_unpack_CSR (A, &Ap, &Ai, &Ax,
                            &Ap_size, &Ai_size, &Ax_size, &iso, &jumbled,
                            NULL)) ;
                        OK (GxB_Matrix_pack_CSR (A, &Ap, &Ai, &Ax,
                            Ap_size, Ai_size, Ax_size, iso, jumbled, NULL)) ;
                    }
                }
                break ;
                case 29:    // C = A
                    fill (A) ;
                    OK (GrB_Matrix_dup (&C, A)) ;
                    R = C ;
                    break ;
                case 30:    // C = deserialize (serialize (A))
                    fill (A) ;
                    OK (GxB_Matrix_serialize (&blob, &blob_size, A, NULL)) ;
                    OK (GxB_Matrix_deserialize (&C, NULL, blob, blob_size,
                        NULL)) ;
                    mxFree (blob) ;
                    blob = NULL ;
                    R = C ;
                    break ;
                default: ;
            }
            // the integer sizes are not checked for GrB_Matrix_clear,
            // GxB_load_Matrix_from_Container, and GxB_Matrix_pack_*
            bool check_bits = !(k == 4 || k == 27 || k == 28) ;
            npass += check_hints (R, h, g, check_bits, op_name [k]) ;
            OK (GrB_Matrix_free (&A)) ;
            OK (GrB_Matrix_free (&C)) ;
            ntests++ ;
        }

        OK (GrB_Matrix_free (&B)) ;
        OK (GrB_Matrix_free (&S)) ;
    }

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    // restore the global hints
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, global_r,
        GxB_ROWINDEX_INTEGER_HINT)) ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, global_c,
        GxB_COLINDEX_INTEGER_HINT)) ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, global_p,
        GxB_OFFSET_INTEGER_HINT)) ;

    FREE_ALL ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test56:  %d / %d tests passed\n\n", npass, ntests) ;
    // CHECK (npass == ntests) ;
    if (npass != ntests) printf ("\n################# GB_mex_test56 FAILED\n") ;
}
