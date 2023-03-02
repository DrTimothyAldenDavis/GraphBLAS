//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/gauss_demo: Gaussian integers
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GraphBLAS.h"

//------------------------------------------------------------------------------
// the Gaussian integer: real and imaginary parts
//------------------------------------------------------------------------------

typedef struct
{
    int32_t real ;
    int32_t imag ;
}
gauss ;

// repeat the typedef as a string, to give to GraphBLAS
#define GAUSS_DEFN              \
"typedef struct "               \
"{ "                            \
   "int32_t real ; "            \
   "int32_t imag ; "            \
"} "                            \
"gauss ;"

//------------------------------------------------------------------------------
// addgauss: add two Gaussian integers
//------------------------------------------------------------------------------

void addgauss (gauss *z, const gauss *x, const gauss *y)
{
    z->real = x->real + y->real ;
    z->imag = x->imag + y->imag ;
}

#define ADDGAUSS_DEFN                                           \
"void addgauss (gauss *z, const gauss *x, const gauss *y)   \n" \
"{                                                          \n" \
"    z->real = x->real + y->real ;                          \n" \
"    z->imag = x->imag + y->imag ;                          \n" \
"}"

//------------------------------------------------------------------------------
// multgauss: multiply two Gaussian integers
//------------------------------------------------------------------------------

void multgauss (gauss *z, const gauss *x, const gauss *y)
{
    z->real = x->real * y->real - x->imag * y->imag ;
    z->imag = x->real * y->imag + x->imag * y->real ;
}

#define MULTGAUSS_DEFN                                          \
"void multgauss (gauss *z, const gauss *x, const gauss *y)  \n" \
"{                                                          \n" \
"    z->real = x->real * y->real - x->imag * y->imag ;      \n" \
"    z->imag = x->real * y->imag + x->imag * y->real ;      \n" \
"}"

//------------------------------------------------------------------------------
// printgauss: print a Gauss matrix
//------------------------------------------------------------------------------

// This is a very slow way to print a large matrix, so using this approach is
// not recommended for large matrices.  However, it looks nice for this demo
// since the matrix is small.

#define TRY(method)             \
{                               \
    GrB_Info info = method ;    \
    if (info != GrB_SUCCESS)    \
    {                           \
        printf ("info: %d error! Line %d\n", info, __LINE__)  ; \
        fflush (stdout) ;       \
        abort ( ) ;             \
    }                           \
}

void printgauss (GrB_Matrix A)
{
    // print the matrix
    GrB_Index m, n ;
    TRY (GrB_Matrix_nrows (&m, A)) ;
    TRY (GrB_Matrix_ncols (&n, A)) ;
    printf ("size: %d-by-%d\n", (int) m, (int) n) ;
    for (int i = 0 ; i < m ; i++)
    {
        printf ("row %2d: ", i) ;
        for (int j = 0 ; j < n ; j++)
        {
            gauss a ;
            GrB_Info info = GrB_Matrix_extractElement_UDT (&a, A, i, j) ;
            if (info == GrB_NO_VALUE)
            {
                printf ("      .     ") ;
            }
            else if (info == GrB_SUCCESS)
            {
                printf (" (%4d,%4d)", a.real, a.imag) ;
            }
            else TRY (GrB_PANIC) ;
        }
        printf ("\n") ;
    }
    printf ("\n") ;
}

//------------------------------------------------------------------------------
// gauss main program
//------------------------------------------------------------------------------

int main (void)
{

    // start GraphBLAS
    TRY (GrB_init (GrB_NONBLOCKING)) ;
    TRY (GxB_set (GxB_BURBLE, true)) ;

    // create the Gauss type
    GrB_Type Gauss ;
    TRY (GxB_Type_new (&Gauss, sizeof (gauss), "gauss", GAUSS_DEFN)) ;
    TRY (GxB_print (Gauss, 3)) ;

    // create the AddGauss operators
    GrB_BinaryOp AddGauss ; 
    TRY (GxB_BinaryOp_new (&AddGauss, (void *) addgauss, Gauss, Gauss, Gauss,
        "addgauss", ADDGAUSS_DEFN)) ;
    TRY (GxB_print (AddGauss, 3)) ;

    // create the AddMonoid
    gauss zero ;
    zero.real = 0 ;
    zero.imag = 0 ;
    GrB_Monoid AddMonoid ;
    TRY (GrB_Monoid_new_UDT (&AddMonoid, AddGauss, &zero)) ;
    TRY (GxB_print (AddMonoid, 3)) ;

    // create the MultGauss operator
    GrB_BinaryOp MultGauss ;
    TRY (GxB_BinaryOp_new (&MultGauss, (void *) multgauss,
        Gauss, Gauss, Gauss, "multgauss", MULTGAUSS_DEFN)) ;
    TRY (GxB_print (MultGauss, 3)) ;

    // create the GaussSemiring
    GrB_Semiring GaussSemiring ;
    TRY (GrB_Semiring_new (&GaussSemiring, AddMonoid, MultGauss)) ;
    TRY (GxB_print (GaussSemiring, 3)) ;

    // create a 4-by-4 Gauss matrix, each entry A(i,j) = (i+1,2-j),
    // except A(0,0) is missing
    GrB_Matrix A, D ;
    TRY (GrB_Matrix_new (&A, Gauss, 4, 4)) ;
    TRY (GrB_Matrix_new (&D, GrB_BOOL, 4, 4)) ;
    gauss a ;
    for (int i = 0 ; i < 4 ; i++)
    {
        TRY (GrB_Matrix_setElement (D, 1, i, i)) ;
        for (int j = 0 ; j < 4 ; j++)
        {
            if (i == 0 && j == 0) continue ;
            a.real = i+1 ;
            a.imag = 2-j ;
            TRY (GrB_Matrix_setElement_UDT (A, &a, i, j)) ;
        }
    }
    printf ("\n=============== A matrix:\n") ;
    printgauss (A) ;

    // a = sum (A)
    TRY (GrB_Matrix_reduce_UDT (&a, NULL, AddMonoid, A, NULL)) ;
    printf ("\nsum (A) = (%d,%d)\n", a.real, a.imag) ;

    // A = A*A
    TRY (GrB_mxm (A, NULL, NULL, GaussSemiring, A, A, NULL)) ;
    printf ("\n=============== A^2 matrix:\n") ;
    printgauss (A) ;

    // a = sum (A)
    TRY (GrB_Matrix_reduce_UDT (&a, NULL, AddMonoid, A, NULL)) ;
    printf ("\nsum (A^2) = (%d,%d)\n", a.real, a.imag) ;

    // C<D> = A*A' where A and D are sparse
    GrB_Matrix C ;
    TRY (GrB_Matrix_new (&C, Gauss, 4, 4)) ;
    printgauss (C) ;
    TRY (GxB_set (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    TRY (GxB_set (D, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    TRY (GrB_mxm (C, D, NULL, GaussSemiring, A, A, GrB_DESC_T1)) ;
    printf ("\n=============== diag(AA') matrix:\n") ;
    printgauss (C) ;

    // C = D*A
    GrB_free (&D) ;
    TRY (GrB_Matrix_new (&D, Gauss, 4, 4)) ;
    TRY (GxB_set (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    TRY (GxB_set (D, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    TRY (GrB_select (D, NULL, NULL, GrB_DIAG, A, 0, NULL)) ;
    printgauss (D) ;
    TRY (GrB_mxm (C, NULL, NULL, GaussSemiring, D, A, NULL)) ;
    printgauss (C) ;

    // C = A*D
    TRY (GrB_mxm (C, NULL, NULL, GaussSemiring, A, D, NULL)) ;
    printgauss (C) ;

    // C = (1,2) then C += A*A' where C is full
    gauss ciso ;
    ciso.real = 1 ;
    ciso.imag = -2 ;
    TRY (GrB_Matrix_assign_UDT (C, NULL, NULL, &ciso, GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C) ;
    printgauss (A) ;
    TRY (GrB_mxm (C, NULL, AddGauss, GaussSemiring, A, A, GrB_DESC_T1)) ;
    printgauss (C) ;

    // C += B*A where B is full and A is sparse
    GrB_Matrix B ;
    TRY (GrB_Matrix_new (&B, Gauss, 4, 4)) ;
    TRY (GrB_Matrix_assign_UDT (B, NULL, NULL, &ciso, GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    TRY (GrB_mxm (C, NULL, AddGauss, GaussSemiring, B, A, NULL)) ;
    printgauss (C) ;

    // C += A*B where B is full and A is sparse
    TRY (GrB_mxm (C, NULL, AddGauss, GaussSemiring, A, B, NULL)) ;
    printgauss (C) ;

    // C = ciso+A
    TRY (GrB_apply (C, NULL, NULL, AddGauss, (void *) &ciso, A, NULL)) ;
    printgauss (C) ;

    // C = A*ciso
    TRY (GrB_apply (C, NULL, NULL, MultGauss, A, (void *) &ciso, NULL)) ;
    printgauss (C) ;

    // free everything and finalize GraphBLAS
    GrB_free (&A) ;
    GrB_free (&B) ;
    GrB_free (&D) ;
    GrB_free (&C) ;
    GrB_free (&Gauss) ;
    GrB_free (&AddGauss) ;
    GrB_free (&AddMonoid) ;
    GrB_free (&MultGauss) ;
    GrB_free (&GaussSemiring) ;
    GrB_finalize ( ) ;
}

