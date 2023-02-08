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

void printgauss (GrB_Matrix A)
{
    // print the matrix
    GrB_Index m, n ;
    GrB_Matrix_nrows (&m, A) ;
    GrB_Matrix_ncols (&n, A) ;
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
            else
            {
                printf (" (%4d,%4d)", a.real, a.imag) ;
            }
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
    GrB_init (GrB_NONBLOCKING) ;
    GxB_set (GxB_BURBLE, true) ;

    // create the Gauss type
    GrB_Type Gauss ;
    GxB_Type_new (&Gauss, sizeof (gauss), "gauss", GAUSS_DEFN) ;
    GxB_print (Gauss, 3) ;

    // create the AddGauss operators
    GrB_BinaryOp AddGauss ; 
    GxB_BinaryOp_new (&AddGauss, (void *) addgauss, Gauss, Gauss, Gauss,
        "addgauss", ADDGAUSS_DEFN) ;
    GxB_print (AddGauss, 3) ;

    // create the AddMonoid
    gauss zero ;
    zero.real = 0 ;
    zero.imag = 0 ;
    GrB_Monoid AddMonoid ;
    GrB_Monoid_new_UDT (&AddMonoid, AddGauss, &zero) ;
    GxB_print (AddMonoid, 3) ;

    // create the MultGauss operator
    GrB_BinaryOp MultGauss ;
    GxB_BinaryOp_new (&MultGauss, (void *) multgauss, Gauss, Gauss, Gauss,
        "multgauss", MULTGAUSS_DEFN) ;
    GxB_print (MultGauss, 3) ;

    // create the GaussSemiring
    GrB_Semiring GaussSemiring ;
    GrB_Semiring_new (&GaussSemiring, AddMonoid, MultGauss) ;
    GxB_print (GaussSemiring, 3) ;

    // create a 4-by-4 Gauss matrix, each entry A(i,j) = (i+1,2-j),
    // except A(0,0) is missing
    GrB_Matrix A ;
    GrB_Matrix_new (&A, Gauss, 4, 4) ;
    gauss a ;
    for (int i = 0 ; i < 4 ; i++)
    {
        for (int j = 0 ; j < 4 ; j++)
        {
            if (i == 0 && j == 0) continue ;
            a.real = i+1 ;
            a.imag = 2-j ;
            GrB_Matrix_setElement_UDT (A, &a, i, j) ;
        }
    }
    printf ("\n=============== A matrix:\n") ;
    printgauss (A) ;

    // a = sum (A)
    GrB_Matrix_reduce_UDT (&a, NULL, AddMonoid, A, NULL) ;
    printf ("\nsum (A) = (%d,%d)\n", a.real, a.imag) ;

    // A = A*A
    GrB_mxm (A, NULL, NULL, GaussSemiring, A, A, NULL) ;
    printf ("\n=============== A^2 matrix:\n") ;
    printgauss (A) ;

    // a = sum (A)
    GrB_Matrix_reduce_UDT (&a, NULL, AddMonoid, A, NULL) ;
    printf ("\nsum (A^2) = (%d,%d)\n", a.real, a.imag) ;

    // free everything and finalize GraphBLAS
    GrB_free (&A) ;
    GrB_free (&Gauss) ;
    GrB_free (&AddGauss) ;
    GrB_free (&AddMonoid) ;
    GrB_free (&MultGauss) ;
    GrB_free (&GaussSemiring) ;
    GrB_finalize ( ) ;
}

