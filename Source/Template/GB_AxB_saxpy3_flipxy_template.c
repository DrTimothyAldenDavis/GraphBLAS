//------------------------------------------------------------------------------
// GB_AxB_saxpy3_flipxy_template: C=A*B for GB_AxB_saxpy3_generic
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{
    if (flipxy)
    { 
        // t = B(k,j) * A(i,k)
        #undef  GB_MULT
        #define GB_MULT(t, aik, bkj, i, k, j) fmult (GB_ADDR (t), bkj, aik)
        #include "GB_AxB_saxpy3_template.c"
    }
    else
    { 
        // t = A(i,k) * B(k,j)
        #undef  GB_MULT
        #define GB_MULT(t, aik, bkj, i, k, j) fmult (GB_ADDR (t), aik, bkj)
        #include "GB_AxB_saxpy3_template.c"
    }
}
break ;

#undef GB_HASH_FINEGUS
#undef GB_HASH_TYPE
#undef GB_HASH_COARSE
