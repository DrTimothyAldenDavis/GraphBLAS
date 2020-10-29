//SPDX-License-Identifier: Apache-2.0

#define chunksize 128 

#define ASSERT
#define GB_RESTRICT __restrict__
//#define GB_GETA( aval, ax, p) aval = (T_Z)ax[ ( p )]
//#define GB_GETB( bval, bx, p) bval = (T_Z)bx[ ( p )]
#define GB_ADD_F( f , s)  f = GB_ADD ( f, s ) 
#define GB_C_MULT( c, a, b)  c = GB_MULT( (a), (b) )
#define GB_MULTADD( c, a ,b ) GB_ADD_F( (c), GB_MULT( (a),(b) ) )
#define GB_DOT_TERMINAL ( c )   
//# if ( c == TERMINAL_VALUE) break;

#define GB_IMIN( A, B) ( (A) < (B) ) ?  (A) : (B)
#define GB_IMAX( A, B) ( (A) > (B) ) ?  (A) : (B)

#define GB_FLIP(i)             (-(i)-2)
#define GB_IS_FLIPPED(i)       ((i) < 0)
#define GB_IS_ZOMBIE(i)        ((i) < 0)
#define GB_IS_NOT_FLIPPED(i)   ((i) >= 0)
#define GB_IS_NOT_ZOMBIE(i)    ((i) >= 0)
#define GB_UNFLIP(i)           (((i) < 0) ? GB_FLIP(i) : (i))

#define GB_NNZ(A) (((A)->nzmax > 0) ? ((A)->p [(A)->nvec] - (A)->p [0]) : 0 )

// GB_PART and GB_PARTITION:  divide the index range 0:n-1 uniformly
// for nthreads.  GB_PART(tid,n,nthreads) is the first index for thread tid.
#define GB_PART(tid,n,nthreads)  \
    GB_IMIN( ((tid) * ((double)(n))/((double)(nthreads))), n)

// thread tid will operate on the range k1:(k2-1)
#define GB_PARTITION(k1,k2,n,tid,nthreads)                                  \
    k1 = ((tid) ==  0          ) ?  0  : GB_PART ((tid),  n, nthreads) ;    \
    k2 = ((tid) == (nthreads)-1) ? (n) : GB_PART ((tid)+1,n, nthreads) ;

//------------------------------------------------------------------------------
// GB_BINARY_SEARCH
//------------------------------------------------------------------------------

// search for integer i in the list X [pleft...pright]; no zombies.
// The list X [pleft ... pright] is in ascending order.  It may have
// duplicates.

#define GB_TRIM_BINARY_SEARCH(i,X,pleft,pright)                             \
{                                                                           \
    /* binary search of X [pleft ... pright] for integer i */               \
    while (pleft < pright)                                                  \
    {                                                                       \
        int64_t pmiddle = (pleft + pright) / 2 ;                            \
        if (X [pmiddle] < i)                                                \
        {                                                                   \
            /* if in the list, it appears in [pmiddle+1..pright] */         \
            pleft = pmiddle + 1 ;                                           \
        }                                                                   \
        else                                                                \
        {                                                                   \
            /* if in the list, it appears in [pleft..pmiddle] */            \
            pright = pmiddle ;                                              \
        }                                                                   \
    }                                                                       \
    /* binary search is narrowed down to a single item */                   \
    /* or it has found the list is empty */                                 \
    /*ASSERT (pleft == pright || pleft == pright + 1) ;*/                   \
}

//------------------------------------------------------------------------------
// GB_SPLIT_BINARY_SEARCH: binary search, and then partition the list
//------------------------------------------------------------------------------

// If found is true then X [pleft] == i.  If duplicates appear then X [pleft]
//    is any one of the entries with value i in the list.
// If found is false then
//    X [original_pleft ... pleft-1] < i and
//    X [pleft ... original_pright] > i holds, and pleft-1 == pright
// If X has no duplicates, then whether or not i is found,
//    X [original_pleft ... pleft-1] < i and
//    X [pleft ... original_pright] >= i holds.
#define GB_SPLIT_BINARY_SEARCH(i,X,pleft,pright,found)                      \
{                                                                           \
    GB_BINARY_SEARCH (i, X, pleft, pright, found)                           \
    if (!found && (pleft == pright))                                        \
    {                                                                       \
        if (i > X [pleft])                                                  \
        {                                                                   \
            pleft++ ;                                                       \
        }                                                                   \
        else                                                                \
        {                                                                   \
            pright++ ;                                                      \
        }                                                                   \
    }                                                                       \
}


//------------------------------------------------------------------------------
// GB_BINARY_SEARCH: binary search and check if found
//------------------------------------------------------------------------------

// If found is true then X [pleft == pright] == i.  If duplicates appear then
// X [pleft] is any one of the entries with value i in the list.
// If found is false then
//    X [original_pleft ... pleft-1] < i and
//    X [pleft+1 ... original_pright] > i holds.
// The value X [pleft] may be either < or > i.
#define GB_BINARY_SEARCH(i,X,pleft,pright,found)                            \
{                                                                           \
    GB_TRIM_BINARY_SEARCH (i, X, pleft, pright) ;                           \
    found = (pleft == pright && X [pleft] == i) ;                           \
}

#include "../Source/Template/GB_lookup_template.c"
#include "../Source/Template/GB_search_for_vector_template.c"

#undef GB_DOT_MERGE
// cij += A(k,i) * B(k,j), for merge operation
#define GB_DOT_MERGE                                                \
{                                                                   \
    GB_GETA ( aki= (T_Z)Ax[pA]) ;       /* aki = A(k,i) */          \
    GB_GETB ( bkj= (T_Z)Bx[pB]) ;       /* bkj = B(k,j) */          \
    if (cij_exists)                                                 \
    {                                                               \
        GB_MULTADD (cij, aki, bkj) ;    /* cij += aki * bkj */      \
    }                                                               \
    else                                                            \
    {                                                               \
        /* cij = A(k,i) * B(k,j), and add to the pattern    */      \
        cij_exists = true ;                                         \
        GB_C_MULT (cij, aki, bkj) ;     /* cij  = aki * bkj */      \
    }                                                               \
}


typedef void (*GxB_binary_function) (void *, const void *, const void *) ;

typedef struct GB_Pending_struct *GB_Pending ;

//------------------------------------------------------------------------------
// type codes for GrB_Type
//------------------------------------------------------------------------------

typedef enum
{
    // the 14 scalar types: 13 built-in types, and one user-defined type code
    GB_ignore_code  = 0,
    GB_BOOL_code    = 0,        // 'logical' in MATLAB
    GB_INT8_code    = 1,
    GB_UINT8_code   = 2,
    GB_INT16_code   = 3,
    GB_UINT16_code  = 4,
    GB_INT32_code   = 5,
    GB_UINT32_code  = 6,
    GB_INT64_code   = 7,
    GB_UINT64_code  = 8,
    GB_FP32_code    = 9,        // float ('single' in MATLAB)
    GB_FP64_code    = 10,       // double
    GB_FC32_code    = 11,       // float complex ('single complex' in MATLAB)
    GB_FC64_code    = 12,       // double complex
    GB_UDT_code     = 13        // void *, user-defined type
}
GB_Type_code ;                  // enumerated type code

typedef enum
{
    //--------------------------------------------------------------------------
    // NOP
    //--------------------------------------------------------------------------

    GB_NOP_opcode = 0,  // no operation

    //--------------------------------------------------------------------------
    // primary unary operators x=f(x)
    //--------------------------------------------------------------------------

    GB_ONE_opcode,      // z = 1
    GB_IDENTITY_opcode, // z = x
    GB_AINV_opcode,     // z = -x
    GB_ABS_opcode,      // z = abs(x) ; except z is real if x is complex
    GB_MINV_opcode,     // z = 1/x ; special cases for bool and integers
    GB_LNOT_opcode,     // z = !x
    GB_BNOT_opcode,     // z = ~x (bitwise complement)

    //--------------------------------------------------------------------------
    // unary operators for floating-point types (real and complex)
    //--------------------------------------------------------------------------

    GB_SQRT_opcode,     // z = sqrt (x)
    GB_LOG_opcode,      // z = log (x)
    GB_EXP_opcode,      // z = exp (x)

    GB_SIN_opcode,      // z = sin (x)
    GB_COS_opcode,      // z = cos (x)
    GB_TAN_opcode,      // z = tan (x)

    GB_ASIN_opcode,     // z = asin (x)
    GB_ACOS_opcode,     // z = acos (x)
    GB_ATAN_opcode,     // z = atan (x)

    GB_SINH_opcode,     // z = sinh (x)
    GB_COSH_opcode,     // z = cosh (x)
    GB_TANH_opcode,     // z = tanh (x)

    GB_ASINH_opcode,    // z = asinh (x)
    GB_ACOSH_opcode,    // z = acosh (x)
    GB_ATANH_opcode,    // z = atanh (x)

    GB_CEIL_opcode,     // z = ceil (x)
    GB_FLOOR_opcode,    // z = floor (x)
    GB_ROUND_opcode,    // z = round (x)
    GB_TRUNC_opcode,    // z = trunc (x)

    GB_EXP2_opcode,     // z = exp2 (x)
    GB_EXPM1_opcode,    // z = expm1 (x)
    GB_LOG10_opcode,    // z = log10 (x)
    GB_LOG1P_opcode,    // z = log1P (x)
    GB_LOG2_opcode,     // z = log2 (x)

    //--------------------------------------------------------------------------
    // unary operators for real floating-point types
    //--------------------------------------------------------------------------

    GB_LGAMMA_opcode,   // z = lgamma (x)
    GB_TGAMMA_opcode,   // z = tgamma (x)
    GB_ERF_opcode,      // z = erf (x)
    GB_ERFC_opcode,     // z = erfc (x)
    GB_FREXPX_opcode,   // z = frexpx (x), mantissa from ANSI C11 frexp
    GB_FREXPE_opcode,   // z = frexpe (x), exponent from ANSI C11 frexp

    //--------------------------------------------------------------------------
    // unary operators for complex types only
    //--------------------------------------------------------------------------

    GB_CONJ_opcode,     // z = conj (x)

    //--------------------------------------------------------------------------
    // unary operators where z is real and x is complex
    //--------------------------------------------------------------------------

    GB_CREAL_opcode,    // z = creal (x)
    GB_CIMAG_opcode,    // z = cimag (x)
    GB_CARG_opcode,     // z = carg (x)

    //--------------------------------------------------------------------------
    // unary operators where z is bool and x is any floating-point type
    //--------------------------------------------------------------------------

    GB_ISINF_opcode,    // z = isinf (x)
    GB_ISNAN_opcode,    // z = isnan (x)
    GB_ISFINITE_opcode, // z = isfinite (x)

    //--------------------------------------------------------------------------
    // binary operators z=f(x,y) that return the same type as their inputs
    //--------------------------------------------------------------------------

    GB_FIRST_opcode,    // z = x
    GB_SECOND_opcode,   // z = y
    GB_ANY_opcode,      // z = x or y, selected arbitrarily
    GB_PAIR_opcode,     // z = 1
    GB_MIN_opcode,      // z = min(x,y)
    GB_MAX_opcode,      // z = max(x,y)
    GB_PLUS_opcode,     // z = x + y
    GB_MINUS_opcode,    // z = x - y
    GB_RMINUS_opcode,   // z = y - x
    GB_TIMES_opcode,    // z = x * y
    GB_DIV_opcode,      // z = x / y ; special cases for bool and ints
    GB_RDIV_opcode,     // z = y / x ; special cases for bool and ints
    GB_POW_opcode,      // z = pow (x,y)

    GB_ISEQ_opcode,     // z = (x == y)
    GB_ISNE_opcode,     // z = (x != y)
    GB_ISGT_opcode,     // z = (x >  y)
    GB_ISLT_opcode,     // z = (x <  y)
    GB_ISGE_opcode,     // z = (x >= y)
    GB_ISLE_opcode,     // z = (x <= y)

    GB_LOR_opcode,      // z = (x != 0) || (y != 0)
    GB_LAND_opcode,     // z = (x != 0) && (y != 0)
    GB_LXOR_opcode,     // z = (x != 0) != (y != 0)

    GB_BOR_opcode,      // z = (x | y), bitwise or
    GB_BAND_opcode,     // z = (x & y), bitwise and
    GB_BXOR_opcode,     // z = (x ^ y), bitwise xor
    GB_BXNOR_opcode,    // z = ~(x ^ y), bitwise xnor
    GB_BGET_opcode,     // z = bitget (x,y)
    GB_BSET_opcode,     // z = bitset (x,y)
    GB_BCLR_opcode,     // z = bitclr (x,y)
    GB_BSHIFT_opcode,   // z = bitshift (x,y)

    //--------------------------------------------------------------------------
    // binary operators z=f(x,y) that return bool (TxT -> bool)
    //--------------------------------------------------------------------------

    GB_EQ_opcode,       // z = (x == y)
    GB_NE_opcode,       // z = (x != y)
    GB_GT_opcode,       // z = (x >  y)
    GB_LT_opcode,       // z = (x <  y)
    GB_GE_opcode,       // z = (x >= y)
    GB_LE_opcode,       // z = (x <= y)

    //--------------------------------------------------------------------------
    // binary operators for real floating-point types (TxT -> T)
    //--------------------------------------------------------------------------

    GB_ATAN2_opcode,        // z = atan2 (x,y)
    GB_HYPOT_opcode,        // z = hypot (x,y)
    GB_FMOD_opcode,         // z = fmod (x,y)
    GB_REMAINDER_opcode,    // z = remainder (x,y)
    GB_COPYSIGN_opcode,     // z = copysign (x,y)
    GB_LDEXP_opcode,        // z = ldexp (x,y)

    //--------------------------------------------------------------------------
    // binary operator z=f(x,y) where z is complex, x,y real:
    //--------------------------------------------------------------------------

    GB_CMPLX_opcode,        // z = cmplx (x,y)

    //--------------------------------------------------------------------------
    // user-defined: unary and binary operators
    //--------------------------------------------------------------------------

    GB_USER_opcode          // user-defined operator
}
GB_Opcode ;


#define GB_LEN 128

struct GB_Type_opaque       // content of GrB_Type
{
    int64_t magic ;         // for detecting uninitialized objects
    size_t size ;           // size of the type
    GB_Type_code code ;     // the type code
    char name [GB_LEN] ;    // name of the type
} ;

typedef struct GB_Type_opaque *GrB_Type;

struct GB_BinaryOp_opaque   // content of GrB_BinaryOp
{
    int64_t magic ;         // for detecting uninitialized objects
    GrB_Type xtype ;        // type of x
    GrB_Type ytype ;        // type of y
    GrB_Type ztype ;        // type of z
    GxB_binary_function function ;        // a pointer to the binary function
    char name [GB_LEN] ;    // name of the binary operator
    GB_Opcode opcode ;      // operator opcode
} ;

typedef struct GB_BinaryOp_opaque *GrB_BinaryOp ;

//------------------------------------------------------------------------------
// GB_Pending data structure
//------------------------------------------------------------------------------
typedef unsigned char GB_void;

struct GB_Pending_struct    // list of pending tuples for a matrix
{
    int64_t n ;         // number of pending tuples to add to matrix
    int64_t nmax ;      // size of i,j,x
    bool sorted ;       // true if pending tuples are in sorted order
    int64_t *i ;        // row indices of pending tuples
    int64_t *j ;        // col indices of pending tuples; NULL if A->vdim <= 1
    GB_void *x ;        // values of pending tuples
    GrB_Type type ;     // the type of s
    size_t size ;       // type->size
    GrB_BinaryOp op ;   // operator to assemble pending tuples
} ;

typedef enum
{
    // for all GrB_Descriptor fields:
    GxB_DEFAULT = 0,    // default behavior of the method

    // for GrB_OUTP only:
    GrB_REPLACE = 1,    // clear the output before assigning new values to it

    // for GrB_MASK only:
    GrB_COMP = 2,       // use the structural complement of the input
    GrB_SCMP = 2,       // same as GrB_COMP (deprecated; use GrB_COMP instead)
    GrB_STRUCTURE = 4,  // use the only pattern of the mask, not its values

    // for GrB_INP0 and GrB_INP1 only:
    GrB_TRAN = 3,       // use the transpose of the input

    // for GxB_GPU_CONTROL only:
    GxB_GPU_ALWAYS  = 4,
    GxB_GPU_NEVER   = 5,

    // for GxB_AxB_METHOD only:
    GxB_AxB_GUSTAVSON = 1001,   // gather-scatter saxpy method
    GxB_AxB_HEAP      = 1002,   // heap-based saxpy method
    GxB_AxB_DOT       = 1003,   // dot product
    GxB_AxB_HASH      = 1004,   // hash-based saxpy method
    GxB_AxB_SAXPY     = 1005    // saxpy method (any kind)
}
GrB_Desc_Value ;

//Basic matrix container class
struct GB_Matrix_opaque     // content of GrB_Matrix
{
    #include "../Source/Template/GB_matrix.h"
} ;
typedef struct GB_Matrix_opaque *GrB_Matrix ;
/*
class Matrix {
  public:
     int64_t magic;
     size_t type_size;
     double hyper_ratio;
     int64_t plen;
     int64_t vlen;
     int64_t vdim;
     int64_t nvec;
     int64_t nvec_nonempty ;
     int64_t *h;
     int64_t *p;
     int64_t *i;
     void *x;                   // type determined at runtime
     int64_t nzmax;             // GB_NNZ (A), normally  A->p [A->nvec]
     uint64_t nzombies;      // A->nzombies
     int64_t hfirst ;
     bool is_filled = false;
     bool is_hyper = false;
};
typedef Matrix *GrB_Matrix; 
*/
