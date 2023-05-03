//==============================================================================
// GrB_get and GrB_set
//==============================================================================

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef enum
{
    GrB_OUTP = 0,   // descriptor for output of a method
    GrB_MASK = 1,   // descriptor for the mask input of a method
    GrB_INP0 = 2,   // descriptor for the first input of a method
    GrB_INP1 = 3,   // descriptor for the second input of a method
    GrB_NAME = 10,  // name of an object (string)
    GrB_LIBRARY_VER_MAJOR = 11, // libary major version (GrB_Scalar, int32)
    GrB_LIBRARY_VER_MINOR = 12, // libary minor version (GrB_Scalar, int32)
    GrB_LIBRARY_VER_PATCH = 13, // libary minor version (GrB_Scalar, int32)
    GrB_API_VER_MAJOR = 14,     // API major version (GrB_Scalar, int32)
    GrB_API_VER_MINOR = 15,     // API minor version (GrB_Scalar, int32)
    GrB_API_VER_PATCH = 16,     // API minor version (GrB_Scalar, int32)
    GrB_BLOCKING_MODE = 17,     // blocking mode (GrB_Mode)
    GrB_STORAGE_ORIENTATION_HINT = 100, // matrix orientation
    GrB_STORAGE_SPARSITY_HINT    = 101, // matrix format
    GrB_ELTYPE_CODE              = 102, // entry type (enum)
    GrB_INPUT1TYPE_CODE          = 103, // input1 type (enum)
    GrB_INPUT2TYPE_CODE          = 104, // input2 type (enum)
    GrB_OUTPUTTYPE_CODE          = 105, // output type (enum)
    GrB_ELTYPE_STRING            = 106, // entry type (string)
    GrB_INPUT1TYPE_STRING        = 107, // input1 type (string)
    GrB_INPUT2TYPE_STRING        = 108, // input2 type (string)
    GrB_OUTPUTTYPE_STRING        = 109, // output type (string)
}
GrB_Field ;

typedef enum
{

    GrB_SUCCESS = 0,            // all is well

    //--------------------------------------------------------------------------
    // informational codes, not an error:
    //--------------------------------------------------------------------------

    GrB_NO_VALUE = 1,           // A(i,j) requested but not there
    GxB_EXHAUSTED = 2,          // iterator is exhausted

    //--------------------------------------------------------------------------
    // errors:
    //--------------------------------------------------------------------------

    GrB_UNINITIALIZED_OBJECT = -1,  // object has not been initialized
    GrB_NULL_POINTER = -2,          // input pointer is NULL
    GrB_INVALID_VALUE = -3,         // generic error; some value is bad
    GrB_INVALID_INDEX = -4,         // row or column index is out of bounds
    GrB_DOMAIN_MISMATCH = -5,       // object domains are not compatible
    GrB_DIMENSION_MISMATCH = -6,    // matrix dimensions do not match
    GrB_OUTPUT_NOT_EMPTY = -7,      // output matrix already has values
    GrB_NOT_IMPLEMENTED = -8,       // method not implemented
    GrB_PANIC = -101,               // unknown error
    GrB_OUT_OF_MEMORY = -102,       // out of memory
    GrB_INSUFFICIENT_SPACE = -103,  // output array not large enough
    GrB_INVALID_OBJECT = -104,      // object is corrupted
    GrB_INDEX_OUT_OF_BOUNDS = -105, // row or col index out of bounds
    GrB_EMPTY_OBJECT = -106         // an object does not contain a value

}
GrB_Info ;

//==============================================================================

struct GB_Scalar_opaque     // a simple GrB_Scalar, just to be self-contained
{
    int stuff ;
} ;

struct GB_Vector_opaque     // a simple GrB_Vector, just to be self-contained
{
    int morestuff [8] ;
} ;

typedef struct GB_Scalar_opaque *GrB_Scalar ;
typedef struct GB_Vector_opaque *GrB_Vector ;

GrB_Info GrB_Scalar_set_Scalar (GrB_Scalar o, GrB_Field f, GrB_Scalar t) ;
GrB_Info GrB_Scalar_set_String (GrB_Scalar o, GrB_Field f, char *t) ;
GrB_Info GrB_Scalar_set_ENUM   (GrB_Scalar o, GrB_Field f, int t) ;
GrB_Info GrB_Scalar_set_VOID   (GrB_Scalar o, GrB_Field f, void *t /*, int n */) ;

GrB_Info GrB_Vector_set_Scalar (GrB_Vector o, GrB_Field f, GrB_Scalar t) ;
GrB_Info GrB_Vector_set_String (GrB_Vector o, GrB_Field f, char *t) ;
GrB_Info GrB_Vector_set_ENUM   (GrB_Vector o, GrB_Field f, int t) ;
GrB_Info GrB_Vector_set_VOID   (GrB_Vector o, GrB_Field f, void *t /*, int n */) ;

/*
GrB_Info GrB_Matrix_set_Scalar (GrB_Matrix o, GrB_Field f, GrB_Scalar t) ;
GrB_Info GrB_Matrix_set_String (GrB_Matrix o, GrB_Field f, char *t) ;
GrB_Info GrB_Matrix_set_ENUM   (GrB_Matrix o, GrB_Field f, int t) ;
GrB_Info GrB_Matrix_set_VOID   (GrB_Matrix o, GrB_Field f, void *t, int n) ;

GrB_Info GrB_UnaryOp_set_Scalar (GrB_UnaryOp o, GrB_Field f, GrB_Scalar t) ;
GrB_Info GrB_UnaryOp_set_String (GrB_UnaryOp o, GrB_Field f, char *t) ;
GrB_Info GrB_UnaryOp_set_ENUM   (GrB_UnaryOp o, GrB_Field f, int t) ;
GrB_Info GrB_UnaryOp_set_VOID   (GrB_UnaryOp o, GrB_Field f, void *t, int n) ;

GrB_Info GrB_IndexUnaryOp_set_Scalar (GrB_IndexUnaryOp o, GrB_Field f, GrB_Scalar t) ;
GrB_Info GrB_IndexUnaryOp_set_String (GrB_IndexUnaryOp o, GrB_Field f, char *t) ;
GrB_Info GrB_IndexUnaryOp_set_ENUM   (GrB_IndexUnaryOp o, GrB_Field f, int t) ;
GrB_Info GrB_IndexUnaryOp_set_VOID   (GrB_IndexUnaryOp o, GrB_Field f, void *t, int n) ;

GrB_Info GrB_BinaryOp_set_Scalar (GrB_BinaryOp o, GrB_Field f, GrB_Scalar t) ;
GrB_Info GrB_BinaryOp_set_String (GrB_BinaryOp o, GrB_Field f, char *t) ;
GrB_Info GrB_BinaryOp_set_ENUM   (GrB_BinaryOp o, GrB_Field f, int t) ;
GrB_Info GrB_BinaryOp_set_VOID   (GrB_BinaryOp o, GrB_Field f, void *t, int n) ;

GrB_Info GrB_Monoid_set_Scalar (GrB_Monoid o, GrB_Field f, GrB_Scalar t) ;
GrB_Info GrB_Monoid_set_String (GrB_Monoid o, GrB_Field f, char *t) ;
GrB_Info GrB_Monoid_set_ENUM   (GrB_Monoid o, GrB_Field f, int t) ;
GrB_Info GrB_Monoid_set_VOID   (GrB_Monoid o, GrB_Field f, void *t, int n) ;

GrB_Info GrB_Semiring_set_Scalar (GrB_Semiring o, GrB_Field f, GrB_Scalar t) ;
GrB_Info GrB_Semiring_set_String (GrB_Semiring o, GrB_Field f, char *t) ;
GrB_Info GrB_Semiring_set_ENUM   (GrB_Semiring o, GrB_Field f, int t) ;
GrB_Info GrB_Semiring_set_VOID   (GrB_Semiring o, GrB_Field f, void *t, int n) ;

GrB_Info GrB_Descriptor_set_Scalar (GrB_Descriptor o, GrB_Field f, GrB_Scalar t) ;
GrB_Info GrB_Descriptor_set_String (GrB_Descriptor o, GrB_Field f, char *t) ;
GrB_Info GrB_Descriptor_set_ENUM   (GrB_Descriptor o, GrB_Field f, int t) ;
GrB_Info GrB_Descriptor_set_VOID   (GrB_Descriptor o, GrB_Field f, void *t, int n) ;

GrB_Info GrB_Type_set_Scalar (GrB_Type o, GrB_Field f, GrB_Scalar t) ;
GrB_Info GrB_Type_set_String (GrB_Type o, GrB_Field f, char *t) ;
GrB_Info GrB_Type_set_ENUM   (GrB_Type o, GrB_Field f, int t) ;
GrB_Info GrB_Type_set_VOID   (GrB_Type o, GrB_Field f, void *t, int n) ;
*/

GrB_Info GrB_Global_set_Scalar (GrB_Field f, GrB_Scalar t) ;
GrB_Info GrB_Global_set_String (GrB_Field f, char *t) ;
GrB_Info GrB_Global_set_ENUM   (GrB_Field f, int t) ;
GrB_Info GrB_Global_set_VOID   (GrB_Field f, void *t /*, int n */) ;

// OK:
#define GB_GLOBAL_SET(f,t)                                                  \
    _Generic                                                                \
    (                                                                       \
        (t),                                                                \
            GrB_Scalar  : GrB_Global_set_Scalar ,                           \
            char *      : GrB_Global_set_String ,                           \
            int         : GrB_Global_set_ENUM   ,                           \
            void *      : GrB_Global_set_VOID                               \
    ) (f,t)

// OK:
#define GB_SCALAR_SET(o,f,t)                                                \
    _Generic                                                                \
    (                                                                       \
        (t),                                                                \
            GrB_Scalar  : GrB_Scalar_set_Scalar ,                           \
            char *      : GrB_Scalar_set_String ,                           \
            int         : GrB_Scalar_set_ENUM   ,                           \
            void *      : GrB_Scalar_set_VOID                               \
    ) (o,f,t)

// OK:
#define GB_SET(o,f,t)                                                       \
    _Generic                                                                \
    (                                                                       \
        (o),                                                                \
            GrB_Scalar :                                                    \
                _Generic                                                    \
                (                                                           \
                    (t),                                                    \
                        GrB_Scalar  : GrB_Scalar_set_Scalar ,               \
                        char *      : GrB_Scalar_set_String ,               \
                        int         : GrB_Scalar_set_ENUM   ,               \
                        void *      : GrB_Scalar_set_VOID                   \
                ) ,                                                         \
            GrB_Vector :                                                    \
                _Generic                                                    \
                (                                                           \
                    (t),                                                    \
                        GrB_Scalar  : GrB_Vector_set_Scalar ,               \
                        char *      : GrB_Vector_set_String ,               \
                        int         : GrB_Vector_set_ENUM   ,               \
                        void *      : GrB_Vector_set_VOID                   \
                )                                                           \
    ) (o,f,t)

// broken:
#define GrB_set(arg1,arg2,...)                                              \
    _Generic                                                                \
    (                                                                       \
        (arg1),                                                             \
            GrB_Scalar : GB_SCALAR_SET (arg1, arg2, __VA_ARGS__) ,          \
            int        : GB_GLOBAL_SET (arg1, arg2)                         \
    )

//==============================================================================
// functions
//==============================================================================

GrB_Info GrB_Scalar_set_Scalar (GrB_Scalar o, GrB_Field f, GrB_Scalar t)
{
    printf ("scalar set scalar: o: %d, f: %d, t->stuff [%d]\n",
        o->stuff, f, t->stuff) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_set_String (GrB_Scalar o, GrB_Field f, char *t)
{
    printf ("scalar set char   : o: %d, f: %d, t [%s]\n",
        o->stuff, f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_set_ENUM   (GrB_Scalar o, GrB_Field f, int t)
{
    printf ("scalar set enum   : o: %d, f: %d, t [%d]\n",
        o->stuff, f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_set_VOID   (GrB_Scalar o, GrB_Field f, void *t /*, int n */)
{
    printf ("scalar set void   : o: %d, f: %d, t [%p]\n",
        o->stuff, f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_set_Scalar (GrB_Vector o, GrB_Field f, GrB_Scalar t)
{
    printf ("vector set scalar: o: %d, f: %d, t->stuff [%d]\n",
        o->morestuff [0], f, t->stuff) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_set_String (GrB_Vector o, GrB_Field f, char *t)
{
    printf ("vector set char   : o: %d, f: %d, t [%s]\n",
        o->morestuff [0], f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_set_ENUM   (GrB_Vector o, GrB_Field f, int t)
{
    printf ("vector set enum   : o: %d, f: %d, t [%d]\n",
        o->morestuff [0], f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_set_VOID   (GrB_Vector o, GrB_Field f, void *t /*, int n */)
{
    printf ("vector set void   : o: %d, f: %d, t [%p]\n",
        o->morestuff [0], f, t) ;
    return (GrB_SUCCESS) ;
}


GrB_Info GrB_Global_set_Scalar (GrB_Field f, GrB_Scalar t)
{
    printf ("global set scalar: f: %d, t->stuff [%d]\n",
        f, t->stuff) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Global_set_String (GrB_Field f, char *t)
{
    printf ("global set char   : f: %d, t [%s]\n", f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Global_set_ENUM   (GrB_Field f, int t)
{
    printf ("global set enum   : f: %d, t [%d]\n", f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Global_set_VOID   (GrB_Field f, void *t /*, int n */)
{
    printf ("global set void   : f: %d, t [%p]\n", f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_new (GrB_Scalar *xhandle)
{
    (*xhandle) = malloc (sizeof (struct GB_Scalar_opaque)) ;
}

GrB_Info GrB_Scalar_free (GrB_Scalar *xhandle)
{
    GrB_Scalar x = *xhandle ;
    free (x) ;
    (*xhandle) = NULL ;
}

GrB_Info GrB_Vector_new (GrB_Vector *vhandle)
{
    (*vhandle) = calloc (1, sizeof (struct GB_Scalar_opaque)) ;
}

GrB_Info GrB_Vector_free (GrB_Vector *xhandle)
{
    GrB_Vector x = *xhandle ;
    free (x) ;
    (*xhandle) = NULL ;
}

int main (void)
{
    GrB_Scalar x, y ;
    GrB_Vector v ;
    GrB_Scalar_new (&x) ;
    GrB_Scalar_new (&y) ;
    GrB_Vector_new (&v) ;
    x->stuff = 42 ;
    y->stuff = 99 ;
    v->morestuff [0] = 777 ;
//  memset (v->morestuff, 0, 8 * sizeof (int)) ;
    int garbage [4] ;
    void *g = garbage ;

//  GrB_set (x, GrB_NAME, "mine") ;
//  GrB_set (3, 99) ;

    // using int's 1 to 6 for the GrB_Field:

    printf ("\nScalar set:\n") ;
    GB_SCALAR_SET (x, 1, "mine") ;
    GB_SCALAR_SET (x, 2, 32) ;
    GB_SCALAR_SET (x, 3, x) ;
    GB_SCALAR_SET (x, 4, g) ;

    printf ("\nGlobal set:\n") ;
    GB_GLOBAL_SET (4, "yours") ;
    GB_GLOBAL_SET (5, 101) ;
    GB_GLOBAL_SET (6, y) ;
    GB_GLOBAL_SET (7, g) ;

    printf ("\nScalar set but more generic:\n") ;
    GB_SET (x, 1, "mine") ;
    GB_SET (x, 2, 32) ;
    GB_SET (x, 3, x) ;
    GB_SET (x, 4, g) ;

    printf ("\nVector set but more generic:\n") ;
    GB_SET (v, 1, "mine") ;
    GB_SET (v, 2, 32) ;
    GB_SET (v, 3, x) ;
    GB_SET (v, 4, g) ;

    // GrB_set (x, GrB_NAME, "mine") ;
    // GrB_set (3, 99) ;

    GrB_Scalar_free (&x) ;
    GrB_Scalar_free (&y) ;
    GrB_Vector_free (&v) ;
}

