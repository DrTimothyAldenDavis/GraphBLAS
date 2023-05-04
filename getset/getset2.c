//==============================================================================
// GrB_get and GrB_set : Tim Davis' revision to the draft spec
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
// user-visible things
//==============================================================================

typedef struct GB_Scalar_opaque *GrB_Scalar ;
typedef struct GB_Vector_opaque *GrB_Vector ;
typedef struct GB_Global_opaque *GrB_Global ;
extern GrB_Global GrB_GLOBAL ;

//==============================================================================
// just placeholders for opaque stuff:
//==============================================================================

struct GB_Global_opaque     // a simple GrB_Scalar, just to be self-contained
{
    int nothing_important ;
} ;
struct GB_Scalar_opaque     // a simple GrB_Scalar, just to be self-contained
{
    int scalarstuff ;
} ;
struct GB_Vector_opaque     // a simple GrB_Vector, just to be self-contained
{
    int vectorstuff [8] ;
} ;

struct GB_Global_opaque GB_Global_thing ;
GrB_Global GrB_GLOBAL = &GB_Global_thing ;

//==============================================================================
// more user-visible things
//==============================================================================

typedef uint64_t GrB_Index ;

GrB_Info GrB_Scalar_set_Scalar (GrB_Scalar o, GrB_Scalar t, GrB_Field f) ;
GrB_Info GrB_Scalar_set_String (GrB_Scalar o, char *t     , GrB_Field f) ;
GrB_Info GrB_Scalar_set_ENUM   (GrB_Scalar o, int t       , GrB_Field f) ;
GrB_Info GrB_Scalar_set_VOID   (GrB_Scalar o, void *t     , GrB_Field f, size_t n) ;

GrB_Info GrB_Vector_set_Scalar (GrB_Vector o, GrB_Scalar t, GrB_Field f) ;
GrB_Info GrB_Vector_set_String (GrB_Vector o, char *t     , GrB_Field f) ;
GrB_Info GrB_Vector_set_ENUM   (GrB_Vector o, int t       , GrB_Field f) ;
GrB_Info GrB_Vector_set_VOID   (GrB_Vector o, void *t     , GrB_Field f, size_t n) ;

GrB_Info GrB_Global_set_Scalar (GrB_Global o, GrB_Scalar t, GrB_Field f) ;
GrB_Info GrB_Global_set_String (GrB_Global o, char *t     , GrB_Field f) ;
GrB_Info GrB_Global_set_ENUM   (GrB_Global o, int t       , GrB_Field f) ;
GrB_Info GrB_Global_set_VOID   (GrB_Global o, void *t     , GrB_Field f, size_t n) ;

// OK this works: note the use of __VA_ARGS__, which is "f" for most methods,
// or "f,n" for the set_VOID methods. 
#define GrB_set(o,t,...)                                                    \
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
                ) ,                                                         \
            GrB_Global :                                                    \
                _Generic                                                    \
                (                                                           \
                    (t),                                                    \
                        GrB_Scalar  : GrB_Global_set_Scalar ,               \
                        char *      : GrB_Global_set_String ,               \
                        int         : GrB_Global_set_ENUM   ,               \
                        void *      : GrB_Global_set_VOID                   \
                )                                                           \
    ) (o, t, __VA_ARGS__)

GrB_Info GrB_Scalar_get_Scalar (GrB_Scalar o, GrB_Scalar t, GrB_Field f) ;
GrB_Info GrB_Scalar_get_String (GrB_Scalar o, char *t     , GrB_Field f) ;
GrB_Info GrB_Scalar_get_ENUM   (GrB_Scalar o, int *t      , GrB_Field f) ;
GrB_Info GrB_Scalar_get_SIZE   (GrB_Scalar o, size_t *t   , GrB_Field f) ;
GrB_Info GrB_Scalar_get_VOID   (GrB_Scalar o, void *t     , GrB_Field f) ;

GrB_Info GrB_Vector_get_Scalar (GrB_Vector o, GrB_Scalar t, GrB_Field f) ;
GrB_Info GrB_Vector_get_String (GrB_Vector o, char *t     , GrB_Field f) ;
GrB_Info GrB_Vector_get_ENUM   (GrB_Vector o, int *t      , GrB_Field f) ;
GrB_Info GrB_Vector_get_SIZE   (GrB_Vector o, size_t *t   , GrB_Field f) ;
GrB_Info GrB_Vector_get_VOID   (GrB_Vector o, void *t     , GrB_Field f) ;

GrB_Info GrB_Global_get_Scalar (GrB_Global o, GrB_Scalar t, GrB_Field f) ;
GrB_Info GrB_Global_get_String (GrB_Global o, char *t     , GrB_Field f) ;
GrB_Info GrB_Global_get_ENUM   (GrB_Global o, int *t      , GrB_Field f) ;
GrB_Info GrB_Global_get_SIZE   (GrB_Global o, size_t *t   , GrB_Field f) ;
GrB_Info GrB_Global_get_VOID   (GrB_Global o, void *t     , GrB_Field f) ;

// OK this works:
#define GrB_get(o,t,f)                                                      \
    _Generic                                                                \
    (                                                                       \
        (o),                                                                \
            GrB_Scalar :                                                    \
                _Generic                                                    \
                (                                                           \
                    (t),                                                    \
                        GrB_Scalar  : GrB_Scalar_get_Scalar ,               \
                        char *      : GrB_Scalar_get_String ,               \
                        int *       : GrB_Scalar_get_ENUM   ,               \
                        size_t *    : GrB_Scalar_get_SIZE   ,               \
                        void *      : GrB_Scalar_get_VOID                   \
                ) ,                                                         \
            GrB_Vector :                                                    \
                _Generic                                                    \
                (                                                           \
                    (t),                                                    \
                        GrB_Scalar  : GrB_Vector_get_Scalar ,               \
                        char *      : GrB_Vector_get_String ,               \
                        int *       : GrB_Vector_get_ENUM   ,               \
                        size_t *    : GrB_Vector_get_SIZE   ,               \
                        void *      : GrB_Vector_get_VOID                   \
                ) ,                                                         \
            GrB_Global :                                                    \
                _Generic                                                    \
                (                                                           \
                    (t),                                                    \
                        GrB_Scalar  : GrB_Global_get_Scalar ,               \
                        char *      : GrB_Global_get_String ,               \
                        int *       : GrB_Global_get_ENUM   ,               \
                        size_t *    : GrB_Global_get_SIZE   ,               \
                        void *      : GrB_Global_get_VOID                   \
                )                                                           \
    ) (o, t, f)

//==============================================================================
// functions
//==============================================================================

GrB_Info GrB_Scalar_set_Scalar (GrB_Scalar o, GrB_Scalar t, GrB_Field f)
{
    printf ("scalar set scalar : o: %d, f: %d, t->scalarstuff [%d]\n",
        o->scalarstuff, f, t->scalarstuff) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_set_String (GrB_Scalar o, char *t     , GrB_Field f)
{
    printf ("scalar set char   : o: %d, f: %d, t [%s]\n",
        o->scalarstuff, f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_set_ENUM   (GrB_Scalar o, int t       , GrB_Field f)
{
    printf ("scalar set enum   : o: %d, f: %d, t [%d]\n",
        o->scalarstuff, f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_set_VOID   (GrB_Scalar o, void *t     , GrB_Field f, size_t n)
{
    printf ("scalar set void   : o: %d, f: %d, t [%p] n %lu\n",
        o->scalarstuff, f, t, n) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_get_Scalar (GrB_Scalar o, GrB_Scalar t, GrB_Field f)
{
    t->scalarstuff = o->scalarstuff ;
    printf ("scalar get scalar : o: %d, f: %d, t->scalarstuff [%d]\n",
        o->scalarstuff, f, t->scalarstuff) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_get_String (GrB_Scalar o, char *t     , GrB_Field f)
{
    strcpy (t, "whatever") ;
    printf ("scalar get char   : o: %d, f: %d, t [%s]\n",
        o->scalarstuff, f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_get_ENUM   (GrB_Scalar o, int *t      , GrB_Field f)
{
    (*t) = o->scalarstuff ;
    printf ("scalar get enum   : o: %d, f: %d, t [%d]\n",
        o->scalarstuff, f, *t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_get_SIZE   (GrB_Scalar o, size_t *t   , GrB_Field f)
{
    (*t) = strlen ("more whatever") + 1 ;
    printf ("scalar get size   : o: %d, f: %d, t [%lu]\n",
        o->scalarstuff, f, *t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_get_VOID   (GrB_Scalar o, void *t     , GrB_Field f)
{
    size_t len = strlen ("more whatever") + 1 ;
    memcpy (t, "more whatever", len) ;
    printf ("scalar get void   : o: %d, f: %d, t [%p]\n",
        o->scalarstuff, f, t) ;
    return (GrB_SUCCESS) ;
}

//==============================================================================

GrB_Info GrB_Vector_set_Scalar (GrB_Vector o, GrB_Scalar t, GrB_Field f)
{
    printf ("vector set scalar : o: %d, f: %d, t->scalarstuff [%d]\n",
        o->vectorstuff [0], f, t->scalarstuff) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_set_String (GrB_Vector o, char *t     , GrB_Field f)
{
    printf ("vector set char   : o: %d, f: %d, t [%s]\n",
        o->vectorstuff [0], f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_set_ENUM   (GrB_Vector o, int t       , GrB_Field f)
{
    printf ("vector set enum   : o: %d, f: %d, t [%d]\n",
        o->vectorstuff [0], f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_set_VOID   (GrB_Vector o, void *t     , GrB_Field f, size_t n)
{
    printf ("vector set void   : o: %d, f: %d, t [%p] n %lu\n",
        o->vectorstuff [0], f, t, n) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_get_Scalar (GrB_Vector o, GrB_Scalar t, GrB_Field f)
{
    t->scalarstuff = o->vectorstuff [0] ;
    printf ("vector get scalar : o: %d, f: %d, t->scalarstuff [%d]\n",
        o->vectorstuff [0], f, t->scalarstuff) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_get_String (GrB_Vector o, char *t     , GrB_Field f)
{
    strcpy (t, "whatever") ;
    printf ("vector get char   : o: %d, f: %d, t [%s]\n",
        o->vectorstuff [0], f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_get_ENUM   (GrB_Vector o, int *t      , GrB_Field f)
{
    (*t) = o->vectorstuff [0] ;
    printf ("vector get enum   : o: %d, f: %d, t [%d]\n",
        o->vectorstuff [0], f, *t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_get_SIZE   (GrB_Vector o, size_t *t   , GrB_Field f)
{
    (*t) = strlen ("more whatever") + 1 ;
    printf ("vector get size   : o: %d, f: %d, t [%lu]\n",
        o->vectorstuff [0], f, *t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_get_VOID   (GrB_Vector o, void *t     , GrB_Field f)
{
    size_t len = strlen ("more whatever") + 1 ;
    memcpy (t, "more whatever", len) ;
    printf ("vector get void   : o: %d, f: %d, t [%p]\n",
        o->vectorstuff [0], f, t) ;
    return (GrB_SUCCESS) ;
}

//==============================================================================

GrB_Info GrB_Global_set_Scalar (GrB_Global o, GrB_Scalar t, GrB_Field f)
{
    o->nothing_important = t->scalarstuff ;
    printf ("global set scalar : f: %d, t->scalarstuff [%d]\n",
        f, t->scalarstuff) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Global_set_String (GrB_Global o, char *t     , GrB_Field f)
{
    printf ("global set char   : f: %d, t [%s]\n", f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Global_set_ENUM   (GrB_Global o, int t       , GrB_Field f)
{
    o->nothing_important = t ;
    printf ("global set enum   : f: %d, t [%d]\n", f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Global_set_VOID   (GrB_Global o, void *t     , GrB_Field f, size_t n)
{
    printf ("global set void   : f: %d, t [%p] n %lu\n", f, t, n) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Global_get_Scalar (GrB_Global o, GrB_Scalar t, GrB_Field f)
{
    t->scalarstuff = o->nothing_important ;
    printf ("global get scalar : o: %d, f: %d, t->scalarstuff [%d]\n",
        o->nothing_important, f, t->scalarstuff) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Global_get_String (GrB_Global o, char *t     , GrB_Field f)
{
    strcpy (t, "globular whatever") ;
    printf ("global get char   : o: %d, f: %d, t [%s]\n",
        o->nothing_important, f, t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Global_get_ENUM   (GrB_Global o, int *t      , GrB_Field f)
{
    (*t) = o->nothing_important ;
    printf ("global get enum   : o: %d, f: %d, t [%d]\n",
        o->nothing_important, f, *t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Global_get_SIZE   (GrB_Global o, size_t *t   , GrB_Field f)
{
    (*t) = strlen ("still more global whatever") + 1 ;
    printf ("global get size   : o: %d, f: %d, t [%lu]\n",
        o->nothing_important, f, *t) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Global_get_VOID   (GrB_Global o, void *t     , GrB_Field f)
{
    size_t len = strlen ("still more more global whatever") + 1 ;
    memcpy (t, "still more global whatever", len) ;
    printf ("global get void   : o: %d, f: %d, t [%p]\n",
        o->nothing_important, f, t) ;
    return (GrB_SUCCESS) ;
}

//==============================================================================

GrB_Info GrB_Scalar_new (GrB_Scalar *xhandle)
{
    (*xhandle) = malloc (sizeof (struct GB_Scalar_opaque)) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Scalar_free (GrB_Scalar *xhandle)
{
    GrB_Scalar x = *xhandle ;
    free (x) ;
    (*xhandle) = NULL ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_new (GrB_Vector *vhandle)
{
    (*vhandle) = calloc (1, sizeof (struct GB_Scalar_opaque)) ;
    return (GrB_SUCCESS) ;
}

GrB_Info GrB_Vector_free (GrB_Vector *xhandle)
{
    GrB_Vector x = *xhandle ;
    free (x) ;
    (*xhandle) = NULL ;
    return (GrB_SUCCESS) ;
}

int main (void)
{
    GrB_Scalar x, y ;
    GrB_Vector v ;
    GrB_Scalar_new (&x) ;
    GrB_Scalar_new (&y) ;
    GrB_Vector_new (&v) ;
    x->scalarstuff = 42 ;
    y->scalarstuff = 99 ;
    v->vectorstuff [0] = 777 ;
    int garbage [4] ;
    void *g = garbage ;

    // using int's 1 to 12 for the GrB_Field:

    printf ("\nScalar set:\n") ;
    GrB_set (x, "mine", 1) ;
    GrB_set (x, 32, 2) ;
    GrB_set (x, y, 3) ;
    GrB_set (x, g, 4, 1010) ;

    printf ("\nScalar get:\n") ;
    size_t size = 0 ;
    char *mystuff ;
    GrB_get (x, &size, 1) ;
    printf ("got size %lu\n", size) ;
    mystuff = malloc (size) ;
    GrB_get (x, mystuff, 3) ;
    printf ("my stuff is [%s]\n", mystuff) ;
    int w ;
    GrB_get (x, &w, 2) ;
    printf ("my integer w is [%d]\n", w) ;
    GrB_get (x, y, 3) ;
    printf ("got from x: %d\n", y->scalarstuff) ;

    char *myvoid ;
    GrB_get (x, &size, 4) ;
    printf ("got void size %lu\n", size) ;
    myvoid = malloc (size) ;
    GrB_get (x, myvoid, 4) ;
    printf ("got void x: %s\n", myvoid) ;
    free (mystuff) ;
    free (myvoid) ;

    printf ("\nVector set:\n") ;
    GrB_set (v, "mine", 9) ;
    GrB_set (v, 32, 10) ;
    GrB_set (v, x, 11) ;
    GrB_set (v, g, 12, 909) ;

    printf ("\nGlobal set:\n") ;
    GrB_set (GrB_GLOBAL, "yours", 4) ;
    GrB_set (GrB_GLOBAL, 101, 5) ;
    GrB_set (GrB_GLOBAL, y, 6) ;
    GrB_set (GrB_GLOBAL, g, 7, 777) ;

    GrB_Scalar_free (&x) ;
    GrB_Scalar_free (&y) ;
    GrB_Vector_free (&v) ;
}

