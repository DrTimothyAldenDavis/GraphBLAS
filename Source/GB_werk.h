//------------------------------------------------------------------------------
// GB_werk.h: definitions for werkspace management on the Werk stack
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_WERK_H
#define GB_WERK_H

//------------------------------------------------------------------------------
// GB_Werk: error logging and Werk space
//------------------------------------------------------------------------------

// Error messages are logged in Werk->logger_handle, on the stack which is
// handle to the input/output matrix/vector (typically C).  If the user-defined
// data types, operators, etc have really long names, the error messages are
// safely truncated (via snprintf).  This is intentional, but gcc with
// -Wformat-truncation will print a warning (see pragmas above).  Ignore the
// warning.

// GB_WERK_SIZE is the size of a small fixed-sized array in the Werk, used
// for small werkspace allocations (typically O(# of threads or # tasks)).
// GB_WERK_SIZE must be a multiple of 8.  The Werk->Stack array is placed first
// in the GB_Werk struct, to ensure proper alignment.

#define GB_WERK_SIZE 16384

typedef struct
{
    GB_void Stack [GB_WERK_SIZE] ;  // werkspace stack
    const char *where ;             // GraphBLAS function where error occurred
    char **logger_handle ;          // error report
    size_t *logger_size_handle ;
    int pwerk ;                     // top of Werk stack, initially zero
}
GB_Werk_struct ;

typedef GB_Werk_struct *GB_Werk ;

// GB_WHERE keeps track of the currently running user-callable function.
// User-callable functions in this implementation are written so that they do
// not call other unrelated user-callable functions (except for GrB_*free).
// Related user-callable functions can call each other since they all report
// the same type-generic name.  Internal functions can be called by many
// different user-callable functions, directly or indirectly.  It would not be
// helpful to report the name of an internal function that flagged an error
// condition.  Thus, each time a user-callable function is entered, it logs the
// name of the function with the GB_WHERE macro.

#define GB_WERK(where_string)                                       \
    /* construct the Werk */                                        \
    GB_Werk_struct Werk_struct ;                                    \
    GB_Werk Werk = &Werk_struct ;                                   \
    /* set Werk->where so GrB_error can report it if needed */      \
    Werk->where = where_string ;                                    \
    /* get the pointer to where any error will be logged */         \
    Werk->logger_handle = NULL ;                                    \
    Werk->logger_size_handle = NULL ;                               \
    /* initialize the Werk stack */                                 \
    Werk->pwerk = 0 ;

// C is a matrix, vector, scalar, or descriptor
#define GB_WHERE(C,where_string)                                    \
    if (!GB_Global_GrB_init_called_get ( ))                         \
    {                                                               \
        return (GrB_PANIC) ; /* GrB_init not called */              \
    }                                                               \
    GB_WERK (where_string)                                          \
    if (C != NULL)                                                  \
    {                                                               \
        /* free any prior error logged in the object */             \
        GB_FREE (&(C->logger), C->logger_size) ;                    \
        Werk->logger_handle = &(C->logger) ;                        \
        Werk->logger_size_handle = &(C->logger_size) ;              \
    }

// create the Werk, with no error logging
#define GB_WHERE1(where_string)                                     \
    if (!GB_Global_GrB_init_called_get ( ))                         \
    {                                                               \
        return (GrB_PANIC) ; /* GrB_init not called */              \
    }                                                               \
    GB_WERK (where_string)

//------------------------------------------------------------------------------
// error logging
//------------------------------------------------------------------------------

// The GB_ERROR macro logs an error in the logger error string.
//
//  if (i >= nrows)
//  {
//      GB_ERROR (GrB_INDEX_OUT_OF_BOUNDS,
//          "Row index %d out of bounds; must be < %d", i, nrows) ;
//  }
//
// The user can then do:
//
//  const char *error ;
//  GrB_error (&error, A) ;
//  printf ("%s", error) ;

const char *GB_status_code (GrB_Info info) ;

// maximum size of the error logger string
#define GB_LOGGER_LEN 384

// log an error in the error logger string and return the error
#define GB_ERROR(info,format,...)                                           \
{                                                                           \
    if (Werk != NULL)                                                       \
    {                                                                       \
        char **logger_handle = Werk->logger_handle ;                        \
        if (logger_handle != NULL)                                          \
        {                                                                   \
            size_t *logger_size_handle = Werk->logger_size_handle ;         \
            (*logger_handle) = GB_CALLOC (GB_LOGGER_LEN+1, char,            \
                logger_size_handle) ;                                       \
            if ((*logger_handle) != NULL)                                   \
            {                                                               \
                snprintf ((*logger_handle), GB_LOGGER_LEN,                  \
                    "GraphBLAS error: %s\nfunction: %s\n" format,           \
                    GB_status_code (info), Werk->where, __VA_ARGS__) ;      \
            }                                                               \
        }                                                                   \
    }                                                                       \
    return (info) ;                                                         \
}

//------------------------------------------------------------------------------
// GB_werk_push/pop: manage werkspace in the Werk stack
//------------------------------------------------------------------------------

// Werk->Stack is a small fixed-size array that is allocated on the stack
// of any user-callable GraphBLAS function.  It is used for small werkspace
// allocations.

// GB_ROUND8(s) rounds up s to a multiple of 8
#define GB_ROUND8(s) (((s) + 7) & (~0x7))

//------------------------------------------------------------------------------
// GB_werk_push: allocate werkspace from the Werk stack or malloc
//------------------------------------------------------------------------------

// The werkspace is allocated from the Werk static if it small enough and space
// is available.  Otherwise it is allocated by malloc.

static inline void *GB_werk_push    // return pointer to newly allocated space
(
    // output
    size_t *size_allocated,         // # of bytes actually allocated
    bool *on_stack,                 // true if werkspace is from Werk stack
    // input
    size_t nitems,                  // # of items to allocate
    size_t size_of_item,            // size of each item
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (on_stack != NULL) ;
    ASSERT (size_allocated != NULL) ;

    //--------------------------------------------------------------------------
    // determine where to allocate the werkspace
    //--------------------------------------------------------------------------

    size_t size ;
    if (Werk == NULL || nitems > GB_WERK_SIZE || size_of_item > GB_WERK_SIZE
        #ifdef GBCOVER
        // Werk stack can be disabled for test coverage
        || (GB_Global_hack_get (1) != 0)
        #endif
    )
    { 
        // no context, or werkspace is too large to allocate from the Werk stack
        (*on_stack) = false ;
    }
    else
    { 
        // try to allocate from the Werk stack
        size = GB_ROUND8 (nitems * size_of_item) ;
        ASSERT (size % 8 == 0) ;        // size is rounded up to a multiple of 8
        size_t freespace = GB_WERK_SIZE - Werk->pwerk ;
        ASSERT (freespace % 8 == 0) ;   // thus freespace is also multiple of 8
        (*on_stack) = (size <= freespace) ;
    }

    //--------------------------------------------------------------------------
    // allocate the werkspace
    //--------------------------------------------------------------------------

    if (*on_stack)
    { 
        // allocate the werkspace from the Werk stack
        GB_void *p = Werk->Stack + Werk->pwerk ;
        Werk->pwerk += (int) size ;
        (*size_allocated) = size ;
        return ((void *) p) ;
    }
    else
    { 
        // allocate the werkspace from malloc
        return (GB_malloc_memory (nitems, size_of_item, size_allocated)) ;
    }
}

//------------------------------------------------------------------------------
// Werk helper macros
//------------------------------------------------------------------------------

// declare a werkspace X of a given type
#define GB_WERK_DECLARE(X,type)                             \
    type *restrict X = NULL ;                               \
    bool X ## _on_stack = false ;                           \
    size_t X ## _nitems = 0, X ## _size_allocated = 0 ;

// push werkspace X
#define GB_WERK_PUSH(X,nitems,type)                                         \
    X ## _nitems = (nitems) ;                                               \
    X = (type *) GB_werk_push (&(X ## _size_allocated), &(X ## _on_stack),  \
        X ## _nitems, sizeof (type), Werk) ; 

// pop werkspace X
#define GB_WERK_POP(X,type)                                                 \
    X = (type *) GB_werk_pop (X, &(X ## _size_allocated), X ## _on_stack,   \
        X ## _nitems, sizeof (type), Werk) ; 

//------------------------------------------------------------------------------
// GB_werk_pop:  free werkspace from the Werk stack
//------------------------------------------------------------------------------

// If the werkspace was allocated from the Werk stack, it must be at the top of
// the stack to free it properly.  Freeing a werkspace in the middle of the
// Werk stack also frees everything above it.  This is not a problem if that
// space is also being freed, but the assertion below ensures that the freeing
// werkspace from the Werk stack is done in LIFO order, like a stack.

static inline void *GB_werk_pop     // free the top block of werkspace memory
(
    // input/output
    void *p,                        // werkspace to free
    size_t *size_allocated,         // # of bytes actually allocated for p
    // input
    bool on_stack,                  // true if werkspace is from Werk stack
    size_t nitems,                  // # of items to allocate
    size_t size_of_item,            // size of each item
    GB_Werk Werk
)
{
    ASSERT (size_allocated != NULL) ;

    if (p == NULL)
    { 
        // nothing to do
    }
    else if (on_stack)
    { 
        // werkspace was allocated from the Werk stack
        ASSERT ((*size_allocated) == GB_ROUND8 (nitems * size_of_item)) ;
        ASSERT (Werk != NULL) ;
        ASSERT ((*size_allocated) % 8 == 0) ;
        ASSERT (((GB_void *) p) + (*size_allocated) ==
                Werk->Stack + Werk->pwerk) ;
        Werk->pwerk = ((GB_void *) p) - Werk->Stack ;
        (*size_allocated) = 0 ;
    }
    else
    { 
        // werkspace was allocated from malloc
        GB_free_memory (&p, *size_allocated) ;
    }
    return (NULL) ;                 // return NULL to indicate p was freed
}

#endif

