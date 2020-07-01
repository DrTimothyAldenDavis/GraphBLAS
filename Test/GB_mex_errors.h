//------------------------------------------------------------------------------
// GB_mex_errors.h: error handling macros
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#define FAIL(s)                                                             \
{                                                                           \
    fprintf (f,"\ntest failure: line %d\n", __LINE__) ;                     \
    fprintf (f,"%s\n", GB_STR(s)) ;                                         \
    fclose (f) ;                                                            \
    mexErrMsgTxt (GB_STR(s) " line: " GB_XSTR(__LINE__)) ;                  \
}

#undef CHECK
#define CHECK(x)    if (!(x)) FAIL(x) ;
#define CHECK2(x,s) if (!(x)) FAIL(s) ;

// assert that a method should return a particular error code
#define ERR(method)                                                         \
{                                                                           \
    info = method ;                                                         \
    fprintf (f, "line %d: info %d\n", __LINE__, info) ;                     \
    if (info != expected) fprintf (f, "got %d expected %d\n", info, expected) ; \
    CHECK2 (info == expected, method) ;                                     \
}

// assert that a method should return a particular error code: with logger
#define ERR1(C,method)                                                      \
{                                                                           \
    info = method ;                                                         \
    fprintf (f, "\nline %d: info %d, error logger:\n", __LINE__, info) ;    \
    char *error_logger ;                                                    \
    GrB_error (&error_logger, C) ;                                          \
    fprintf (f,"[%s]\n", error_logger) ;                                    \
    if (info != expected) fprintf (f, "got %d expected %d\n", info, expected) ; \
    CHECK2 (info == expected, method) ;                                     \
}

// assert that a method should succeed
#define OK(method)                                                          \
{                                                                           \
    info = method ;                                                         \
    if (! (info == GrB_SUCCESS || info == GrB_NO_VALUE))                    \
    {                                                                       \
        fprintf (f,"[%d] >>>>>>>>\n", info) ;                               \
        printf ("[%d] %s\n", info) ;                                        \
        FAIL (method) ;                                                     \
    }                                                                       \
}

