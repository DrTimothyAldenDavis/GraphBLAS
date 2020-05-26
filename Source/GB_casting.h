//------------------------------------------------------------------------------
// GB_casting.h: define the unary typecasting functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#ifndef GB_CASTING_H
#define GB_CASTING_H

//------------------------------------------------------------------------------
// typecasting from double to integer
//------------------------------------------------------------------------------

// The GraphBLAS C API states that typecasting follows the rules of the C
// language.  However, the ANSI C11 language specification states that results
// are undefined when typecasting a float or double to an integer value that is
// outside the range of the integer type.  MATLAB handles this case by
// typecasting a float or double that is larger than the maximum integer to the
// max integer, and a value less than the minimum integer to the min integer.
// GraphBLAS v3.3.0 follows this rule (earlier versions handled this
// differently).  NaN's are typecasted to the integer value zero.

inline int8_t GB_cast_to_int8_t (double x)
{ 
    if (isnan (x)) return (0) ;
    if (x <= (double) INT8_MIN) return (INT8_MIN) ;
    if (x >= (double) INT8_MAX) return (INT8_MAX) ;
    return ((int8_t) x) ;
}

inline int16_t GB_cast_to_int16_t (double x)
{ 
    if (isnan (x)) return (0) ;
    if (x <= (double) INT16_MIN) return (INT16_MIN) ;
    if (x >= (double) INT16_MAX) return (INT16_MAX) ;
    return ((int16_t) x) ;
}

inline int32_t GB_cast_to_int32_t (double x)
{ 
    if (isnan (x)) return (0) ;
    if (x <= (double) INT32_MIN) return (INT32_MIN) ;
    if (x >= (double) INT32_MAX) return (INT32_MAX) ;
    return ((int32_t) x) ;
}

inline int64_t GB_cast_to_int64_t (double x)
{ 
    if (isnan (x)) return (0) ;
    if (x <= (double) INT64_MIN) return (INT64_MIN) ;
    if (x >= (double) INT64_MAX) return (INT64_MAX) ;
    return ((int64_t) x) ;
}

inline uint8_t GB_cast_to_uint8_t (double x)
{ 
    if (isnan (x) || x <= 0) return (0) ;
    if (x >= (double) UINT8_MAX) return (UINT8_MAX) ;
    return ((uint8_t) x) ;
}

inline uint16_t GB_cast_to_uint16_t (double x)
{ 
    if (isnan (x) || x <= 0) return (0) ;
    if (x >= (double) UINT16_MAX) return (UINT16_MAX) ;
    return ((uint16_t) x) ;
}

inline uint32_t GB_cast_to_uint32_t (double x)
{ 
    if (isnan (x) || x <= 0) return (0) ;
    if (x >= (double) UINT32_MAX) return (UINT32_MAX) ;
    return ((uint32_t) x) ;
}

inline uint64_t GB_cast_to_uint64_t (double x)
{ 
    if (isnan (x) || x <= 0) return (0) ;
    if (x >= (double) UINT64_MAX) return (UINT64_MAX) ;
    return ((uint64_t) x) ;
}

//------------------------------------------------------------------------------
// unary typecast operators, used in GB_cast_factory.c.
//------------------------------------------------------------------------------

// construct the typecast functions: z = (ztype) x, where x has type xtype.

// GB_cast_ztype_xtype casts a single value ((xtype) x) into the result
// ((ztype) z).  Functions with the xtype and ztype the same, such as
// GB_cast_double_double, simply copy their input to their output.

// The s parameter is not used in these functions.  It is present because one
// function returned by GB_cast_factory requires it (GB_copy_user_user).

#define GB_CAST_FUNCTION(ztype,xtype)                                   \
inline void GB_cast_ ## ztype ## _ ## xtype                             \
(                                                                       \
    void *z,            /* typecasted output, of type ztype */          \
    const void *x,      /* input value to typecast, of type xtype */    \
    size_t s            /* size of type, for GB_copy_user_user only */  \
)                                                                       \
{                                                                       \
    xtype xx = (*((xtype *) x)) ;                                       \
    ztype zz = GB_CAST (ztype, xx) ;                                    \
    (*((ztype *) z)) = zz ;                                             \
}

//------------------------------------------------------------------------------
// typecast to boolean
//------------------------------------------------------------------------------

// Typecasting a NaN to a bool results in 'true', as defined by the ANSI C11
// standard (NaN converts to true, since Nan != 0 is true).  MATLAB throws an
// error when trying to convert NaN's to logical.  GraphBLAS follows the ANSI
// C11 standard in this case.

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (bool      , bool      )
#undef  GB_CAST
#define GB_CAST(ztype,x) (x != 0)
GB_CAST_FUNCTION (bool      , int8_t    )
GB_CAST_FUNCTION (bool      , int16_t   )
GB_CAST_FUNCTION (bool      , int32_t   )
GB_CAST_FUNCTION (bool      , int64_t   )
GB_CAST_FUNCTION (bool      , uint8_t   )
GB_CAST_FUNCTION (bool      , uint16_t  )
GB_CAST_FUNCTION (bool      , uint32_t  )
GB_CAST_FUNCTION (bool      , uint64_t  )
GB_CAST_FUNCTION (bool      , float     )
GB_CAST_FUNCTION (bool      , double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) (crealf (x) != 0 || cimagf (x) != 0)
GB_CAST_FUNCTION (bool      , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) (creal (x) != 0 || cimag (x) != 0)
GB_CAST_FUNCTION (bool      , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to int8_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (int8_t    , bool      )
GB_CAST_FUNCTION (int8_t    , int8_t    )
GB_CAST_FUNCTION (int8_t    , int16_t   )
GB_CAST_FUNCTION (int8_t    , int32_t   )
GB_CAST_FUNCTION (int8_t    , int64_t   )
GB_CAST_FUNCTION (int8_t    , uint8_t   )
GB_CAST_FUNCTION (int8_t    , uint16_t  )
GB_CAST_FUNCTION (int8_t    , uint32_t  )
GB_CAST_FUNCTION (int8_t    , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int8_t ((double) x)
GB_CAST_FUNCTION (int8_t    , float     )
GB_CAST_FUNCTION (int8_t    , double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int8_t ((double) crealf (x))
GB_CAST_FUNCTION (int8_t    , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int8_t (creal (x))
GB_CAST_FUNCTION (int8_t    , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to int16_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (int16_t   , bool      )
GB_CAST_FUNCTION (int16_t   , int8_t    )
GB_CAST_FUNCTION (int16_t   , int16_t   )
GB_CAST_FUNCTION (int16_t   , int32_t   )
GB_CAST_FUNCTION (int16_t   , int64_t   )
GB_CAST_FUNCTION (int16_t   , uint8_t   )
GB_CAST_FUNCTION (int16_t   , uint16_t  )
GB_CAST_FUNCTION (int16_t   , uint32_t  )
GB_CAST_FUNCTION (int16_t   , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int16_t ((double) x)
GB_CAST_FUNCTION (int16_t   , float     )
GB_CAST_FUNCTION (int16_t   , double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int16_t ((double) crealf (x))
GB_CAST_FUNCTION (int16_t   , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int16_t (creal (x))
GB_CAST_FUNCTION (int16_t   , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to int32_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (int32_t   , bool      )
GB_CAST_FUNCTION (int32_t   , int8_t    )
GB_CAST_FUNCTION (int32_t   , int16_t   )
GB_CAST_FUNCTION (int32_t   , int32_t   )
GB_CAST_FUNCTION (int32_t   , int64_t   )
GB_CAST_FUNCTION (int32_t   , uint8_t   )
GB_CAST_FUNCTION (int32_t   , uint16_t  )
GB_CAST_FUNCTION (int32_t   , uint32_t  )
GB_CAST_FUNCTION (int32_t   , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int32_t ((double) x)
GB_CAST_FUNCTION (int32_t   , float     )
GB_CAST_FUNCTION (int32_t   , double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int32_t ((double) crealf (x))
GB_CAST_FUNCTION (int32_t   , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int32_t (creal (x))
GB_CAST_FUNCTION (int32_t   , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to int64_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (int64_t   , bool      )
GB_CAST_FUNCTION (int64_t   , int8_t    )
GB_CAST_FUNCTION (int64_t   , int16_t   )
GB_CAST_FUNCTION (int64_t   , int32_t   )
GB_CAST_FUNCTION (int64_t   , int64_t   )
GB_CAST_FUNCTION (int64_t   , uint8_t   )
GB_CAST_FUNCTION (int64_t   , uint16_t  )
GB_CAST_FUNCTION (int64_t   , uint32_t  )
GB_CAST_FUNCTION (int64_t   , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int64_t ((double) x)
GB_CAST_FUNCTION (int64_t   , float     )
GB_CAST_FUNCTION (int64_t   , double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int64_t ((double) crealf (x))
GB_CAST_FUNCTION (int64_t   , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_int64_t (creal (x))
GB_CAST_FUNCTION (int64_t   , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to uint8_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (uint8_t   , bool      )
GB_CAST_FUNCTION (uint8_t   , int8_t    )
GB_CAST_FUNCTION (uint8_t   , int16_t   )
GB_CAST_FUNCTION (uint8_t   , int32_t   )
GB_CAST_FUNCTION (uint8_t   , int64_t   )
GB_CAST_FUNCTION (uint8_t   , uint8_t   )
GB_CAST_FUNCTION (uint8_t   , uint16_t  )
GB_CAST_FUNCTION (uint8_t   , uint32_t  )
GB_CAST_FUNCTION (uint8_t   , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint8_t ((double) x)
GB_CAST_FUNCTION (uint8_t   , float     )
GB_CAST_FUNCTION (uint8_t   , double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint8_t ((double) crealf (x))
GB_CAST_FUNCTION (uint8_t   , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint8_t (creal (x))
GB_CAST_FUNCTION (uint8_t   , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to uint16_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (uint16_t  , bool      )
GB_CAST_FUNCTION (uint16_t  , int8_t    )
GB_CAST_FUNCTION (uint16_t  , int16_t   )
GB_CAST_FUNCTION (uint16_t  , int32_t   )
GB_CAST_FUNCTION (uint16_t  , int64_t   )
GB_CAST_FUNCTION (uint16_t  , uint8_t   )
GB_CAST_FUNCTION (uint16_t  , uint16_t  )
GB_CAST_FUNCTION (uint16_t  , uint32_t  )
GB_CAST_FUNCTION (uint16_t  , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint16_t ((double) x)
GB_CAST_FUNCTION (uint16_t  , float     )
GB_CAST_FUNCTION (uint16_t  , double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint16_t ((double) crealf (x))
GB_CAST_FUNCTION (uint16_t  , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint16_t (creal (x))
GB_CAST_FUNCTION (uint16_t  , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to uint32_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (uint32_t  , bool      )
GB_CAST_FUNCTION (uint32_t  , int8_t    )
GB_CAST_FUNCTION (uint32_t  , int16_t   )
GB_CAST_FUNCTION (uint32_t  , int32_t   )
GB_CAST_FUNCTION (uint32_t  , int64_t   )
GB_CAST_FUNCTION (uint32_t  , uint8_t   )
GB_CAST_FUNCTION (uint32_t  , uint16_t  )
GB_CAST_FUNCTION (uint32_t  , uint32_t  )
GB_CAST_FUNCTION (uint32_t  , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint32_t ((double) x)
GB_CAST_FUNCTION (uint32_t  , float     )
GB_CAST_FUNCTION (uint32_t  , double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint32_t ((double) crealf (x))
GB_CAST_FUNCTION (uint32_t  , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint32_t (creal (x))
GB_CAST_FUNCTION (uint32_t  , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to uint64_t
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (uint64_t  , bool      )
GB_CAST_FUNCTION (uint64_t  , int8_t    )
GB_CAST_FUNCTION (uint64_t  , int16_t   )
GB_CAST_FUNCTION (uint64_t  , int32_t   )
GB_CAST_FUNCTION (uint64_t  , int64_t   )
GB_CAST_FUNCTION (uint64_t  , uint8_t   )
GB_CAST_FUNCTION (uint64_t  , uint16_t  )
GB_CAST_FUNCTION (uint64_t  , uint32_t  )
GB_CAST_FUNCTION (uint64_t  , uint64_t  )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint64_t ((double) x)
GB_CAST_FUNCTION (uint64_t  , float     )
GB_CAST_FUNCTION (uint64_t  , double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint64_t ((double) crealf (x))
GB_CAST_FUNCTION (uint64_t  , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GB_cast_to_uint64_t (creal (x))
GB_CAST_FUNCTION (uint64_t  , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to float
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (float     , bool      )
GB_CAST_FUNCTION (float     , int8_t    )
GB_CAST_FUNCTION (float     , int16_t   )
GB_CAST_FUNCTION (float     , int32_t   )
GB_CAST_FUNCTION (float     , int64_t   )
GB_CAST_FUNCTION (float     , uint8_t   )
GB_CAST_FUNCTION (float     , uint16_t  )
GB_CAST_FUNCTION (float     , uint32_t  )
GB_CAST_FUNCTION (float     , uint64_t  )
GB_CAST_FUNCTION (float     , float     )
GB_CAST_FUNCTION (float     , double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) crealf (x)
GB_CAST_FUNCTION (float     , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) ((float) creal (x))
GB_CAST_FUNCTION (float     , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to double
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) (ztype) x
GB_CAST_FUNCTION (double    , bool      )
GB_CAST_FUNCTION (double    , int8_t    )
GB_CAST_FUNCTION (double    , int16_t   )
GB_CAST_FUNCTION (double    , int32_t   )
GB_CAST_FUNCTION (double    , int64_t   )
GB_CAST_FUNCTION (double    , uint8_t   )
GB_CAST_FUNCTION (double    , uint16_t  )
GB_CAST_FUNCTION (double    , uint32_t  )
GB_CAST_FUNCTION (double    , uint64_t  )
GB_CAST_FUNCTION (double    , float     )
GB_CAST_FUNCTION (double    , double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) ((double) crealf (x))
GB_CAST_FUNCTION (double    , GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) creal (x)
GB_CAST_FUNCTION (double    , GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to float complex
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) GxB_CMPLXF ((float) x, (float) 0)
GB_CAST_FUNCTION (GxB_FC32_t, bool      )
GB_CAST_FUNCTION (GxB_FC32_t, int8_t    )
GB_CAST_FUNCTION (GxB_FC32_t, int16_t   )
GB_CAST_FUNCTION (GxB_FC32_t, int32_t   )
GB_CAST_FUNCTION (GxB_FC32_t, int64_t   )
GB_CAST_FUNCTION (GxB_FC32_t, uint8_t   )
GB_CAST_FUNCTION (GxB_FC32_t, uint16_t  )
GB_CAST_FUNCTION (GxB_FC32_t, uint32_t  )
GB_CAST_FUNCTION (GxB_FC32_t, uint64_t  )
GB_CAST_FUNCTION (GxB_FC32_t, float     )
GB_CAST_FUNCTION (GxB_FC32_t, double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) x
GB_CAST_FUNCTION (GxB_FC32_t, GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) GxB_CMPLXF ((float) creal (x), (float) cimag (x))
GB_CAST_FUNCTION (GxB_FC32_t, GxB_FC64_t)

//------------------------------------------------------------------------------
// typecast to double complex
//------------------------------------------------------------------------------

#undef  GB_CAST
#define GB_CAST(ztype,x) GxB_CMPLX ((double) x, (double) 0)
GB_CAST_FUNCTION (GxB_FC64_t, bool      )
GB_CAST_FUNCTION (GxB_FC64_t, int8_t    )
GB_CAST_FUNCTION (GxB_FC64_t, int16_t   )
GB_CAST_FUNCTION (GxB_FC64_t, int32_t   )
GB_CAST_FUNCTION (GxB_FC64_t, int64_t   )
GB_CAST_FUNCTION (GxB_FC64_t, uint8_t   )
GB_CAST_FUNCTION (GxB_FC64_t, uint16_t  )
GB_CAST_FUNCTION (GxB_FC64_t, uint32_t  )
GB_CAST_FUNCTION (GxB_FC64_t, uint64_t  )
GB_CAST_FUNCTION (GxB_FC64_t, float     )
GB_CAST_FUNCTION (GxB_FC64_t, double    )
#undef  GB_CAST
#define GB_CAST(ztype,x) GxB_CMPLX ((double) crealf (x), (double) cimagf (x))
GB_CAST_FUNCTION (GxB_FC64_t, GxB_FC32_t)
#undef  GB_CAST
#define GB_CAST(ztype,x) x
GB_CAST_FUNCTION (GxB_FC64_t, GxB_FC64_t)

#undef  GB_CAST
#undef  GB_CAST_FUNCTION
#endif

