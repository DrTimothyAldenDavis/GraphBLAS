//------------------------------------------------------------------------------
// GrB_Descriptor_get_*: get a field in a descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

//------------------------------------------------------------------------------

static GrB_Info GB_desc_get
(
    GrB_Descriptor desc,        // descriptor to query; NULL is ok
    int *value,                 // return value of the descriptor
    int field                   // parameter to query
)
{

    //--------------------------------------------------------------------------
    // get the parameter
    //--------------------------------------------------------------------------

    switch (field)
    {
        case GrB_OUTP  : GB_cov[10453]++ ;  
// covered (10453): 72

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->out) ;
            break ;

        case GrB_MASK  : GB_cov[10454]++ ;  
// covered (10454): 80

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->mask) ;
            break ;

        case GrB_INP0  : GB_cov[10455]++ ;  
// covered (10455): 68

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->in0) ;
            break ;

        case GrB_INP1  : GB_cov[10456]++ ;  
// covered (10456): 68

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->in1) ;
            break ;

        case GxB_AxB_METHOD  : GB_cov[10457]++ ;  
// covered (10457): 74

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->axb) ;
            break ;

        case GxB_SORT  : GB_cov[10458]++ ;  
// covered (10458): 68

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->do_sort);
            break ;

        case GxB_COMPRESSION  : GB_cov[10459]++ ;  
// covered (10459): 68

            (*value) = (int32_t) ((desc == NULL) ?
                GrB_DEFAULT : desc->compression) ; 
            break ;

        case GxB_IMPORT  : GB_cov[10460]++ ;  
// covered (10460): 70

            (*value) = (int32_t) ((desc == NULL) ? GrB_DEFAULT : desc->import) ;
            if ((*value) != GrB_DEFAULT) (*value) = GxB_SECURE_IMPORT ;
            break ;

        default  : GB_cov[10461]++ ;  
// covered (10461): 1

            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_get_Scalar
(
    GrB_Descriptor desc,
    GrB_Scalar value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Descriptor_get_Scalar (desc, value, field)") ;
    GB_RETURN_IF_FAULTY (desc) ;
    GB_RETURN_IF_NULL_OR_FAULTY (value) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    int i ;
    GrB_Info info = GB_desc_get (desc, &i, field) ;
    if (info == GrB_SUCCESS)
    {   GB_cov[10462]++ ;
// covered (10462): 256
        info = GB_setElement ((GrB_Matrix) value, NULL, &i, 0, 0,   
            GB_INT32_code, Werk) ;
    }
    return (info) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_get_String
//------------------------------------------------------------------------------

#define DNAME(d)                    \
{                                   \
    if (desc == d)                  \
    {                               \
        strcpy (value, #d) ;        \
        return (GrB_SUCCESS) ;      \
    }                               \
}

GrB_Info GrB_Descriptor_get_String
(
    GrB_Descriptor desc,
    char * value,
    GrB_Field field
)
{   GB_cov[10463]++ ;
// covered (10463): 259

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Descriptor_get_String (desc, value, field)") ;
    GB_RETURN_IF_FAULTY (desc) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the name
    //--------------------------------------------------------------------------

    DNAME (GrB_NULL        ) ;
    DNAME (GrB_DESC_T1     ) ;
    DNAME (GrB_DESC_T0     ) ;
    DNAME (GrB_DESC_T0T1   ) ;

    DNAME (GrB_DESC_C      ) ;
    DNAME (GrB_DESC_CT1    ) ;
    DNAME (GrB_DESC_CT0    ) ;
    DNAME (GrB_DESC_CT0T1  ) ;

    DNAME (GrB_DESC_S      ) ;
    DNAME (GrB_DESC_ST1    ) ;
    DNAME (GrB_DESC_ST0    ) ;
    DNAME (GrB_DESC_ST0T1  ) ;

    DNAME (GrB_DESC_SC     ) ;
    DNAME (GrB_DESC_SCT1   ) ;
    DNAME (GrB_DESC_SCT0   ) ;
    DNAME (GrB_DESC_SCT0T1 ) ;

    DNAME (GrB_DESC_R      ) ;
    DNAME (GrB_DESC_RT1    ) ;
    DNAME (GrB_DESC_RT0    ) ;
    DNAME (GrB_DESC_RT0T1  ) ;

    DNAME (GrB_DESC_RC     ) ;
    DNAME (GrB_DESC_RCT1   ) ;
    DNAME (GrB_DESC_RCT0   ) ;
    DNAME (GrB_DESC_RCT0T1 ) ;

    DNAME (GrB_DESC_RS     ) ;
    DNAME (GrB_DESC_RST1   ) ;
    DNAME (GrB_DESC_RST0   ) ;
    DNAME (GrB_DESC_RST0T1 ) ;

    DNAME (GrB_DESC_RSC    ) ;
    DNAME (GrB_DESC_RSCT1  ) ;
    DNAME (GrB_DESC_RSCT0  ) ;
    DNAME (GrB_DESC_RSCT0T1) ;

    // user-defined descriptor
    (*value) = '\0' ;

    printf ("get user name [%s]:%d\n", desc->user_name,
        (int) (desc->user_name_size)) ; 

    if (desc->user_name_size > 0)
    {   GB_cov[10464]++ ;
// covered (10464): 1
        // user-defined descriptor, with name defined by GrB_set
        strcpy (value, desc->user_name) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_get_ENUM
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_get_ENUM
(
    GrB_Descriptor desc,
    int * value,
    GrB_Field field
)
{   GB_cov[10465]++ ;
// covered (10465): 313

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Descriptor_get_ENUM (desc, value, field)") ;
    GB_RETURN_IF_FAULTY (desc) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    return (GB_desc_get (desc, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_get_SIZE
(
    GrB_Descriptor desc,
    size_t * value,
    GrB_Field field
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Descriptor_get_SIZE (desc, value, field)") ;
    GB_RETURN_IF_FAULTY (desc) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc for get", GB0) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    if (field != GrB_NAME)
    {   GB_cov[10466]++ ;
// covered (10466): 1
        return (GrB_INVALID_VALUE) ;
    }

    if (desc != NULL && desc->user_name != NULL)
    {   GB_cov[10467]++ ;
// NOT COVERED (10467):
        (*value) = desc->user_name_size ;
    }
    else
    {
        (*value) = GxB_MAX_NAME_LEN ;
    }
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Descriptor_get_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Descriptor_get_VOID
(
    GrB_Descriptor desc,
    void * value,
    GrB_Field field
)
{   GB_cov[10468]++ ;
// covered (10468): 2
    return (GrB_NOT_IMPLEMENTED) ;
}

