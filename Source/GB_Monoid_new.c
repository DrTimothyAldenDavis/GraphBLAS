//------------------------------------------------------------------------------
// GB_Monoid_new: create a Monoid with a specific type of identity
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// not parallel: this function does O(1) work and is already thread-safe.

#include "GB.h"

GrB_Info GB_Monoid_new          // create a monoid
(
    GrB_Monoid *monoid,         // handle of monoid to create
    const GrB_BinaryOp op,      // binary operator of the monoid
    const void *identity,       // identity value
    const GB_Type_code idcode,  // identity code
    GB_Context Context
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (monoid) ;
    (*monoid) = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (op) ;
    GB_RETURN_IF_NULL (identity) ;

    ASSERT_OK (GB_check (op, "op for monoid", GB0)) ;
    ASSERT (idcode <= GB_UDT_code) ;
    ASSERT (idcode != GB_UCT_code) ;

    // check operator types; all must be identical
    if (op->xtype != op->ztype || op->ytype != op->ztype)
    { 
        return (GB_ERROR (GrB_DOMAIN_MISMATCH, (GB_LOG,
            "All domains of monoid operator must be identical;\n"
            "operator is: [%s] = %s ([%s],[%s])",
            op->ztype->name, op->name, op->xtype->name, op->ytype->name))) ;
    }

    // The idcode must match the monoid->op->ztype->code for built-in types,
    // and this can be rigourously checked.  For all user-defined types,
    // identity is a mere void * pointer, and its actual type cannot be
    // compared with the input op->ztype parameter.  Only the type code,
    // GB_UDT_code or GB_UCT_code, can be checked to see if it matches.  In
    // that case, all that is known is that identity is a void * pointer that
    // points to something, hopefully a scalar of the proper user-defined type.

    // UCT code is treated as UDT, since GB_Monoid_new is never called with
    // an idcode of UCT.
    GB_Type_code zcode = op->ztype->code ;
    if (zcode == GB_UCT_code) zcode = GB_UDT_code ;

    if (idcode != zcode)
    { 
        return (GB_ERROR (GrB_DOMAIN_MISMATCH, (GB_LOG,
            "Identity type [%s]\n"
            "must be identical to monoid operator z=%s(x,y) of type [%s]",
            GB_code_string (idcode), op->name, op->ztype->name))) ;
    }

    //--------------------------------------------------------------------------
    // create the monoid
    //--------------------------------------------------------------------------

    // allocate the monoid
    GB_CALLOC_MEMORY (*monoid, 1, sizeof (struct GB_Monoid_opaque), NULL) ;
    if (*monoid == NULL)
    { 
        // out of memory
        return (GB_OUT_OF_MEMORY) ;
    }

    // initialize the monoid
    GrB_Monoid mon = *monoid ;
    mon->magic = GB_MAGIC ;
    mon->op = op ;
    mon->object_kind = GB_USER_RUNTIME ;
    size_t zsize = op->ztype->size ;
    mon->op_ztype_size = zsize ;
    GB_CALLOC_MEMORY (mon->identity, 1, zsize, NULL) ;
    if (mon->identity == NULL)
    { 
        // out of memory
        GB_FREE_MEMORY (*monoid, 1, sizeof (struct GB_Monoid_opaque)) ;
        return (GB_OUT_OF_MEMORY) ;
    }

    // copy the identity into the monoid.  No typecasting needed.
    memcpy (mon->identity, identity, zsize) ;

    //--------------------------------------------------------------------------
    // set the terminal value
    //--------------------------------------------------------------------------

    // FUTURE:: allow user-defined monoids based on user-defined ops to terminal
    // FUTURE:: move this switch into its own routine, so it can be used
    // by GB_reduce.

    mon->terminal = NULL ;

    #define SET_TERMINAL(ctype,value)                                       \
    {                                                                       \
        GB_CALLOC_MEMORY (mon->terminal, 1, zsize, NULL) ;                  \
        if (mon->terminal == NULL)                                          \
        {                                                                   \
            /* out of memory */                                             \
            GB_FREE_MEMORY (mon->identity, 1, zsize) ;                      \
            GB_FREE_MEMORY (*monoid, 1, sizeof (struct GB_Monoid_opaque)) ; \
            return (GB_OUT_OF_MEMORY) ;                                     \
        }                                                                   \
        ctype *terminal = mon->terminal ;                                   \
        (*terminal) = value ;                                               \
    }                                                                       \
    break ;

    // set the terminal value for built-in operators
    switch (op->opcode)
    {
        case GB_MIN_opcode :

            switch (zcode)
            {
                case GB_INT8_code   : SET_TERMINAL (int8_t  , INT8_MIN  )
                case GB_INT16_code  : SET_TERMINAL (int16_t , INT16_MIN )
                case GB_INT32_code  : SET_TERMINAL (int32_t , INT32_MIN )
                case GB_INT64_code  : SET_TERMINAL (int64_t , INT64_MIN )
                case GB_UINT8_code  : SET_TERMINAL (uint8_t , 0         )
                case GB_UINT16_code : SET_TERMINAL (uint16_t, 0         )
                case GB_UINT32_code : SET_TERMINAL (uint32_t, 0         )
                case GB_UINT64_code : SET_TERMINAL (uint64_t, 0         )
                case GB_FP32_code   : SET_TERMINAL (float   , -INFINITY )
                case GB_FP64_code   : SET_TERMINAL (double  , -INFINITY )
                default : ;
            }
            break ;

        case GB_MAX_opcode :

            switch (zcode)
            {
                case GB_INT8_code   : SET_TERMINAL (int8_t  , INT8_MAX  )
                case GB_INT16_code  : SET_TERMINAL (int16_t , INT16_MAX )
                case GB_INT32_code  : SET_TERMINAL (int32_t , INT32_MAX )
                case GB_INT64_code  : SET_TERMINAL (int64_t , INT64_MAX )
                case GB_UINT8_code  : SET_TERMINAL (uint8_t , UINT8_MAX )
                case GB_UINT16_code : SET_TERMINAL (uint16_t, UINT16_MAX)
                case GB_UINT32_code : SET_TERMINAL (uint32_t, UINT32_MAX)
                case GB_UINT64_code : SET_TERMINAL (uint64_t, UINT64_MAX)
                case GB_FP32_code   : SET_TERMINAL (float   , INFINITY  )
                case GB_FP64_code   : SET_TERMINAL (double  , INFINITY  )
                default : ;
            }
            break ;

        case GB_LOR_opcode :

            if (zcode == GB_BOOL_code) SET_TERMINAL (bool, true)

        case GB_LAND_opcode :

            if (zcode == GB_BOOL_code) SET_TERMINAL (bool, false)

        default :
            ;
    }

    ASSERT_OK (GB_check (mon, "new monoid", GB0)) ;
    return (GrB_SUCCESS) ;
}

