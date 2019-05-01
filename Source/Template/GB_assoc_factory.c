//------------------------------------------------------------------------------
// GB_assoc_factory.c: switch factory for associative operators
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// TODO: add to GB_build, and GB_reduce_to_col to Generator/GB_red.[ch]

// This is a generic body of code for creating hard-coded versions of code for
// 44 combinations of associative operators and built-in types: 10 types (all
// but boolean) with MIN, MAX, PLUS, and TIMES, and one type (boolean) with
// OR, AND, XOR, and EQ

// If GB_INCLUDE_SECOND_OPERATOR is defined then an additional 11 built-in
// workers for the SECOND operator are also created, and 11 for FIRST, for
// GB_builder.

#undef  GB_IGNORE
#define GB_IGNORE 0

if (typecode != GB_BOOL_code)
{

    //--------------------------------------------------------------------------
    // non-boolean case
    //--------------------------------------------------------------------------

    switch (opcode)
    {

        // MAX and MIN monoids have terminal values
        #undef  GB_HAS_TERMINAL
        #define GB_HAS_TERMINAL true

        case GB_MIN_opcode   :

            switch (typecode)
            {
                // MIN terminal value is -infinity, or smallest possible int
                #define GB_DUP(w,t) w = GB_IMIN (w,t)
                case GB_INT8_code   : GB_ASSOC_WORKER (_min, _int8,   int8_t  , INT8_MIN  )
                case GB_INT16_code  : GB_ASSOC_WORKER (_min, _int16,  int16_t , INT16_MIN )
                case GB_INT32_code  : GB_ASSOC_WORKER (_min, _int32,  int32_t , INT32_MIN )
                case GB_INT64_code  : GB_ASSOC_WORKER (_min, _int64,  int64_t , INT64_MIN )
                case GB_UINT8_code  : GB_ASSOC_WORKER (_min, _uint8,  uint8_t , 0         )
                case GB_UINT16_code : GB_ASSOC_WORKER (_min, _uint16, uint16_t, 0         )
                case GB_UINT32_code : GB_ASSOC_WORKER (_min, _uint32, uint32_t, 0         )
                case GB_UINT64_code : GB_ASSOC_WORKER (_min, _uint64, uint64_t, 0         )
                #undef  GB_DUP
                #define GB_DUP(w,t) w = fminf (w,t)
                case GB_FP32_code   : GB_ASSOC_WORKER (_min, _fp32, float   , -INFINITY )
                #undef  GB_DUP
                #define GB_DUP(w,t) w = fmin (w,t)
                case GB_FP64_code   : GB_ASSOC_WORKER (_min, _fp64, double  , -INFINITY )
                #undef  GB_DUP
                default: ;
            }
            break ;

        case GB_MAX_opcode   :

            switch (typecode)
            {
                // MAX terminal value is +infinity, or largest possible int
                #define GB_DUP(w,t) w = GB_IMAX (w,t)
                case GB_INT8_code   : GB_ASSOC_WORKER (_max, _int8,   int8_t  , INT8_MAX  )
                case GB_INT16_code  : GB_ASSOC_WORKER (_max, _int16,  int16_t , INT16_MAX )
                case GB_INT32_code  : GB_ASSOC_WORKER (_max, _int32,  int32_t , INT32_MAX )
                case GB_INT64_code  : GB_ASSOC_WORKER (_max, _int64,  int64_t , INT64_MAX )
                case GB_UINT8_code  : GB_ASSOC_WORKER (_max, _uint8,  uint8_t , UINT8_MAX )
                case GB_UINT16_code : GB_ASSOC_WORKER (_max, _uint16, uint16_t, UINT16_MAX)
                case GB_UINT32_code : GB_ASSOC_WORKER (_max, _uint32, uint32_t, UINT32_MAX)
                case GB_UINT64_code : GB_ASSOC_WORKER (_max, _uint64, uint64_t, UINT64_MAX)
                #undef  GB_DUP
                #define GB_DUP(w,t) w = fmaxf (w,t)
                case GB_FP32_code   : GB_ASSOC_WORKER (_max, _fp32, float   , INFINITY  )
                #undef  GB_DUP
                #define GB_DUP(w,t) w = fmax (w,t)
                case GB_FP64_code   : GB_ASSOC_WORKER (_max, _fp64, double  , INFINITY  )
                #undef  GB_DUP
                default: ;
            }
            break ;

        // PLUS monoids are not terminal
        #undef  GB_HAS_TERMINAL
        #define GB_HAS_TERMINAL false

        case GB_PLUS_opcode  :

            #define GB_DUP(w,t) w += t
            switch (typecode)
            {
                // no terminal value
                case GB_INT8_code   : GB_ASSOC_WORKER (_plus, _int8,   int8_t  , GB_IGNORE)
                case GB_INT16_code  : GB_ASSOC_WORKER (_plus, _int16,  int16_t , GB_IGNORE)
                case GB_INT32_code  : GB_ASSOC_WORKER (_plus, _int32,  int32_t , GB_IGNORE)
                case GB_INT64_code  : GB_ASSOC_WORKER (_plus, _int64,  int64_t , GB_IGNORE)
                case GB_UINT8_code  : GB_ASSOC_WORKER (_plus, _uint8,  uint8_t , GB_IGNORE)
                case GB_UINT16_code : GB_ASSOC_WORKER (_plus, _uint16, uint16_t, GB_IGNORE)
                case GB_UINT32_code : GB_ASSOC_WORKER (_plus, _uint32, uint32_t, GB_IGNORE)
                case GB_UINT64_code : GB_ASSOC_WORKER (_plus, _uint64, uint64_t, GB_IGNORE)
                case GB_FP32_code   : GB_ASSOC_WORKER (_plus, _fp32,   float   , GB_IGNORE)
                case GB_FP64_code   : GB_ASSOC_WORKER (_plus, _fp64,   double  , GB_IGNORE)
                default: ;
            }
            break ;
            #undef  GB_DUP

        case GB_TIMES_opcode :

            #define GB_DUP(w,t) w *= t
            switch (typecode)
            {

                // integer TIMES monoids have terminal values of zero
                #undef  GB_HAS_TERMINAL
                #define GB_HAS_TERMINAL true

                // terminal value is zero
                case GB_INT8_code   : GB_ASSOC_WORKER (_times, _int8,   int8_t  , 0)
                case GB_INT16_code  : GB_ASSOC_WORKER (_times, _int16,  int16_t , 0)
                case GB_INT32_code  : GB_ASSOC_WORKER (_times, _int32,  int32_t , 0)
                case GB_INT64_code  : GB_ASSOC_WORKER (_times, _int64,  int64_t , 0)
                case GB_UINT8_code  : GB_ASSOC_WORKER (_times, _uint8,  uint8_t , 0)
                case GB_UINT16_code : GB_ASSOC_WORKER (_times, _uint16, uint16_t, 0)
                case GB_UINT32_code : GB_ASSOC_WORKER (_times, _uint32, uint32_t, 0)
                case GB_UINT64_code : GB_ASSOC_WORKER (_times, _uint64, uint64_t, 0)

                // floating-point TIMES monoids are not terminal
                #undef  GB_HAS_TERMINAL
                #define GB_HAS_TERMINAL false

                // no terminal value
                case GB_FP32_code   : GB_ASSOC_WORKER (_times, _fp32, float   , GB_IGNORE)
                case GB_FP64_code   : GB_ASSOC_WORKER (_times, _fp64, double  , GB_IGNORE)
                default: ;
            }
            break ;
            #undef  GB_DUP

        //----------------------------------------------------------------------
        // FIRST and SECOND for GB_builder
        //----------------------------------------------------------------------

        // GB_build does not terminate early
        #undef  GB_HAS_TERMINAL
        #define GB_HAS_TERMINAL false

        #ifdef GB_INCLUDE_SECOND_OPERATOR

        case GB_FIRST_opcode :

            #define GB_DUP(w,t) ;      // do nothing; keep the first tuple
            switch (typecode)
            {
                // no terminal value exploited for GB_build
                case GB_INT8_code   : GB_ASSOC_WORKER (_first, _int8,   int8_t  , GB_IGNORE)
                case GB_INT16_code  : GB_ASSOC_WORKER (_first, _int16,  int16_t , GB_IGNORE)
                case GB_INT32_code  : GB_ASSOC_WORKER (_first, _int32,  int32_t , GB_IGNORE)
                case GB_INT64_code  : GB_ASSOC_WORKER (_first, _int64,  int64_t , GB_IGNORE)
                case GB_UINT8_code  : GB_ASSOC_WORKER (_first, _uint8,  uint8_t , GB_IGNORE)
                case GB_UINT16_code : GB_ASSOC_WORKER (_first, _uint16, uint16_t, GB_IGNORE)
                case GB_UINT32_code : GB_ASSOC_WORKER (_first, _uint32, uint32_t, GB_IGNORE)
                case GB_UINT64_code : GB_ASSOC_WORKER (_first, _uint64, uint64_t, GB_IGNORE)
                case GB_FP32_code   : GB_ASSOC_WORKER (_first, _fp32,   float   , GB_IGNORE)
                case GB_FP64_code   : GB_ASSOC_WORKER (_first, _fp64,   double  , GB_IGNORE)
                default: ;
            }
            break ;
            #undef  GB_DUP

        case GB_SECOND_opcode :

            #define GB_DUP(w,t) w = t  // replace with the 2nd tuple
            switch (typecode)
            {
                // no terminal value exploited for GB_build
                case GB_INT8_code   : GB_ASSOC_WORKER (_second, _int8,   int8_t  , GB_IGNORE)
                case GB_INT16_code  : GB_ASSOC_WORKER (_second, _int16,  int16_t , GB_IGNORE)
                case GB_INT32_code  : GB_ASSOC_WORKER (_second, _int32,  int32_t , GB_IGNORE)
                case GB_INT64_code  : GB_ASSOC_WORKER (_second, _int64,  int64_t , GB_IGNORE)
                case GB_UINT8_code  : GB_ASSOC_WORKER (_second, _uint8,  uint8_t , GB_IGNORE)
                case GB_UINT16_code : GB_ASSOC_WORKER (_second, _uint16, uint16_t, GB_IGNORE)
                case GB_UINT32_code : GB_ASSOC_WORKER (_second, _uint32, uint32_t, GB_IGNORE)
                case GB_UINT64_code : GB_ASSOC_WORKER (_second, _uint64, uint64_t, GB_IGNORE)
                case GB_FP32_code   : GB_ASSOC_WORKER (_second, _fp32,   float   , GB_IGNORE)
                case GB_FP64_code   : GB_ASSOC_WORKER (_second, _fp64,   double  , GB_IGNORE)
                default: ;
            }
            break ;
            #undef  GB_DUP

        #endif

        default: ;
    }

}
else
{

    //--------------------------------------------------------------------------
    // boolean case: rename the opcode as needed
    //--------------------------------------------------------------------------

    // The FIRST and SECOND operators are not associative, but are added for
    // GB_builder.

    switch (GB_boolean_rename (opcode))
    {

        // LOR and LAND monoids have terminal values
        #undef  GB_HAS_TERMINAL
        #define GB_HAS_TERMINAL true

        case GB_LOR_opcode : 

            // OR == MAX == PLUS
            // terminal value is true
            #define GB_DUP(w,t) w = (w || t)
            GB_ASSOC_WORKER (_lor, _bool, bool, true)
            #undef  GB_DUP

        case GB_LAND_opcode : 

            // AND == MIN == TIMES
            // terminal value is false
            #define GB_DUP(w,t) w = (w && t)
            GB_ASSOC_WORKER (_land, _bool, bool, false)
            #undef  GB_DUP

        // LXOR and EQ monoids do not have terminal values
        #undef  GB_HAS_TERMINAL
        #define GB_HAS_TERMINAL false

        case GB_LXOR_opcode : 

            // XOR == NE == MINUS == RMINUS == ISNE
            // no terminal value
            #define GB_DUP(w,t) w = (w != t)
            GB_ASSOC_WORKER (_lxor, _bool, bool, GB_IGNORE)
            #undef  GB_DUP

        case GB_EQ_opcode : 

            // EQ == ISEQ
            // no terminal value
            #define GB_DUP(w,t) w = (w == t)
            GB_ASSOC_WORKER (_eq, _bool, bool, GB_IGNORE)
            #undef  GB_DUP

        //----------------------------------------------------------------------
        // FIRST and SECOND for GB_builder
        //----------------------------------------------------------------------

        // GB_build does not terminate early

        #ifdef GB_INCLUDE_SECOND_OPERATOR

        case GB_FIRST_opcode : 

            // FIRST == DIV
            // no terminal value exploited
            #define GB_DUP(w,t) ;      // do nothing; keep the first tuple
            GB_ASSOC_WORKER (_first, _bool, bool, GB_IGNORE)
            #undef  GB_DUP

        case GB_SECOND_opcode : 

            // SECOND == RDIV
            // no terminal value exploited
            #define GB_DUP(w,t) w = t  // replace with the 2nd tuple
            GB_ASSOC_WORKER (_second, _bool, bool, GB_IGNORE)
            #undef  GB_DUP

        #endif

        default: ;
    }
}

