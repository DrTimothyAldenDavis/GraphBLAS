//------------------------------------------------------------------------------
// GB_select_factory: switch factory for C=select(A,thunk)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

switch (opcode)
{

    case GB_TRIL_idxunop_code          :  // C = tril (A,k)

        #ifdef GB_SELECT_PHASE1
        GB_SEL_WORKER (_tril, _iso, GB_void)
        #else
        switch (typecode)
        {
            case GB_ignore_code  : GB_SEL_WORKER (_tril, _iso, GB_void)
            default              : GB_SEL_WORKER (_tril, _any, GB_void)
        }
        break ;
        #endif

    case GB_TRIU_idxunop_code          :  // C = triu (A,k)

        #ifdef GB_SELECT_PHASE1
        GB_SEL_WORKER (_triu, _iso, GB_void)
        #else
        switch (typecode)
        {
            case GB_ignore_code  : GB_SEL_WORKER (_triu, _iso, GB_void)
            default              : GB_SEL_WORKER (_triu, _any, GB_void)
        }
        break ;
        #endif

    case GB_DIAG_idxunop_code          :  // C = diag (A,k)

        #ifdef GB_SELECT_PHASE1
        GB_SEL_WORKER (_diag, _iso, GB_void)
        #else
        switch (typecode)
        {
            case GB_ignore_code  : GB_SEL_WORKER (_diag, _iso, GB_void)
            default              : GB_SEL_WORKER (_diag, _any, GB_void)
        }
        break ;
        #endif

    case GB_OFFDIAG_idxunop_code       :  // C = offdiag (A,k)

        #ifdef GB_SELECT_PHASE1
        GB_SEL_WORKER (_offdiag, _iso, GB_void)
        #else
        switch (typecode)
        {
            case GB_ignore_code  : GB_SEL_WORKER (_offdiag, _iso, GB_void)
            default              : GB_SEL_WORKER (_offdiag, _any, GB_void)
        }
        break ;
        #endif

    case GB_ROWINDEX_idxunop_code     :  // C = rowindex (A,k)

        #ifdef GB_SELECT_PHASE1
        GB_SEL_WORKER (_rowindex, _iso, GB_void)
        #else
        switch (typecode)
        {
            case GB_ignore_code  : GB_SEL_WORKER (_rowindex, _iso, GB_void)
            default              : GB_SEL_WORKER (_rowindex, _any, GB_void)
        }
        break ;
        #endif

    case GB_ROWLE_idxunop_code     :  // C = rowle (A,k)

        // also used for resize
        #ifdef GB_SELECT_PHASE1
        GB_SEL_WORKER (_rowle, _iso, GB_void)
        #else
        switch (typecode)
        {
            case GB_ignore_code  : GB_SEL_WORKER (_rowle, _iso, GB_void)
            default              : GB_SEL_WORKER (_rowle, _any, GB_void)
        }
        break ;
        #endif

    case GB_ROWGT_idxunop_code     :  // C = rowgt (A,k)

        #ifdef GB_SELECT_PHASE1
        GB_SEL_WORKER (_rowgt, _iso, GB_void)
        #else
        switch (typecode)
        {
            case GB_ignore_code  : GB_SEL_WORKER (_rowgt, _iso, GB_void)
            default              : GB_SEL_WORKER (_rowgt, _any, GB_void)
        }
        break ;
        #endif

    case GB_USER_idxunop_code   : // C = user_idxunop (A,k)

        //----------------------------------------------------------------------
        // via the JIT kernel
        //----------------------------------------------------------------------

        #if GB_JIT_ENABLED
        // JIT TODO: idxunop: select with idxunop
        #endif

        ASSERT (op != NULL) ;
        ASSERT (op->ztype != NULL) ;
        ASSERT (op->xtype != NULL) ;
        ASSERT (op->ytype != NULL) ;
        if ((op->ztype != GrB_BOOL) ||
           ((typecode != GB_ignore_code) && (op->xtype != A->type)))
        {
            // typecasting is required
            #ifdef GB_SELECT_PHASE1
            GBURBLE ("(generic select) ") ;
            #endif
            switch (typecode)
            {
                case GB_ignore_code :   // A is iso
                    GB_SEL_WORKER (_idxunop_cast, _iso, GB_void)
                default             :   // A is non-iso
                    GB_SEL_WORKER (_idxunop_cast, _any, GB_void)
            }
        }
        else
        {
            // no typecasting
            switch (typecode)
            {
                case GB_ignore_code : GB_SEL_WORKER (_idxunop, _iso, GB_void)
                default             : GB_SEL_WORKER (_idxunop, _any, GB_void)
            }
        }
        break ;

    //--------------------------------------------------------------------------
    // COL selectors are used only for the bitmap case
    //--------------------------------------------------------------------------

    #ifdef GB_BITMAP_SELECTOR

    case GB_COLINDEX_idxunop_code     :  // C = colindex (A,k)

        switch (typecode)
        {
            case GB_ignore_code  : GB_SEL_WORKER (_colindex, _iso, GB_void)
            default              : GB_SEL_WORKER (_colindex, _any, GB_void)
        }
        break ;

    case GB_COLLE_idxunop_code     :  // C = colle (A,k)

        switch (typecode)
        {
            case GB_ignore_code  : GB_SEL_WORKER (_colle, _iso, GB_void)
            default              : GB_SEL_WORKER (_colle, _any, GB_void)
        }
        break ;

    case GB_COLGT_idxunop_code     :  // C = colgt (A,k)

        switch (typecode)
        {
            case GB_ignore_code  : GB_SEL_WORKER (_colgt, _iso, GB_void)
            default              : GB_SEL_WORKER (_colgt, _any, GB_void)
        }
        break ;

    #endif

    //--------------------------------------------------------------------------
    // nonzombie selectors are not used for the bitmap case
    //--------------------------------------------------------------------------

    #ifndef GB_BITMAP_SELECTOR

    case GB_NONZOMBIE_idxunop_code     :  // C = all entries A(i,j) not a zombie

        #ifdef GB_SELECT_PHASE1
        // phase1: use a single worker for all types, since the test does not
        // depend on the values, just Ai.
        GB_SEL_WORKER (_nonzombie, _iso, GB_void)
        #else
        // phase2:
        switch (typecode)
        {
            case GB_BOOL_code   : GB_SEL_WORKER (_nonzombie, _bool  , bool    )
            case GB_INT8_code   : GB_SEL_WORKER (_nonzombie, _int8  , int8_t  )
            case GB_INT16_code  : GB_SEL_WORKER (_nonzombie, _int16 , int16_t )
            case GB_INT32_code  : GB_SEL_WORKER (_nonzombie, _int32 , int32_t )
            case GB_INT64_code  : GB_SEL_WORKER (_nonzombie, _int64 , int64_t )
            case GB_UINT8_code  : GB_SEL_WORKER (_nonzombie, _uint8 , uint8_t )
            case GB_UINT16_code : GB_SEL_WORKER (_nonzombie, _uint16, uint16_t)
            case GB_UINT32_code : GB_SEL_WORKER (_nonzombie, _uint32, uint32_t)
            case GB_UINT64_code : GB_SEL_WORKER (_nonzombie, _uint64, uint64_t)
            case GB_FP32_code   : GB_SEL_WORKER (_nonzombie, _fp32  , float   )
            case GB_FP64_code   : GB_SEL_WORKER (_nonzombie, _fp64  , double  )
            case GB_FC32_code   : GB_SEL_WORKER (_nonzombie, _fc32, GxB_FC32_t)
            case GB_FC64_code   : GB_SEL_WORKER (_nonzombie, _fc64, GxB_FC64_t)
            case GB_UDT_code    : GB_SEL_WORKER (_nonzombie, _any   , GB_void )
            case GB_ignore_code : GB_SEL_WORKER (_nonzombie, _iso   , GB_void )
            default: ;          // not used
        }
        break ;
        #endif

    #endif

    //--------------------------------------------------------------------------
    // none of these selectop workers are needed when A is iso
    //--------------------------------------------------------------------------

    case GB_VALUEEQ_idxunop_code : // A(i,j) == thunk

        switch (typecode)
        {
            case GB_BOOL_code   : GB_SEL_WORKER (_eq_thunk, _bool  , bool    )
            case GB_INT8_code   : GB_SEL_WORKER (_eq_thunk, _int8  , int8_t  )
            case GB_INT16_code  : GB_SEL_WORKER (_eq_thunk, _int16 , int16_t )
            case GB_INT32_code  : GB_SEL_WORKER (_eq_thunk, _int32 , int32_t )
            case GB_INT64_code  : GB_SEL_WORKER (_eq_thunk, _int64 , int64_t )
            case GB_UINT8_code  : GB_SEL_WORKER (_eq_thunk, _uint8 , uint8_t )
            case GB_UINT16_code : GB_SEL_WORKER (_eq_thunk, _uint16, uint16_t)
            case GB_UINT32_code : GB_SEL_WORKER (_eq_thunk, _uint32, uint32_t)
            case GB_UINT64_code : GB_SEL_WORKER (_eq_thunk, _uint64, uint64_t)
            case GB_FP32_code   : GB_SEL_WORKER (_eq_thunk, _fp32  , float   )
            case GB_FP64_code   : GB_SEL_WORKER (_eq_thunk, _fp64  , double  )
            case GB_FC32_code   : GB_SEL_WORKER (_eq_thunk, _fc32, GxB_FC32_t)
            case GB_FC64_code   : GB_SEL_WORKER (_eq_thunk, _fc64, GxB_FC64_t)
            default: ;          // not used
        }
        break ;

    case GB_VALUENE_idxunop_code : // A(i,j) != thunk

        switch (typecode)
        {
            case GB_INT8_code   : GB_SEL_WORKER (_ne_thunk, _int8  , int8_t  )
            case GB_INT16_code  : GB_SEL_WORKER (_ne_thunk, _int16 , int16_t )
            case GB_INT32_code  : GB_SEL_WORKER (_ne_thunk, _int32 , int32_t )
            case GB_INT64_code  : GB_SEL_WORKER (_ne_thunk, _int64 , int64_t )
            case GB_UINT8_code  : GB_SEL_WORKER (_ne_thunk, _uint8 , uint8_t )
            case GB_UINT16_code : GB_SEL_WORKER (_ne_thunk, _uint16, uint16_t)
            case GB_UINT32_code : GB_SEL_WORKER (_ne_thunk, _uint32, uint32_t)
            case GB_UINT64_code : GB_SEL_WORKER (_ne_thunk, _uint64, uint64_t)
            case GB_FP32_code   : GB_SEL_WORKER (_ne_thunk, _fp32  , float   )
            case GB_FP64_code   : GB_SEL_WORKER (_ne_thunk, _fp64  , double  )
            case GB_FC32_code   : GB_SEL_WORKER (_ne_thunk, _fc32, GxB_FC32_t)
            case GB_FC64_code   : GB_SEL_WORKER (_ne_thunk, _fc64, GxB_FC64_t)
            default: ;          // not used
        }
        break ;

    case GB_VALUEGT_idxunop_code : // A(i,j) > thunk

        switch (typecode)
        {
            case GB_INT8_code   : GB_SEL_WORKER (_gt_thunk, _int8  , int8_t  )
            case GB_INT16_code  : GB_SEL_WORKER (_gt_thunk, _int16 , int16_t )
            case GB_INT32_code  : GB_SEL_WORKER (_gt_thunk, _int32 , int32_t )
            case GB_INT64_code  : GB_SEL_WORKER (_gt_thunk, _int64 , int64_t )
            case GB_UINT8_code  : GB_SEL_WORKER (_gt_thunk, _uint8 , uint8_t )
            case GB_UINT16_code : GB_SEL_WORKER (_gt_thunk, _uint16, uint16_t)
            case GB_UINT32_code : GB_SEL_WORKER (_gt_thunk, _uint32, uint32_t)
            case GB_UINT64_code : GB_SEL_WORKER (_gt_thunk, _uint64, uint64_t)
            case GB_FP32_code   : GB_SEL_WORKER (_gt_thunk, _fp32  , float   )
            case GB_FP64_code   : GB_SEL_WORKER (_gt_thunk, _fp64  , double  )
            default: ;          // not used
        }
        break ;

    case GB_VALUEGE_idxunop_code : // A(i,j) >= thunk

        switch (typecode)
        {
            case GB_INT8_code   : GB_SEL_WORKER (_ge_thunk, _int8  , int8_t  )
            case GB_INT16_code  : GB_SEL_WORKER (_ge_thunk, _int16 , int16_t )
            case GB_INT32_code  : GB_SEL_WORKER (_ge_thunk, _int32 , int32_t )
            case GB_INT64_code  : GB_SEL_WORKER (_ge_thunk, _int64 , int64_t )
            case GB_UINT8_code  : GB_SEL_WORKER (_ge_thunk, _uint8 , uint8_t )
            case GB_UINT16_code : GB_SEL_WORKER (_ge_thunk, _uint16, uint16_t)
            case GB_UINT32_code : GB_SEL_WORKER (_ge_thunk, _uint32, uint32_t)
            case GB_UINT64_code : GB_SEL_WORKER (_ge_thunk, _uint64, uint64_t)
            case GB_FP32_code   : GB_SEL_WORKER (_ge_thunk, _fp32  , float   )
            case GB_FP64_code   : GB_SEL_WORKER (_ge_thunk, _fp64  , double  )
            default: ;          // not used
        }
        break ;

    case GB_VALUELT_idxunop_code : // A(i,j) < thunk

        switch (typecode)
        {
            case GB_INT8_code   : GB_SEL_WORKER (_lt_thunk, _int8  , int8_t  )
            case GB_INT16_code  : GB_SEL_WORKER (_lt_thunk, _int16 , int16_t )
            case GB_INT32_code  : GB_SEL_WORKER (_lt_thunk, _int32 , int32_t )
            case GB_INT64_code  : GB_SEL_WORKER (_lt_thunk, _int64 , int64_t )
            case GB_UINT8_code  : GB_SEL_WORKER (_lt_thunk, _uint8 , uint8_t )
            case GB_UINT16_code : GB_SEL_WORKER (_lt_thunk, _uint16, uint16_t)
            case GB_UINT32_code : GB_SEL_WORKER (_lt_thunk, _uint32, uint32_t)
            case GB_UINT64_code : GB_SEL_WORKER (_lt_thunk, _uint64, uint64_t)
            case GB_FP32_code   : GB_SEL_WORKER (_lt_thunk, _fp32  , float   )
            case GB_FP64_code   : GB_SEL_WORKER (_lt_thunk, _fp64  , double  )
            default: ;          // not used
        }
        break ;

    case GB_VALUELE_idxunop_code : // A(i,j) <= thunk

        switch (typecode)
        {
            case GB_INT8_code   : GB_SEL_WORKER (_le_thunk, _int8  , int8_t  )
            case GB_INT16_code  : GB_SEL_WORKER (_le_thunk, _int16 , int16_t )
            case GB_INT32_code  : GB_SEL_WORKER (_le_thunk, _int32 , int32_t )
            case GB_INT64_code  : GB_SEL_WORKER (_le_thunk, _int64 , int64_t )
            case GB_UINT8_code  : GB_SEL_WORKER (_le_thunk, _uint8 , uint8_t )
            case GB_UINT16_code : GB_SEL_WORKER (_le_thunk, _uint16, uint16_t)
            case GB_UINT32_code : GB_SEL_WORKER (_le_thunk, _uint32, uint32_t)
            case GB_UINT64_code : GB_SEL_WORKER (_le_thunk, _uint64, uint64_t)
            case GB_FP32_code   : GB_SEL_WORKER (_le_thunk, _fp32  , float   )
            case GB_FP64_code   : GB_SEL_WORKER (_le_thunk, _fp64  , double  )
            default: ;          // not used
        }
        break ;

    default: ;
}

