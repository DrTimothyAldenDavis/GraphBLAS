//------------------------------------------------------------------------------
// GraphBLAS/Config/GB_prejit.c: return list of PreJIT kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is configured by cmake from Config/GB_prejit.c.in.

#include "GB.h"
#include "GB_jitifyer.h"
#include "GB_AxB_saxpy3.h"

//------------------------------------------------------------------------------
// prototypes for all PreJIT kernels
//------------------------------------------------------------------------------

#define JIT_DOT2(g) GB_JIT_KERNEL_AXB_DOT2_PROTO(g) ;
#define JIT_DO2N(g) GB_JIT_KERNEL_AXB_DOT2N_PROTO(g) ;
#define JIT_DOT3(g) GB_JIT_KERNEL_AXB_DOT3_PROTO(g) ;
#define JIT_DOT4(g) GB_JIT_KERNEL_AXB_DOT4_PROTO(g) ;
#define JIT_SAXB(g) GB_JIT_KERNEL_AXB_SAXBIT_PROTO(g) ;
#define JIT_SAX3(g) GB_JIT_KERNEL_AXB_SAXPY3_PROTO(g) ;
#define JIT_SAX4(g) GB_JIT_KERNEL_AXB_SAXPY4_PROTO(g) ;
#define JIT_SAX5(g) GB_JIT_KERNEL_AXB_SAXPY5_PROTO(g) ;
#define JIT_ADD(g)  GB_JIT_KERNEL_ADD_PROTO(g) ;
#define JIT_AP1(g)  GB_JIT_KERNEL_APPLY_BIND1ST_PROTO(g) ;
#define JIT_AP2(g)  GB_JIT_KERNEL_APPLY_BIND2ND_PROTO(g) ;
#define JIT_AP0(g)  GB_JIT_KERNEL_APPLY_UNOP_PROTO(g) ;
#define JIT_BLD(g)  GB_JIT_KERNEL_BUILD_PROTO(g) ;
#define JIT_COLS(g) GB_JIT_KERNEL_COLSCALE_PROTO(g) ;
#define JIT_CONB(g) GB_JIT_KERNEL_CONCAT_BITMAP_PROTO(g) ;
#define JIT_CONF(g) GB_JIT_KERNEL_CONCAT_FULL_PROTO(g) ;
#define JIT_CONS(g) GB_JIT_KERNEL_CONCAT_SPARSE_PROTO(g) ;
#define JIT_CS2B(g) GB_JIT_KERNEL_CONVERT_S2B_PROTO(g) ;
#define JIT_EM2(g)  GB_JIT_KERNEL_EMULT_02_PROTO(g) ;
#define JIT_EM3(g)  GB_JIT_KERNEL_EMULT_03_PROTO(g) ;
#define JIT_EM4(g)  GB_JIT_KERNEL_EMULT_04_PROTO(g) ;
#define JIT_EM8(g)  GB_JIT_KERNEL_EMULT_08_PROTO(g) ;
#define JIT_EMB(g)  GB_JIT_KERNEL_EMULT_BITMAP_PROTO(g) ;
#define JIT_EWFA(g) GB_JIT_KERNEL_EWISE_FULLA_PROTO(g) ;
#define JIT_EWFN(g) GB_JIT_KERNEL_EWISE_FULLN_PROTO(g) ;
#define JIT_RED(g)  GB_JIT_KERNEL_REDUCE_PROTO(g) ;
#define JIT_ROWS(g) GB_JIT_KERNEL_ROWSCALE_PROTO(g) ;
#define JIT_SELB(g) GB_JIT_KERNEL_SELECT_BITMAP_PROTO(g) ;
#define JIT_SEL1(g) GB_JIT_KERNEL_SELECT_PHASE1_PROTO(g) ;
#define JIT_SEL2(g) GB_JIT_KERNEL_SELECT_PHASE2_PROTO(g) ;
#define JIT_SPB(g)  GB_JIT_KERNEL_SPLIT_BITMAP_PROTO(g) ;
#define JIT_SPF(g)  GB_JIT_KERNEL_SPLIT_FULL_PROTO(g) ;
#define JIT_SPS(g)  GB_JIT_KERNEL_SPLIT_SPARSE_PROTO(g) ;
#define JIT_SUB(g)  GB_JIT_KERNEL_SUBASSIGN_PROTO(g) ;
#define JIT_TR1(g)  GB_JIT_KERNEL_TRANS_BIND1ST_PROTO(g) ;
#define JIT_TR2(g)  GB_JIT_KERNEL_TRANS_BIND2ND_PROTO(g) ;
#define JIT_TR0(g)  GB_JIT_KERNEL_TRANS_UNOP_PROTO(g) ;
#define JIT_UNI(g)  GB_JIT_KERNEL_UNION_PROTO(g) ;



//------------------------------------------------------------------------------
// prototypes for all PreJIT query kernels
//------------------------------------------------------------------------------

#define JIT_Q(q) GB_JIT_QUERY_PROTO(q) ;



//------------------------------------------------------------------------------
// GB_prejit_kernels: a list of function pointers to PreJIT kernels
//------------------------------------------------------------------------------

#if ( 0 > 0 )
static void *GB_prejit_kernels [0] =
{

} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit_queries: a list of function pointers to PreJIT query kernels
//------------------------------------------------------------------------------

#if ( 0 > 0 )
static void *GB_prejit_queries [0] =
{

} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit_names: a list of names of PreJIT kernels
//------------------------------------------------------------------------------

#if ( 0 > 0 )
static char *GB_prejit_names [0] =
{
""
} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit: return list of PreJIT function pointers and function names
//------------------------------------------------------------------------------

void GB_prejit
(
    int32_t *nkernels,      // return # of kernels
    void **Kernel_handle,   // return list of function pointers to kernels
    void **Query_handle,    // return list of function pointers to queries
    char **Name_handle      // return list of kernel names
)
{
    (*nkernels) = 0 ;
    #if ( 0 == 0 )
    (*Kernel_handle) = NULL ;
    (*Query_handle) = NULL ;
    (*Name_handle) = NULL ;
    #else
    (*Kernel_handle) = GB_prejit_kernels ;
    (*Query_handle) = GB_prejit_queries ;
    (*Name_handle) = GB_prejit_names ;
    #endif
}

