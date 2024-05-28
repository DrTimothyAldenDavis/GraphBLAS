//------------------------------------------------------------------------------
// GraphBLAS/Config/GB_prejit.c: return list of PreJIT kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is configured by cmake from Config/GB_prejit.c.in, which has
// indexed the following 20 kernels in GraphBLAS/PreJIT:

#include "GB.h"
#include "jitifyer/GB_jitifyer.h"
#include "jit_kernels/include/GB_jit_kernel_proto.h"

//------------------------------------------------------------------------------
// prototypes for all PreJIT kernels
//------------------------------------------------------------------------------

JIT_DOT4 (GB_jit__AxB_dot4__03fe900eee0eeec5__addgauss_multgauss)
JIT_SAXB (GB_jit__AxB_saxbit__03fe100eee0eee8a__addgauss_multgauss)
JIT_SAXB (GB_jit__AxB_saxbit__03fe100eee0eee8a__mycx_plus_mycx_times)
JIT_SAX3 (GB_jit__AxB_saxpy3__03fe100eee0eee45__wildadd_wildmult)
JIT_SAX4 (GB_jit__AxB_saxpy4__03fe300eee0eeec7__addgauss_multgauss)
JIT_SAX5 (GB_jit__AxB_saxpy5__03fe500eee0eeecd__addgauss_multgauss)
JIT_ADD  (GB_jit__add__1002e0e0eee00__wildtype)
JIT_AP0  (GB_jit__apply_unop__0000db0dbe__fx64_cmplx_imag)
JIT_AP0  (GB_jit__apply_unop__0000db0dbe__fx64_cmplx_real)
JIT_BLD  (GB_jit__build__02eeeee__wildtype)
JIT_SUB  (GB_jit__subassign_22__1000eee0eec8__addgauss)
JIT_SUB  (GB_jit__subassign_22__1000eee0eec8__multgauss)
JIT_TR1  (GB_jit__trans_bind1st__0000eee0efe41__multgauss)
JIT_TR2  (GB_jit__trans_bind2nd__0000eee0eef44__multgauss)
JIT_TR0  (GB_jit__trans_unop__08006e06ef__realgauss)
JIT_TR0  (GB_jit__trans_unop__0802ee0ee5__gauss)
JIT_TR0  (GB_jit__trans_unop__0802ee0ee5__wildtype)
JIT_TR0  (GB_jit__trans_unop__0802ee0eef__gauss)
JIT_UOP  (GB_jit__user_op__0__addgauss)
JIT_UTYP (GB_jit__user_type__0__gauss)


//------------------------------------------------------------------------------
// prototypes for all PreJIT query kernels
//------------------------------------------------------------------------------

JIT_Q (GB_jit__AxB_dot4__03fe900eee0eeec5__addgauss_multgauss_query)
JIT_Q (GB_jit__AxB_saxbit__03fe100eee0eee8a__addgauss_multgauss_query)
JIT_Q (GB_jit__AxB_saxbit__03fe100eee0eee8a__mycx_plus_mycx_times_query)
JIT_Q (GB_jit__AxB_saxpy3__03fe100eee0eee45__wildadd_wildmult_query)
JIT_Q (GB_jit__AxB_saxpy4__03fe300eee0eeec7__addgauss_multgauss_query)
JIT_Q (GB_jit__AxB_saxpy5__03fe500eee0eeecd__addgauss_multgauss_query)
JIT_Q (GB_jit__add__1002e0e0eee00__wildtype_query)
JIT_Q (GB_jit__apply_unop__0000db0dbe__fx64_cmplx_imag_query)
JIT_Q (GB_jit__apply_unop__0000db0dbe__fx64_cmplx_real_query)
JIT_Q (GB_jit__build__02eeeee__wildtype_query)
JIT_Q (GB_jit__subassign_22__1000eee0eec8__addgauss_query)
JIT_Q (GB_jit__subassign_22__1000eee0eec8__multgauss_query)
JIT_Q (GB_jit__trans_bind1st__0000eee0efe41__multgauss_query)
JIT_Q (GB_jit__trans_bind2nd__0000eee0eef44__multgauss_query)
JIT_Q (GB_jit__trans_unop__08006e06ef__realgauss_query)
JIT_Q (GB_jit__trans_unop__0802ee0ee5__gauss_query)
JIT_Q (GB_jit__trans_unop__0802ee0ee5__wildtype_query)
JIT_Q (GB_jit__trans_unop__0802ee0eef__gauss_query)
JIT_Q (GB_jit__user_op__0__addgauss_query)
JIT_Q (GB_jit__user_type__0__gauss_query)


//------------------------------------------------------------------------------
// GB_prejit_kernels: a list of function pointers to PreJIT kernels
//------------------------------------------------------------------------------

#if ( 20 > 0 )
static void *GB_prejit_kernels [20] =
{
GB_jit__AxB_dot4__03fe900eee0eeec5__addgauss_multgauss,
GB_jit__AxB_saxbit__03fe100eee0eee8a__addgauss_multgauss,
GB_jit__AxB_saxbit__03fe100eee0eee8a__mycx_plus_mycx_times,
GB_jit__AxB_saxpy3__03fe100eee0eee45__wildadd_wildmult,
GB_jit__AxB_saxpy4__03fe300eee0eeec7__addgauss_multgauss,
GB_jit__AxB_saxpy5__03fe500eee0eeecd__addgauss_multgauss,
GB_jit__add__1002e0e0eee00__wildtype,
GB_jit__apply_unop__0000db0dbe__fx64_cmplx_imag,
GB_jit__apply_unop__0000db0dbe__fx64_cmplx_real,
GB_jit__build__02eeeee__wildtype,
GB_jit__subassign_22__1000eee0eec8__addgauss,
GB_jit__subassign_22__1000eee0eec8__multgauss,
GB_jit__trans_bind1st__0000eee0efe41__multgauss,
GB_jit__trans_bind2nd__0000eee0eef44__multgauss,
GB_jit__trans_unop__08006e06ef__realgauss,
GB_jit__trans_unop__0802ee0ee5__gauss,
GB_jit__trans_unop__0802ee0ee5__wildtype,
GB_jit__trans_unop__0802ee0eef__gauss,
GB_jit__user_op__0__addgauss,
GB_jit__user_type__0__gauss
} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit_queries: a list of function pointers to PreJIT query kernels
//------------------------------------------------------------------------------

#if ( 20 > 0 )
static void *GB_prejit_queries [20] =
{
GB_jit__AxB_dot4__03fe900eee0eeec5__addgauss_multgauss_query,
GB_jit__AxB_saxbit__03fe100eee0eee8a__addgauss_multgauss_query,
GB_jit__AxB_saxbit__03fe100eee0eee8a__mycx_plus_mycx_times_query,
GB_jit__AxB_saxpy3__03fe100eee0eee45__wildadd_wildmult_query,
GB_jit__AxB_saxpy4__03fe300eee0eeec7__addgauss_multgauss_query,
GB_jit__AxB_saxpy5__03fe500eee0eeecd__addgauss_multgauss_query,
GB_jit__add__1002e0e0eee00__wildtype_query,
GB_jit__apply_unop__0000db0dbe__fx64_cmplx_imag_query,
GB_jit__apply_unop__0000db0dbe__fx64_cmplx_real_query,
GB_jit__build__02eeeee__wildtype_query,
GB_jit__subassign_22__1000eee0eec8__addgauss_query,
GB_jit__subassign_22__1000eee0eec8__multgauss_query,
GB_jit__trans_bind1st__0000eee0efe41__multgauss_query,
GB_jit__trans_bind2nd__0000eee0eef44__multgauss_query,
GB_jit__trans_unop__08006e06ef__realgauss_query,
GB_jit__trans_unop__0802ee0ee5__gauss_query,
GB_jit__trans_unop__0802ee0ee5__wildtype_query,
GB_jit__trans_unop__0802ee0eef__gauss_query,
GB_jit__user_op__0__addgauss_query,
GB_jit__user_type__0__gauss_query
} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit_names: a list of names of PreJIT kernels
//------------------------------------------------------------------------------

#if ( 20 > 0 )
static char *GB_prejit_names [20] =
{
"GB_jit__AxB_dot4__03fe900eee0eeec5__addgauss_multgauss",
"GB_jit__AxB_saxbit__03fe100eee0eee8a__addgauss_multgauss",
"GB_jit__AxB_saxbit__03fe100eee0eee8a__mycx_plus_mycx_times",
"GB_jit__AxB_saxpy3__03fe100eee0eee45__wildadd_wildmult",
"GB_jit__AxB_saxpy4__03fe300eee0eeec7__addgauss_multgauss",
"GB_jit__AxB_saxpy5__03fe500eee0eeecd__addgauss_multgauss",
"GB_jit__add__1002e0e0eee00__wildtype",
"GB_jit__apply_unop__0000db0dbe__fx64_cmplx_imag",
"GB_jit__apply_unop__0000db0dbe__fx64_cmplx_real",
"GB_jit__build__02eeeee__wildtype",
"GB_jit__subassign_22__1000eee0eec8__addgauss",
"GB_jit__subassign_22__1000eee0eec8__multgauss",
"GB_jit__trans_bind1st__0000eee0efe41__multgauss",
"GB_jit__trans_bind2nd__0000eee0eef44__multgauss",
"GB_jit__trans_unop__08006e06ef__realgauss",
"GB_jit__trans_unop__0802ee0ee5__gauss",
"GB_jit__trans_unop__0802ee0ee5__wildtype",
"GB_jit__trans_unop__0802ee0eef__gauss",
"GB_jit__user_op__0__addgauss",
"GB_jit__user_type__0__gauss"
} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit: return list of PreJIT function pointers and function names
//------------------------------------------------------------------------------

void GB_prejit
(
    int32_t *nkernels,      // return # of kernels
    void ***Kernel_handle,  // return list of function pointers to kernels
    void ***Query_handle,   // return list of function pointers to queries
    char ***Name_handle     // return list of kernel names
)
{
    (*nkernels) = 20 ;
    #if ( 20 == 0 )
    (*Kernel_handle) = NULL ;
    (*Query_handle) = NULL ;
    (*Name_handle) = NULL ;
    #else
    (*Kernel_handle) = GB_prejit_kernels ;
    (*Query_handle) = GB_prejit_queries ;
    (*Name_handle) = GB_prejit_names ;
    #endif
}

