//------------------------------------------------------------------------------
// GB_jitifyer.h: definitions for the CPU jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_JITIFYER_H
#define GB_JITIFYER_H

#include <dlfcn.h>
#include "GB_jit_kernel_proto.h"

//------------------------------------------------------------------------------
// get list of PreJIT kernels: function pointers and names
//------------------------------------------------------------------------------

void GB_prejit
(
    int32_t *nkernels,      // return # of kernels
    void **Kernel_handle,   // return list of function pointers to kernels
    void **Query_handle,    // return list of function pointers to queries
    char **Name_handle      // return list of kernel names
) ;

//------------------------------------------------------------------------------
// list of jitifyed kernels
//------------------------------------------------------------------------------

// kernel families

typedef enum
{
    GB_jit_apply_family  = 0,
    GB_jit_assign_family = 1,
    GB_jit_build_family  = 2,
    GB_jit_ewise_family  = 3,
    GB_jit_mxm_family    = 4,
    GB_jit_reduce_family = 5,
    GB_jit_select_family = 6
}
GB_jit_family ;

// FIXME: make this an enum:

// reduce to scalar
#define GB_JIT_KERNEL_REDUCE        1   /* GB_reduce_to_scalar  */

// C<M> = A*B, except for row/col scale (which are ewise methods)
#define GB_JIT_KERNEL_AXB_DOT2      2   /* GB_AxB_dot2          */
#define GB_JIT_KERNEL_AXB_DOT2N     3   /* GB_AxB_dot2          */
#define GB_JIT_KERNEL_AXB_DOT3      4   /* GB_AxB_dot3          */
#define GB_JIT_KERNEL_AXB_DOT4      5   /* GB_AxB_dot4          */
#define GB_JIT_KERNEL_AXB_SAXBIT    6   /* GB_AxB_saxbit        */
#define GB_JIT_KERNEL_AXB_SAXPY3    7   /* GB_AxB_saxpy3        */
#define GB_JIT_KERNEL_AXB_SAXPY4    8   /* GB_AxB_saxpy4        */
#define GB_JIT_KERNEL_AXB_SAXPY5    9   /* GB_AxB_saxpy5        */

// ewise methods:
#define GB_JIT_KERNEL_COLSCALE      10  /* GB_colscale              */
#define GB_JIT_KERNEL_ROWSCALE      11  /* GB_rowscale              */
#define GB_JIT_KERNEL_ADD           12  /* GB_add_phase2            */
#define GB_JIT_KERNEL_UNION         13  /* GB_add_phase2            */
#define GB_JIT_KERNEL_EMULT2        14  /* GB_emult_02              */
#define GB_JIT_KERNEL_EMULT3        15  /* GB_emult_03              */
#define GB_JIT_KERNEL_EMULT4        16  /* GB_emult_04              */
#define GB_JIT_KERNEL_EMULT_BITMAP  17  /* GB_emult_bitmap          */
#define GB_JIT_KERNEL_EMULT8        18  /* GB_emult_08_phase2       */
#define GB_JIT_KERNEL_EWISEFA       19  /* GB_ewise_fulla      */
#define GB_JIT_KERNEL_EWISEFN       20  /* GB_ewise_fulln    */
#define GB_JIT_KERNEL_APPLYBIND1    21  /* GB_apply_op              */
#define GB_JIT_KERNEL_APPLYBIND2    22  /* GB_apply_op              */
#define GB_JIT_KERNEL_TRANSBIND1    23  /* GB_transpose_op          */
#define GB_JIT_KERNEL_TRANSBIND2    24  /* GB_transpose_op          */

// apply (unary and idxunary op) methods:
#define GB_JIT_KERNEL_APPLYUNOP     25  /* GB_apply_op, GB_cast_array       */
#define GB_JIT_KERNEL_TRANSUNOP     26  /* GB_transpose_op, GB_transpose_ix */
#define GB_JIT_KERNEL_CONVERTS2B    101
#define GB_JIT_KERNEL_CONCAT_SPARSE 102
#define GB_JIT_KERNEL_CONCAT_FULL   103
#define GB_JIT_KERNEL_CONCAT_BITMAP 104
#define GB_JIT_KERNEL_SPLIT_SPARSE  105
#define GB_JIT_KERNEL_SPLIT_FULL    106
#define GB_JIT_KERNEL_SPLIT_BITMAP  107

// build method:
#define GB_JIT_KERNEL_BUILD         27  /* GB_builder               */

// select methods:
#define GB_JIT_KERNEL_SELECT1       28  /* GB_select_sparse         */
#define GB_JIT_KERNEL_SELECT2       29  /* GB_select_sparse         */
#define GB_JIT_KERNEL_SELECT_BITMAP 30  /* GB_select_bitmap         */

// assign/subassign methods:
#define GB_JIT_KERNEL_SUBASSIGN_05d 36  /* GB_subassign_05d         */
#define GB_JIT_KERNEL_SUBASSIGN_06d 37  /* GB_subassign_06d         */
#define GB_JIT_KERNEL_SUBASSIGN_22  51  /* GB_subassign_22          */
#define GB_JIT_KERNEL_SUBASSIGN_23  52  /* GB_subassign_23          */
#define GB_JIT_KERNEL_SUBASSIGN_25  53  /* GB_subassign_25          */

// assign/subassign methods: todo
#define GB_JIT_KERNEL_SUBASSIGN_01  31  /* GB_subassign_01          */
#define GB_JIT_KERNEL_SUBASSIGN_02  32  /* GB_subassign_02          */
#define GB_JIT_KERNEL_SUBASSIGN_03  33  /* GB_subassign_03          */
#define GB_JIT_KERNEL_SUBASSIGN_04  34  /* GB_subassign_04          */
#define GB_JIT_KERNEL_SUBASSIGN_05  35  /* GB_subassign_05          */
#define GB_JIT_KERNEL_SUBASSIGN_06n 38  /* GB_subassign_06n         */
#define GB_JIT_KERNEL_SUBASSIGN_06s 39  /* GB_subassign_06s_and_14  */
#define GB_JIT_KERNEL_SUBASSIGN_07  40  /* GB_subassign_07          */
#define GB_JIT_KERNEL_SUBASSIGN_08n 41  /* GB_subassign_08n         */
#define GB_JIT_KERNEL_SUBASSIGN_08s 42  /* GB_subassign_08s_and_16  */
#define GB_JIT_KERNEL_SUBASSIGN_09  43  /* GB_subassign_09          */
#define GB_JIT_KERNEL_SUBASSIGN_10  44  /* GB_subassign_10_and_18   */
#define GB_JIT_KERNEL_SUBASSIGN_11  45  /* GB_subassign_11          */
#define GB_JIT_KERNEL_SUBASSIGN_12  46  /* GB_subassign_12_and_20   */
#define GB_JIT_KERNEL_SUBASSIGN_13  47  /* GB_subassign_13          */
#define GB_JIT_KERNEL_SUBASSIGN_15  48  /* GB_subassign_15          */
#define GB_JIT_KERNEL_SUBASSIGN_17  49  /* GB_subassign_17          */
#define GB_JIT_KERNEL_SUBASSIGN_19  50  /* GB_subassign_19          */

// bitmap assign/subassign: todo
#define GB_JIT_KERNEL_ASSIGN_BITMAP_M_ACC           54  /* GB_bitmap_assign_M_accum             */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_M_ACC_WHOLE     55  /* GB_bitmap_assign_M_accum_whole       */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_M_NOACC         56  /* GB_bitmap_assign_M_noaccum           */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_M_NOACC_WHOLE   57  /* GB_bitmap_assign_M_noaccum_whole     */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_FM_ACC          58  /* GB_bitmap_assign_fullM_accum         */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_FM_ACC_WHOLE    59  /* GB_bitmap_assign_fullM_accum_whole   */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_FM_NOACC        60  /* GB_bitmap_assign_fullM_noaccum       */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_FM_NOACC_WHOLE  61  /* GB_bitmap_assign_fullM_noaccum_whole */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_NOM_ACC         62  /* GB_bitmap_assign_noM_accum           */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_NOM_ACC_WHOLE   63  /* GB_bitmap_assign_noM_accum_whole     */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_NOM_NOACC       64  /* GB_bitmap_assign_noM_noaccum         */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_NOM_NOACC_WHOLE 65  /* GB_bitmap_assign_noM_noaccum_whole   */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_NM_ACC          66  /* GB_bitmap_assign_notM_accum          */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_NM_ACC_WHOLE    67  /* GB_bitmap_assign_notM_accum_whole    */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_NM_NOACC        68  /* GB_bitmap_assign_notM_noaccum        */
#define GB_JIT_KERNEL_ASSIGN_BITMAP_NM_NOACC_WHOLE  69  /* GB_bitmap_assign_notM_noaccum_whole  */

// subref methods: todo
// GB_bitmap_subref
// GB_subref_phase3

// concat/split: todo
// GB_split_bitmap
// GB_split_full
// GB_split_sparse

// masker methods: todo
// GB_masker_phase1
// GB_masker_phase2

// Kronecker: todo
// GB_kroner

// utilities: todo
// GB_check_if_iso
// GB_convert_bitmap_worker
// GB_expand_iso
// GB_sort

//------------------------------------------------------------------------------
// GB_jitifyer_entry: an entry in the jitifyer hash table
//------------------------------------------------------------------------------

struct GB_jit_encoding_struct
{
    uint64_t code ;         // from GB_enumify_*
    uint32_t kcode ;        // which kernel
    uint32_t suffix_len ;   // length of the suffix (0 for builtin)
} ;

typedef struct GB_jit_encoding_struct GB_jit_encoding ;

struct GB_jit_entry_struct
{
    uint64_t hash ;             // hash code for the problem
    GB_jit_encoding encoding ;  // encoding of the problem, except for suffix
    char *suffix ;              // kernel suffix for user-defined op / types,
                                // NULL for built-in kernels
    void *dl_handle ;           // handle from dlopen, to be passed to dlclose
    void *dl_function ;         // address of kernel function
    int64_t prejit_index ;      // -1: JIT kernel or checked PreJIT kernel
                                // >= 0: index of unchecked PreJIT kernel
} ;

typedef struct GB_jit_entry_struct GB_jit_entry ;

//------------------------------------------------------------------------------
// GB_jitifyer methods for GraphBLAS
//------------------------------------------------------------------------------

char *GB_jitifyer_libfolder (void) ;    // return path to library folder

GrB_Info GB_jitifyer_load
(
    // output:
    void **dl_function,         // pointer to JIT kernel
    // input:
    GB_jit_family family,       // kernel family
    const char *kname,          // kname for the kernel_name
    uint64_t hash,              // hash code for the kernel
    GB_jit_encoding *encoding,  // encoding of the problem
    const char *suffix,         // suffix for the kernel_name (NULL if none)
    // operator and type definitions
    GrB_Semiring semiring,
    GrB_Monoid monoid,
    GB_Operator op,
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
) ;

GrB_Info GB_jitifyer_worker
(
    // output:
    void **dl_function,         // pointer to JIT kernel
    // input:
    GB_jit_family family,       // kernel family
    const char *kname,          // kname for the kernel_name
    uint64_t hash,              // hash code for the kernel
    GB_jit_encoding *encoding,  // encoding of the problem
    const char *suffix,         // suffix for the kernel_name (NULL if none)
    // operator and type definitions
    GrB_Semiring semiring,
    GrB_Monoid monoid,
    GB_Operator op,
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
) ;

void *GB_jitifyer_lookup    // return dl_function pointer, or NULL if not found
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash_encoding (encoding) ;
    GB_jit_encoding *encoding,
    const char *suffix,
    // output
    int64_t *k1,            // location of kernel in PreJIT table
    int64_t *kk             // location of hash entry in hash table
) ;

bool GB_jitifyer_insert         // return true if successful, false if failure
(
    // input:
    uint64_t hash,              // hash for the problem
    GB_jit_encoding *encoding,  // primary encoding
    const char *suffix,         // suffix for user-defined types/operators
    void *dl_handle,            // library handle from dlopen (NULL for PreJIT)
    void *dl_function,          // function handle from dlsym
    int32_t prejit_index        // index into PreJIT table; -1 if JIT kernel
) ;

uint64_t GB_jitifyer_hash_encoding
(
    GB_jit_encoding *encoding
) ;

uint64_t GB_jitifyer_hash
(
    const void *bytes,      // any string of bytes
    size_t nbytes,          // # of bytes to hash
    bool jitable            // true if the object can be JIT'd
) ;

// to query a library for its type and operator definitions
typedef GB_JIT_QUERY_PROTO ((*GB_jit_query_func)) ;

bool GB_jitifyer_query
(
    GB_jit_query_func dl_query,
    uint64_t hash,              // hash code for the kernel
    // operator and type definitions
    GrB_Semiring semiring,
    GrB_Monoid monoid,
    GB_Operator op,
    GrB_Type type1,
    GrB_Type type2,
    GrB_Type type3
) ;

int GB_jitifyer_compile (char *kernel_name) ;  // compile a kernel

GrB_Info GB_jitifyer_init (void) ;  // initialize the JIT

void GB_jitifyer_finalize (bool freeall) ;      // finalize the JIT

void GB_jitifyer_table_free (bool freeall) ;    // free the JIT table

GrB_Info GB_jitifyer_alloc_space (void) ;

GrB_Info GB_jitifyer_include (void) ;

void GB_jitifyer_set_control (int control) ;
GxB_JIT_Control GB_jitifyer_get_control (void) ;

const char *GB_jitifyer_get_source_path (void) ;
GrB_Info GB_jitifyer_set_source_path (const char *new_source_path) ;
GrB_Info GB_jitifyer_set_source_path_worker (const char *new_source_path) ;

const char *GB_jitifyer_get_cache_path (void) ;
GrB_Info GB_jitifyer_set_cache_path (const char *new_cache_path) ;
GrB_Info GB_jitifyer_set_cache_path_worker (const char *new_cache_path) ;

const char *GB_jitifyer_get_C_compiler (void) ;
GrB_Info GB_jitifyer_set_C_compiler (const char *new_C_compiler) ;
GrB_Info GB_jitifyer_set_C_compiler_worker (const char *new_C_compiler) ;

const char *GB_jitifyer_get_C_flags (void) ;
GrB_Info GB_jitifyer_set_C_flags (const char *new_C_flags) ;
GrB_Info GB_jitifyer_set_C_flags_worker (const char *new_C_flags) ;

const char *GB_jitifyer_get_C_link_flags (void) ;
GrB_Info GB_jitifyer_set_C_link_flags (const char *new_C_link_flags) ;
GrB_Info GB_jitifyer_set_C_link_flags_worker (const char *new_C_link_flags) ;

#endif

