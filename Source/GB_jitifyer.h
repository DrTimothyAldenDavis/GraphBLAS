//------------------------------------------------------------------------------
// GB_jitifyer.h: definitions for the CPU jitifyer
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_JITIFYER_H
#define GB_JITIFYER_H

#include <dlfcn.h>

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
#define GB_JIT_KERNEL_EWISEFA       19  /* GB_ewise_full_accum      */
#define GB_JIT_KERNEL_EWISEFN       20  /* GB_ewise_full_noaccum    */
#define GB_JIT_KERNEL_APPLYBIND1    21  /* GB_apply_op              */
#define GB_JIT_KERNEL_APPLYBIND2    22  /* GB_apply_op              */
#define GB_JIT_KERNEL_TRANSBIND1    23  /* GB_transpose_op          */
#define GB_JIT_KERNEL_TRANSBIND2    24  /* GB_transpose_op          */

// unop methods:
#define GB_JIT_KERNEL_APPLYUNOP     25  /* GB_apply_op, GB_cast_array       */
#define GB_JIT_KERNEL_TRANSUNOP     26  /* GB_transpose_op, GB_transpose_ix */
#define GB_JIT_KERNEL_CONVERTS2B   101

// build method:
#define GB_JIT_KERNEL_BUILD         27  /* GB_builder               */

// select methods:
#define GB_JIT_KERNEL_SELECT1       28  /* GB_select_sparse         */
#define GB_JIT_KERNEL_SELECT2       29  /* GB_select_sparse         */
#define GB_JIT_KERNEL_SELECT_BITMAP 30  /* GB_select_bitmap         */

// assign/subassign methods: in progress
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
// GB_concat_bitmap
// GB_concat_full
// GB_concat_sparse
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
// GB_convert_sparse_to_bitmap
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
    size_t suffix_size ;        // size of suffix malloc'd block
    void *dl_handle ;           // handle from dlopen, to be passed to dlclose
    void *dl_function ;         // address of function itself, from dlsym
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

void *GB_jitifyer_lookup    // return dl_function pointer, or NULL if not found
(
    // input:
    uint64_t hash,          // hash = GB_jitifyer_hash_encoding (encoding) ;
    GB_jit_encoding *encoding,
    const char *suffix
) ;

bool GB_jitifyer_insert         // return true if successful, false if failure
(
    // input:
    uint64_t hash,              // hash for the problem
    GB_jit_encoding *encoding,  // primary encoding
    const char *suffix,         // suffix for user-defined types/operators
    void *dl_handle,            // library handle from dlopen
    void *dl_function           // function handle from dlsym
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

void GB_jitifyer_finalize (void) ;

// to query a library for its type and operator definitions
typedef const char *(*GB_jit_query_defn_func) (int k) ;

// to query a library for its type and operator definitions
typedef bool (*GB_jit_query_monoid_func)
(
    void *id,
    void *term,
    size_t id_size,
    size_t term_size
) ;

// to query a library for its version
typedef void (*GB_jit_query_version_func)
(
    int *version
) ;

bool GB_jitifyer_match_defn     // return true if definitions match
(
    // input:
    void *dl_query,             // query_defn function pointer
    int k,                      // compare current_defn with query_defn (k)
    const char *current_defn    // current definition (or NULL if not present)
) ;

bool GB_jitifyer_match_idterm   // return true if monoid id and term match
(
    void *dl_handle,            // dl_handle for the jit kernel library
    GrB_Monoid monoid           // current monoid to compare
) ;

bool GB_jitifyer_match_version
(
    void *dl_handle             // dl_handle for the jit kernel library
) ;

int GB_jitifyer_compile         // return result of system() call
(
    const char *kernel_name     // kernel to compile
) ;

#endif

