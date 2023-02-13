//------------------------------------------------------------------------------
// GB_mask_shared_definitions.h: common macros for masks
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_mask_shared_defintions.h provides default definitions for all masks,
// if the special cases have not been #define'd prior to #include'ing this
// file.  This file is not used for the JIT kernels.

#ifndef GB_MASK_SHARED_DEFINITIONS_H
#define GB_MASK_SHARED_DEFINITIONS_H

// mask type for generic and pre-generated kernels is GB_void
#ifndef GB_M_TYPE
#define GB_M_TYPE GB_void
#endif

// GB_MCAST(Mx,p,msize) reads the value of the mask at position p.
#ifndef GB_MCAST
#define GB_MCAST(Mx,p,msize) GB_mcast ((GB_void *) Mx, p, msize)
#endif

#endif

