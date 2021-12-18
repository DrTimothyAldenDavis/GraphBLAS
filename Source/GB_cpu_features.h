//------------------------------------------------------------------------------
// GB_cpu_features.h: cpu features for GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Include files from Google's cpu_features package.

#ifndef GB_CPU_FEATURES_H
#define GB_CPU_FEATURES_H

#include "cpu_features_macros.h"

#if defined ( CPU_FEATURES_ARCH_ARM )
// 32-bit ARM
#include "cpuinfo_arm.h"
#endif

#if defined ( CPU_FEATURES_ARCH_AARCH64 )
// 64-bit ARM
#include "cpuinfo_aarch64.h"
#endif

#if defined ( CPU_FEATURES_ARCH_MIPS )
// MIPS
#include "cpuinfo_mips.h"
#endif

#if defined ( CPU_FEATURES_ARCH_PPC )
// IBM Power
#include "cpuinfo_ppc.h"
#endif

#if defined ( CPU_FEATURES_ARCH_X86 )
// Intel x86 (also AMD)
#include "cpuinfo_x86.h"
#endif

#endif

