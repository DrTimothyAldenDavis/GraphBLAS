/*
 * Copyright (c) 2019,2020 NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef GB_TYPE_NAME_H
#define GB_TYPE_NAME_H

#include <string>
#include <typeinfo>
#include <type_traits>
#include <memory>
#include <stdint.h>
#include <cstdlib>

extern "C" {
#include "GB.h"
};

/**---------------------------------------------------------------------------*
 * @file type_name.hpp
 * @brief Defines the mapping between concrete C++ types and strings.
 *---------------------------------------------------------------------------**/
namespace jit {

template <typename T>
class type_name {
public:
  static const char *name;
};

#define DECLARE_TYPE_NAME(x)  template<> inline const char *jit::type_name<x>::name = #x;
#define GET_TYPE_NAME(x) (jit::type_name<decltype(x)>::name)

DECLARE_TYPE_NAME(int);
DECLARE_TYPE_NAME(int&);
DECLARE_TYPE_NAME(int*);
DECLARE_TYPE_NAME(int8_t);
DECLARE_TYPE_NAME(int8_t&);
DECLARE_TYPE_NAME(int8_t*);
DECLARE_TYPE_NAME(unsigned char);
DECLARE_TYPE_NAME(unsigned char&);
DECLARE_TYPE_NAME(unsigned char*);
DECLARE_TYPE_NAME(unsigned int);
DECLARE_TYPE_NAME(unsigned int&);
DECLARE_TYPE_NAME(unsigned int*);
DECLARE_TYPE_NAME(unsigned int64_t);
DECLARE_TYPE_NAME(unsigned int64_t&);
DECLARE_TYPE_NAME(unsigned int64_t*);
DECLARE_TYPE_NAME(long);
DECLARE_TYPE_NAME(long&);
DECLARE_TYPE_NAME(long*);
DECLARE_TYPE_NAME(float);
DECLARE_TYPE_NAME(float&);
DECLARE_TYPE_NAME(float*);
DECLARE_TYPE_NAME(double);
DECLARE_TYPE_NAME(double&);
DECLARE_TYPE_NAME(double*);
DECLARE_TYPE_NAME(bool);



    inline const std::string grb_str_type(GB_Type_code grb_type_code) {
    switch(grb_type_code) {
        case GB_BOOL_code:
            return "bool";
        case GB_INT8_code:
            return "int8_t";
        case GB_UINT8_code:
            return "uint8_t";
        case GB_INT16_code:
            return "int16_t";
        case GB_UINT16_code:
            return "uint16_t";
        case GB_INT32_code:
            return "int32_t";
        case GB_UINT32_code:
            return "uint32_t";
        case GB_INT64_code:
            return "int64_t";
        case GB_UINT64_code:
            return "uint64_t";
        case GB_FP32_code:
            return "float";
        case GB_FP64_code:
            return "double";
        default:
            printf("Error: GrB_Type not supported.\n");
            exit(1);
    }
}
}  // namespace jit

#endif
