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

#ifndef GB_CONV_TYPE_H
#define GB_CONV_TYPE_H
extern "C" {

#include "GB.h"
};
#include <stdint.h>


/**---------------------------------------------------------------------------*
 * @file type_convert.hpp
 * @brief Defines the mapping between concrete C++ types and Grb types.
 *---------------------------------------------------------------------------**/
namespace cuda {

template <typename T>
GrB_Type to_grb_type();

template<> GrB_Type to_grb_type<int8_t>() { return GrB_INT8; }
template<> GrB_Type to_grb_type<int16_t>() { return GrB_INT16; }
template<> GrB_Type to_grb_type<int32_t>() { return GrB_INT32; }
template<> GrB_Type to_grb_type<int64_t>() { return GrB_INT64; }
template<> GrB_Type to_grb_type<uint8_t>() { return GrB_UINT8; }
template<> GrB_Type to_grb_type<uint16_t>() { return GrB_UINT16; }
template<> GrB_Type to_grb_type<uint32_t>() { return GrB_UINT32; }
template<> GrB_Type to_grb_type<uint64_t>() { return GrB_UINT64; }
template<> GrB_Type to_grb_type<float>() { return GrB_FP32; }
template<> GrB_Type to_grb_type<double>() { return GrB_FP64; }
template<> GrB_Type to_grb_type<bool>() { return GrB_BOOL; }

}  // namespace cuda
#endif
