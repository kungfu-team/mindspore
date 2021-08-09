/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/kernel_compiler/gpu/math/truncate_mod_kernel.h"

namespace mindspore {
namespace kernel {
// fp32
MS_REG_GPU_KERNEL_ONE(
  TruncateMod,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  TruncateModOpGpuKernel, float)
// fp16
MS_REG_GPU_KERNEL_ONE(
  TruncateMod,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
  TruncateModOpGpuKernel, half)
// int32
MS_REG_GPU_KERNEL_ONE(
  TruncateMod,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  TruncateModOpGpuKernel, int)
// int8_t
MS_REG_GPU_KERNEL_ONE(
  TruncateMod, KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
  TruncateModOpGpuKernel, int8_t)
// uint8_t
MS_REG_GPU_KERNEL_ONE(
  TruncateMod,
  KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
  TruncateModOpGpuKernel, uint8_t)
}  // namespace kernel
}  // namespace mindspore

