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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSETODENSE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSETODENSE_CPU_KERNEL_H_
#include <memory>
#include <unordered_map>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename I, typename T>
class SparseToDenseCPUKernel : public CPUKernel {
 public:
  SparseToDenseCPUKernel() = default;
  ~SparseToDenseCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void CheckParam(const CNodePtr &kernel_node);
  std::vector<size_t> indices_shape_;
  std::vector<size_t> values_shape_;
  std::vector<size_t> dense_shape_shape_;
  std::vector<size_t> output_shape_;
};

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt32),
                      SparseToDenseCPUKernel, int32_t, int32_t);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt8),
                      SparseToDenseCPUKernel, int32_t, int8_t);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt8),
                      SparseToDenseCPUKernel, int32_t, uint8_t);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt16),
                      SparseToDenseCPUKernel, int32_t, int16_t);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeUInt16),
                      SparseToDenseCPUKernel, int32_t, uint16_t);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeInt64),
                      SparseToDenseCPUKernel, int32_t, int64_t);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat16),
                      SparseToDenseCPUKernel, int32_t, float16);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat),
                      SparseToDenseCPUKernel, int32_t, float);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat64),
                      SparseToDenseCPUKernel, int32_t, double);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeBool),
                      SparseToDenseCPUKernel, int32_t, bool);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt32),
                      SparseToDenseCPUKernel, int64_t, int32_t);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt64),
                      SparseToDenseCPUKernel, int64_t, int64_t);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt8),
                      SparseToDenseCPUKernel, int64_t, int8_t);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeUInt8)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt8),
                      SparseToDenseCPUKernel, int64_t, uint8_t);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeInt16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeInt16),
                      SparseToDenseCPUKernel, int64_t, int16_t);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeUInt16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeUInt16),
                      SparseToDenseCPUKernel, int64_t, uint16_t);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat16),
                      SparseToDenseCPUKernel, int64_t, float16);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat),
                      SparseToDenseCPUKernel, int64_t, float);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeFloat64)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeFloat64),
                      SparseToDenseCPUKernel, int64_t, double);

MS_REG_CPU_KERNEL_T_S(SparseToDense,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeInt64)
                        .AddInputAttr(kNumberTypeBool)
                        .AddInputAttr(kNumberTypeInt64)
                        .AddOutputAttr(kNumberTypeBool),
                      SparseToDenseCPUKernel, int64_t, bool);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSETODENSE_CPU_KERNEL_H_
