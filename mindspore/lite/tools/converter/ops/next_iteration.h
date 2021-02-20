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

#ifndef LITE_MINDSPORE_LITE_C_OPS_NEXTITERATION_H_
#define LITE_MINDSPORE_LITE_C_OPS_NEXTITERATION_H_

#include <vector>
#include <set>
#include <cmath>
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {

class NextIteration : public PrimitiveC {
 public:
  NextIteration() { op_type_ = ConverterPrimitiveType_NextIteration; }
  ~NextIteration() = default;
  MS_DECLARE_PARENT(NextIteration, PrimitiveC);
  explicit NextIteration(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_NEXTITERATION_H_
