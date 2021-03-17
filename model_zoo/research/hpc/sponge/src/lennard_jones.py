# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""lennard jones"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from mindspore.ops import operations as P


class Lennard_Jones_Information(nn.Cell):
    """class Lennard Jones Information"""

    def __init__(self, controller):
        super(Lennard_Jones_Information, self).__init__()
        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)
        self.atom_LJ_type = Tensor(np.asarray(self.atom_LJ_type, dtype=np.int32), mstype.int32)
        self.LJ_A = Tensor(np.asarray(self.LJ_A, dtype=np.float32), mstype.float32)
        self.LJ_B = Tensor(np.asarray(self.LJ_B, dtype=np.float32), mstype.float32)
        self.LJ_energy_sum = 0
        self.LJ_energy = 0

    def read_information_from_amberfile(self, file_path):
        """read information from amberfile"""
        file = open(file_path, 'r')
        context = file.readlines()
        file.close()

        for idx, val in enumerate(context):
            if idx < len(context) - 1:
                if "%FLAG POINTERS" in val + context[idx + 1] and "%FORMAT(10I8)" in val + context[idx + 1]:
                    start_idx = idx + 2
                    count = 0
                    value = list(map(int, context[start_idx].strip().split()))
                    self.atom_numbers = value[0]
                    self.atom_type_numbers = value[1]
                    self.pair_type_numbers = int(self.atom_type_numbers * (self.atom_type_numbers + 1) / 2)
                    print(self.pair_type_numbers)
                    break
        self.atom_LJ_type = [0] * self.atom_numbers
        for idx, val in enumerate(context):
            if "%FLAG ATOM_TYPE_INDEX" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.atom_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                for i in range(self.atom_numbers):
                    self.atom_LJ_type[i] = information[i] - 1
                break
        self.LJ_A = [0] * self.pair_type_numbers
        for idx, val in enumerate(context):
            if "%FLAG LENNARD_JONES_ACOEF" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.pair_type_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                for i in range(self.pair_type_numbers):
                    self.LJ_A[i] = 12.0 * information[i]
                break
        self.LJ_B = [0] * self.pair_type_numbers
        for idx, val in enumerate(context):
            if "%FLAG LENNARD_JONES_BCOEF" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.pair_type_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                for i in range(self.pair_type_numbers):
                    self.LJ_B[i] = 6.0 * information[i]
                break

    def LJ_Energy(self, uint_crd_with_LJ, uint_dr_to_dr_cof, nl_atom_numbers, nl_atom_serial, cutoff_square):
        """compute LJ energy"""
        uint_crd, LJtype, charge = uint_crd_with_LJ
        self.LJ_energy = P.LJEnergy(self.atom_numbers, cutoff_square) \
            (uint_crd, LJtype, charge, uint_dr_to_dr_cof, nl_atom_numbers, nl_atom_serial, self.LJ_A, self.LJ_B)
        self.LJ_energy_sum = P.ReduceSum()(self.LJ_energy)
        return self.LJ_energy_sum

    def LJ_Force_With_PME_Direct_Force(self, atom_numbers, uint_crd_with_LJ, uint_dr_to_dr_cof, nl_number, nl_serial,
                                       cutoff, beta):
        """compute LJ force with PME direct force"""
        assert atom_numbers == self.atom_numbers
        assert isinstance(uint_crd_with_LJ, tuple)
        uint_crd_f, LJtype, charge = uint_crd_with_LJ
        self.ljfd = P.LJForceWithPMEDirectForce(atom_numbers, cutoff, beta)
        frc = self.ljfd(uint_crd_f, LJtype, charge, uint_dr_to_dr_cof, nl_number, nl_serial, self.LJ_A, self.LJ_B)
        return frc
