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
"""md information"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from mindspore.ops import operations as P


class md_information(nn.Cell):
    """class md information"""

    def __init__(self, controller):
        super(md_information, self).__init__()
        CONSTANT_TIME_CONVERTION = 20.455
        CONSTANT_UINT_MAX_FLOAT = 4294967296.0
        self.md_task = controller.md_task
        self.mode = 0 if "mode" not in controller.Command_Set else int(controller.Command_Set["mode"])
        self.dt = 0.001 * CONSTANT_TIME_CONVERTION if "dt" not in controller.Command_Set \
            else float(controller.Command_Set["dt"]) * CONSTANT_TIME_CONVERTION
        self.skin = 2.0 if "skin" not in controller.Command_Set \
            else float(controller.Command_Set["skin"])
        self.trans_vec = [self.skin, self.skin, self.skin]
        self.trans_vec_minus = -1 * self.trans_vec
        self.step_limit = 1000 if "step_limit" not in controller.Command_Set else int(
            controller.Command_Set["step_limit"])
        self.netfrc = 0 if "net_force" not in controller.Command_Set else int(controller.Command_Set["net_force"])
        self.ntwx = 1000 if "write_information_interval" not in controller.Command_Set else \
            int(controller.Command_Set["write_information_interval"])
        self.ntce = self.step_limit + 1 if "calculate_energy_interval" not in controller.Command_Set else \
            int(controller.Command_Set["calculate_energy_interval"])
        self.atom_numbers = 0
        self.residue_numbers = 0
        self.density = 0.0
        self.lin_serial = []
        self.h_res_start = []
        self.h_res_end = []
        self.h_mass = []
        self.h_mass_inverse = []
        self.h_charge = []
        self.steps = 0

        if controller.amber_parm is not None:
            self.read_basic_system_information_from_amber_file(controller.amber_parm)

        if "amber_irest" in controller.Command_Set:
            amber_irest = int(controller.Command_Set["amber_irest"])
            if controller.initial_coordinates_file is not None:
                self.read_basic_system_information_from_rst7(controller.initial_coordinates_file, amber_irest)

        self.crd_to_uint_crd_cof = [CONSTANT_UINT_MAX_FLOAT / self.box_length[0],
                                    CONSTANT_UINT_MAX_FLOAT / self.box_length[1],
                                    CONSTANT_UINT_MAX_FLOAT / self.box_length[2]]
        self.uint_dr_to_dr_cof = [1.0 / self.crd_to_uint_crd_cof[0], 1.0 / self.crd_to_uint_crd_cof[1],
                                  1.0 / self.crd_to_uint_crd_cof[2]]
        self.density *= 1e24 / 6.023e23 / (self.box_length[0] * self.box_length[1] * self.box_length[2])
        self.frc = Tensor(np.zeros((self.atom_numbers, 3)), mstype.float32)
        self.crd = Tensor(np.array(self.coordinate, dtype=np.float32).reshape((self.atom_numbers, 3)), mstype.float32)
        self.crd_n = np.array(self.coordinate).reshape([self.atom_numbers, 3])
        self.crd_old = Tensor(np.zeros([self.atom_numbers, 3], dtype=np.float32), mstype.float32)
        self.uint_crd = Tensor(np.zeros([self.atom_numbers, 3], dtype=np.uint32), mstype.uint32)
        self.charge = Tensor(self.h_charge, mstype.float32)
        self.crd_to_uint_crd_cof_n = np.array(self.crd_to_uint_crd_cof)
        self.crd_to_uint_crd_cof = Tensor(self.crd_to_uint_crd_cof, mstype.float32)
        self.uint_dr_to_dr_cof = Tensor(self.uint_dr_to_dr_cof, mstype.float32)
        self.uint_crd_with_LJ = None
        self.d_mass_inverse = Tensor(self.h_mass_inverse, mstype.float32)
        self.d_res_start = Tensor(self.h_res_start, mstype.int32)
        self.d_res_end = Tensor(self.h_res_end, mstype.int32)
        self.d_mass = Tensor(self.h_mass, mstype.float32)

    def process1(self, context):
        """process1: read basic system information from amber file"""
        for idx, val in enumerate(context):
            if idx < len(context) - 1:
                if "%FLAG POINTERS" in val + context[idx + 1] and "%FORMAT(10I8)" in val + context[idx + 1]:
                    start_idx = idx + 2
                    value = list(map(int, context[start_idx].strip().split()))
                    self.atom_numbers = value[0]
                    count = len(value) - 1
                    while count < 10:
                        start_idx += 1
                        value = list(map(int, context[start_idx].strip().split()))
                        count += len(value)
                    self.residue_numbers = list(map(int, context[start_idx].strip().split()))[10 - (count - 10)]
                    break

    def read_basic_system_information_from_amber_file(self, path):
        """read basic system information from amber file"""
        file = open(path, 'r')
        context = file.readlines()
        file.close()
        self.process1(context)

        if self.residue_numbers != 0 and self.atom_numbers != 0:
            for idx, val in enumerate(context):
                if "%FLAG RESIDUE_POINTER" in val:
                    count = 0
                    start_idx = idx
                    while count != self.residue_numbers:
                        start_idx += 1
                        if "%FORMAT" in context[start_idx]:
                            continue
                        else:
                            value = list(map(int, context[start_idx].strip().split()))
                            self.lin_serial.extend(value)
                            count += len(value)
                    for i in range(self.residue_numbers - 1):
                        self.h_res_start.append(self.lin_serial[i] - 1)
                        self.h_res_end.append(self.lin_serial[i + 1] - 1)
                    self.h_res_start.append(self.lin_serial[-1] - 1)
                    self.h_res_end.append(self.atom_numbers + 1 - 1)
                    break

            for idx, val in enumerate(context):
                if "%FLAG MASS" in val:
                    count = 0
                    start_idx = idx
                    while count != self.atom_numbers:
                        start_idx += 1
                        if "%FORMAT" in context[start_idx]:
                            continue
                        else:
                            value = list(map(float, context[start_idx].strip().split()))
                            self.h_mass.extend(value)
                            count += len(value)
                    for i in range(self.atom_numbers):
                        if self.h_mass[i] == 0:
                            self.h_mass_inverse.append(0.0)
                        else:
                            self.h_mass_inverse.append(1.0 / self.h_mass[i])
                        self.density += self.h_mass[i]
                    break
            for idx, val in enumerate(context):
                if "%FLAG CHARGE" in val:
                    count = 0
                    start_idx = idx
                    while count != self.atom_numbers:
                        start_idx += 1
                        if "%FORMAT" in context[start_idx]:
                            continue
                        else:
                            value = list(map(float, context[start_idx].strip().split()))
                            self.h_charge.extend(value)
                            count += len(value)
                    break

    def read_basic_system_information_from_rst7(self, path, irest):
        """read basic system information from rst7"""
        file = open(path, 'r')
        context = file.readlines()
        file.close()
        atom_numbers = int(context[1].strip().split()[0])
        if atom_numbers != self.atom_numbers:
            print("ERROR")
        else:
            print("check atom_numbers")
        information = []
        count = 0
        start_idx = 1
        if irest == 1:
            self.simulation_start_time = float(context[1].strip().split()[1])
            while count <= 6 * self.atom_numbers + 3:
                start_idx += 1
                # print(start_idx)
                value = list(map(float, context[start_idx].strip().split()))
                information.extend(value)
                count += len(value)
            self.coordinate = information[: 3 * self.atom_numbers]
            self.velocity = information[3 * self.atom_numbers: 6 * self.atom_numbers]
            self.box_length = information[6 * self.atom_numbers:6 * self.atom_numbers + 3]
        else:
            while count <= 3 * self.atom_numbers + 3:
                start_idx += 1
                value = list(map(float, context[start_idx].strip().split()))
                information.extend(value)
                count += len(value)
            self.coordinate = information[: 3 * self.atom_numbers]
            self.velocity = [0.0] * (3 * self.atom_numbers)
            self.box_length = information[3 * self.atom_numbers:3 * self.atom_numbers + 3]
        self.vel = Tensor(self.velocity, mstype.float32).reshape((self.atom_numbers, 3))
        self.acc = Tensor(np.zeros((self.atom_numbers, 3), dtype=np.float32), mstype.float32)

    def MD_Information_Crd_To_Uint_Crd(self):
        """transform the crd to uint crd"""
        uint_crd = self.crd.asnumpy() * (0.5 * self.crd_to_uint_crd_cof.asnumpy()) * 2
        self.uint_crd = Tensor(uint_crd, mstype.uint32)

        return self.uint_crd

    def Centerize(self):
        return

    def MD_Information_Temperature(self):
        """compute temperature"""
        self.mdtemp = P.MDTemperature(self.residue_numbers, self.atom_numbers)
        self.res_ek_energy = self.mdtemp(self.d_res_start, self.d_res_end, self.vel, self.d_mass)
        self.d_temperature = P.ReduceSum()(self.res_ek_energy)
        return self.d_temperature
