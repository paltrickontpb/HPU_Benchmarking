+ ###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import torch
import os
import habana_frameworks.torch.core

custom_add_op_lib_path = "../build/src/libadd_lib.so"
my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind('/')
base_dir = my_dir[:my_len]
torch.ops.load_library(os.path.join(base_dir, custom_add_op_lib_path))
print(torch.ops.custom_op.custom_add_f32)

a_cpu = torch.rand((6, 6))
b_cpu = torch.rand((6, 6))
c_cpu = torch.zeros((6, 6))

a_hpu = a_cpu.to('hpu')
b_hpu = b_cpu.to('hpu')
c_hpu = c_cpu.to('hpu')

c_hpu = torch.ops.custom_op.custom_add_f32(a_hpu, b,hpu)

c_cpu = a_cpu + b_cpu

assert(torch.equal(c_hpu.detach().cpu(), c_cpu.detach().cpu()))