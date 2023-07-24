import torch
import habana_frameworks.torch.core

valueA = 1456.0
valueB = 32.0
expected_division_result = valueA / valueB

def test_custom_div_op_function(custom_op_lib_path):
    torch.ops.load_library(custom_op_lib_path)
    print(torch.ops.custom_op.custom_topk)
    a_cpu = valueA
    b_cpu = valueB
    c_cpu = expected_division_result

    a_hpu = a_cpu.to('hpu')
    b_hpu = b_cpu.to('hpu')
    c_hpu = torch.ops.custom_op.custom_div(a_hpu, b_hpu)
    
    assert(torch.equal(c_hpu.detach().cpu(), c_cpu.detach().cpu()))

libpath = "./build/"
test_custom_div_op_function(libpath)

