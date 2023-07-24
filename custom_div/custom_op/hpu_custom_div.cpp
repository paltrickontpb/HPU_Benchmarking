
// TODO: import exposed file
#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <synapse_common_types.hpp>

bool register_custom_div() {
    // Registering ustom_op::custom_div
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};

    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      //auto k = inputs[1].toInt(); // k
      //auto dim = inputs[2].toInt(); // dim
      std::vector<int64_t> result_sizes = self.sizes().vec();
      //if (result_sizes.size() > 0) {
      //  result_sizes[dim] = k;
      //}
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};
    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    /* user param callback
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(synBeamParams);
      auto self = inputs[0].toTensor(); // input
      params->bsw = inputs[1].toInt(); // k
      auto dim = inputs[2].toInt(); // axis
      params->axis = self.dim() - dim - 1;
      params->bottomK = inputs[3].toBool(); // bottom
      return params;
    };*/

    // acctual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_div", //schema name
        "ustomdiv_fwd_f32", // guid
        inputs_desc,
        outputs_desc,
        nullptr);
    std::cout << "cpp registered custom_op::custom_div\n";
    return true;
}

at::Tensor custom_div_execute(
    torch::Tensor input_a,
    torch::Tensor input_b) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_div();
  std::vector<c10::IValue> inputs{input_a, input_b};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_div");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_div(Tensor input_a, Tensor input_b) -> (Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_div", custom_div_execute);
}
