import torch
from deepseekv3.moe_v0 import ModelArgs, MoE
from deepseekv3.utils import npy_to_torch_tensor

def load_weights(model, base_path: str):
    for i in range(model.n_routed_experts):
        if model.experts[i] is None:
            continue
        model.experts[i].w1.weight.data = npy_to_torch_tensor(f"{base_path}/expert_{i}_w1.npy")
        model.experts[i].w2.weight.data = npy_to_torch_tensor(f"{base_path}/expert_{i}_w2.npy")
        model.experts[i].w3.weight.data = npy_to_torch_tensor(f"{base_path}/expert_{i}_w3.npy")
    model.shared_experts.w1.weight.data = npy_to_torch_tensor(f"{base_path}/shared_w1.npy")
    model.shared_experts.w2.weight.data = npy_to_torch_tensor(f"{base_path}/shared_w2.npy")
    model.shared_experts.w3.weight.data = npy_to_torch_tensor(f"{base_path}/shared_w3.npy")

def load_tensors(base_path: str):
    input_tensor = npy_to_torch_tensor(f"{base_path}/input.npy")
    indices = npy_to_torch_tensor(f"{base_path}/indices.npy").long()
    scales = npy_to_torch_tensor(f"{base_path}/scales.npy")
    output = npy_to_torch_tensor(f"{base_path}/output.npy")
    return input_tensor, indices, scales, output

if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)
    model_args = ModelArgs(
        dim=2048,
        moe_inter_dim=1408,
        n_routed_experts=64,
        n_shared_experts=2,
        n_activated_experts=6,
        base_path = "/scratch/zgh23/step-perf/data"
    )
    model = MoE(model_args)
    load_weights(model, model_args.base_path)
    input_tensor, indices, scales, expected_output = load_tensors(model_args.base_path)
    with torch.no_grad():
        output = model(input_tensor, scales, indices)
    # Check the result
    if torch.allclose(output, expected_output, rtol=1e-3):
        print("Test passed: Output matches expected output.")
    else:
        print("Test failed: Output does not match expected output.")
        # print the idx where output - expected_output is not close
        diff = output - expected_output

        # Get indices where values are not close
        mismatch = ~torch.isclose(output, expected_output, rtol=1e-3)

        # Print detailed diagnostic info
        print("Test failed: Output does not match expected output.")
        print("Mismatch indices:", mismatch.nonzero(as_tuple=True))
        print("Output values:", output[mismatch])
        print("Expected values:", expected_output[mismatch])
        print("Differences:", diff[mismatch])
        