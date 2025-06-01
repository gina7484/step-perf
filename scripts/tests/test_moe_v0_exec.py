import torch
from deepseekv3.moe_v0 import ModelArgs, MoE
from deepseekv3.utils import torch_tensor_to_npy, create_indices, create_route_scale
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

def create_weights(model: MoE):
    torch.manual_seed(42)
    for expert in model.experts:
        if expert is None:
            continue
        expert.w1.weight.data = torch.randn_like(expert.w1.weight)
        expert.w2.weight.data = torch.randn_like(expert.w2.weight)
        expert.w3.weight.data = torch.randn_like(expert.w3.weight)
    shared = model.shared_experts
    shared.w1.weight.data = torch.randn_like(shared.w1.weight)
    shared.w2.weight.data = torch.randn_like(shared.w2.weight)
    shared.w3.weight.data = torch.randn_like(shared.w3.weight)

def create_inputs(n_routed_experts: int, dim: int, n_activated_experts: int, num_tokens: int):
    input_tensor = torch.randn(num_tokens, dim, dtype=torch.float32)
    # Get random expert_dist
    expert_distribution = torch.rand(n_routed_experts, dtype=torch.float32)
    expert_distribution /= expert_distribution.sum()
    print(f"Expert distribution: {expert_distribution.tolist()}")
    # Create indices for expert selection
    indices = create_indices(expert_distribution, num_tokens, n_activated_experts)
    # Create route scales for each selected expert
    scales = create_route_scale(num_tokens, n_activated_experts)
    return input_tensor, indices, scales

def save_weights(model: MoE, base_path: str):
    for i, expert in enumerate(model.experts):
        if expert is None:
            continue
        torch_tensor_to_npy(expert.w1.weight, f"{base_path}/expert_{i}_w1.npy", dtype=np.float32)
        torch_tensor_to_npy(expert.w2.weight, f"{base_path}/expert_{i}_w2.npy", dtype=np.float32)
        torch_tensor_to_npy(expert.w3.weight, f"{base_path}/expert_{i}_w3.npy", dtype=np.float32)
    torch_tensor_to_npy(model.shared_experts.w1.weight, f"{base_path}/shared_w1.npy", dtype=np.float32)
    torch_tensor_to_npy(model.shared_experts.w2.weight, f"{base_path}/shared_w2.npy", dtype=np.float32)
    torch_tensor_to_npy(model.shared_experts.w3.weight, f"{base_path}/shared_w3.npy", dtype=np.float32)

def save_tensors(base_path: str, input, indices, scales, output):
    torch_tensor_to_npy(input, f"{base_path}/input.npy", dtype=np.float32)
    torch_tensor_to_npy(indices, f"{base_path}/indices.npy", dtype=np.int64)
    torch_tensor_to_npy(scales, f"{base_path}/scales.npy", dtype=np.float32)
    torch_tensor_to_npy(output, f"{base_path}/output.npy", dtype=np.float32)


class MoEParallelTester:
    def __init__(self, model_args: ModelArgs, input_tensor, indices, scales):
        self.model_args = model_args
        self.input_tensor = input_tensor
        self.indices = indices
        self.scales = scales
        self.output = None
    
    def run_parallel(self, rank, world_size, return_dict):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        model = MoE(self.model_args)
        create_weights(model)
        print(f"Rank {rank} finished loading weights.")
        with torch.no_grad():
            output = model(self.input_tensor, self.scales, self.indices)
            save_weights(model, self.model_args.base_path) 
        if rank == 0:
            return_dict['output'] = output
        dist.destroy_process_group()


    def run_worker(self, rank, world_size, return_dict):
        return self.run_parallel(rank, world_size, return_dict)

    def kickoff(self, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Use Manager to share data between processes
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # Launch processes
        mp.spawn(self.run_worker, args=(world_size, return_dict), nprocs=world_size, join=True)
        
        # Get output from shared dictionary
        self.output = return_dict.get('output', None)

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
    expert_par = 8
    num_tokens = 10
    
    input_tensor, indices, scales = create_inputs(
        model_args.n_routed_experts, model_args.dim, model_args.n_activated_experts, num_tokens
    )
    tester = MoEParallelTester(model_args, input_tensor, indices, scales)
    tester.kickoff(world_size=expert_par)
    if tester.output is not None:
        save_tensors(model_args.base_path, input_tensor, indices, scales, tester.output)
        print("Test completed successfully and data saved.")
    else:
        print("Test failed or no output generated.")


