from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


class ParallelismConfig:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.rank = dist.get_rank()
    
parallel_config = None

def get_world_size():
    return parallel_config.world_size

def get_rank():
    return parallel_config.rank
        

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        dim (int): Model dimension.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        world_size (int): Number of distributed processes or devices.
    """

    dim: int = 2048
    moe_inter_dim: int = 1408
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()

class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module with index & scale as forward arguments

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.world_size = get_world_size()
        assert args.n_routed_experts % self.world_size == 0, f"Number of experts must be divisible by world size (world_size={self.world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // self.world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = get_rank() * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = Linear(args.dim, args.n_shared_experts * args.moe_inter_dim) # Only consider expert paralleism for now

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        if self.world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)

def initialize_parallelism(expert_parallel_size: int):
    global parallel_config
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo')
    
    parallel_config = ParallelismConfig(world_size=expert_parallel_size)