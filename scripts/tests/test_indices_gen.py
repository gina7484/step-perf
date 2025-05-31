import torch
from deepseekv3.utils import create_indices

def verify_distribution(indices, num_experts):
    """
    Verify that the empirical distribution matches the target distribution.
    
    Args:
        indices: [num_tokens, num_selected] tensor of expert indices
        num_experts: total number of experts
        
    Returns:
        empirical_dist: [num_experts] empirical probability distribution
    """
    flat_indices = indices.flatten()
    counts = torch.bincount(flat_indices, minlength=num_experts)
    empirical_dist = counts.float() / counts.sum()
    return empirical_dist

if __name__ == "__main__":
    # Example parameters
    num_experts = 8
    num_tokens = 1000
    num_selected = 3  # Select 3 unique experts per token
    
    # Create a target expert_distribution (e.g., some experts are more likely)
    expert_dist = torch.tensor([0.2, 0.15, 0.1, 0.05, 0.15, 0.1, 0.15, 0.1])

    indices1 = create_indices(expert_dist, num_tokens, num_selected)
    empirical1 = verify_distribution(indices1, num_experts)
    print("\nSimple multinomial sampling:")
    print("Empirical distribution:", empirical1)
    print("Expert distribution:", expert_dist)
    print("Max difference:", (empirical1 - expert_dist).abs().max())