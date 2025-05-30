import numpy as np
import torch

def torch_tensor_to_npy(tensor: torch.Tensor, filename, dtype):
    np.save(filename, tensor.cpu().numpy().astype(dtype))

def npy_to_torch_tensor(filename):
    data = np.load(filename, allow_pickle=True)
    return torch.tensor(data, dtype=torch.float32)

def create_multihot(expert_dist, num_tokens, num_selected):
    """
    expert_dist: [num_experts] tensor of probabilities for each expert
    num_tokens: the total number of tokens
    num_selected: the number of experts to select for each token

    Return a [num_tokens, num_experts] tensor where each row is a multihot vector
    representing the selected experts for that token.
    """
    num_experts = expert_dist.shape[0]
    selected_experts = torch.multinomial(expert_dist, num_samples=num_selected, replacement=True)
    multihot = torch.zeros((num_tokens, num_experts), dtype=torch.float32)
    for i in range(num_tokens):
        multihot[i, selected_experts[i]] = 1.0
    return multihot

def create_indices(expert_dist, num_tokens, num_selected):
    """
    Create expert selection indices that maintain the target expert_distribution.
    Each row contains unique expert indices (no duplicates per token).
    
    Args:
        expert_dist: [num_experts] tensor of probabilities for each expert
        num_tokens: the total number of tokens
        num_selected: the number of experts to select for each token

    Returns:
        indices: [num_tokens, num_selected] tensor of indices where each row contains the indices
        of the selected experts for that token. Each row has unique indices.
        
    The empirical selection frequency across all tokens will approximate expert_dist.
    """
    num_experts = expert_dist.shape[0]
    
    if num_selected > num_experts:
        raise ValueError(f"Cannot select {num_selected} unique experts from {num_experts} total experts")
    
    # Ensure expert_dist is normalized
    expert_dist = expert_dist / expert_dist.sum()
    
    # Sample unique experts for each token
    indices = torch.zeros(num_tokens, num_selected, dtype=torch.long)
    
    for i in range(num_tokens):
        # Sample without replacement for this token
        selected = torch.multinomial(expert_dist, num_selected, replacement=False)
        indices[i] = selected
    
    return indices

def create_route_scale(num_tokens, num_selected):
    """
    Return a [num_tokens, num_selected] tensor where each row are the scale factors for each selected expert.
    The summation of the scale factors across selected experts for each token equal 1.
    Each scale factor is a random float32 value between 0 and 1.
    """
    
    # Generate random values between 0 and 1 for each token and selected expert
    random_values = torch.rand(num_tokens, num_selected, dtype=torch.float32)
    
    # Normalize each row so the scale factors sum to 1
    scale_factors = random_values / random_values.sum(dim=1, keepdim=True)
    
    return scale_factors


if __name__ == "__main__":
    # Example parameters
    num_experts = 8
    num_tokens = 1000
    num_selected = 3  # Select 3 unique experts per token
    
    # Create a target expert_distribution (e.g., some experts are more likely)
    expert_dist = torch.tensor([0.2, 0.15, 0.1, 0.05, 0.15, 0.1, 0.15, 0.1])


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

    indices1 = create_indices(expert_dist, num_tokens, num_selected)
    empirical1 = verify_distribution(indices1, num_experts)
    print("\nSimple multinomial sampling:")
    print("Empirical distribution:", empirical1)
    print("Expert distribution:", expert_dist)
    print("Max difference:", (empirical1 - expert_dist).abs().max())