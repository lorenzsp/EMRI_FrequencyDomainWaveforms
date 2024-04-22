import numpy as np

def de_proposal(current_state, F=0.5, CR=0.5):
    """
    Provides a proposal for MCMC using Differential Evolution (DE/rand/1).

    Parameters:
        current_state (numpy.ndarray): The current state of the MCMC chain. Shape: (n_walkers, n_params).
        F (float): The differential weight (default is 0.5).
        CR (float): The crossover probability (default is 0.5).

    Returns:
        numpy.ndarray: The proposed state. Shape: (n_walkers, n_params).
    """
    n_walkers, n_params = current_state.shape

    # Randomly select three distinct indices for each walker
    indices = np.random.choice(n_walkers, size=(n_walkers, 3), replace=True)
    
    # Generate mutant vectors using DE/rand/1
    mutant_vectors = current_state[indices[:, 0]] + F * (current_state[indices[:, 1]] - current_state[indices[:, 2]])

    # Perform crossover with the current state to create the proposed state
    crossover_mask = (np.random.rand(n_walkers, n_params) <= CR) | (np.arange(n_params) == np.random.randint(n_params, size=(n_walkers, 1)))
    proposed_state = np.where(crossover_mask, mutant_vectors, current_state)

    return proposed_state

# Example usage:
n_walkers = 10
n_params = 3
current_state = np.random.rand(n_walkers, n_params)  # Example current state

proposed_state = de_proposal(current_state)
print("Current State:\n", current_state)
print("\nProposed State:\n", proposed_state)

print("\ncheck State:\n", current_state==proposed_state)