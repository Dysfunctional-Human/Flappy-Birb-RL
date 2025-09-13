import numpy as np
import copy

# Creating the Agent class
class Agent:
    def __init__(self, in_dim, hidden, out_dim):
        """Initializing an Agent object

        Args:
            in_dim (int): Input dimensions (Here, 4 - [dx, dy, vy, gap_half])
            hidden (int): Dimensions of hidden layer
            out_dim (int): Output dimensions (Here, 1 - jump or not jump) 
        """
        self.shapes = []        # Array of shapes of layers
        last = in_dim       
        
        for h in hidden:
            self.shapes.append((last, h))
            last = h
            
        self.shapes.append((last, out_dim))
        
        rng = np.random.default_rng()       
        self.W = [rng.normal(0, 0.5, size = s) for s in self.shapes]        # weights array
        self.b = [np.zeros((s[1],), dtype=np.float32) for s in self.shapes]     # bias array
        
    def clone(self):
        """Make a deep copy of the agent for passing onto next generation

        Returns:
            Agent: Clone of current agent
        """
        m = object.__new__(Agent)     # Creates an empty object of Agent class, without triggering the init function, thus we don't need to give the function parameters      
        m.shapes = copy.deepcopy(self.shapes)       
        m.W = [copy.deepcopy(w) for w in self.W]
        m.b = [copy.deepcopy(bias) for bias in self.b]
        return m
    
    def mutate(self, sigma_w, sigma_b, p):
        """Mutating current agent's weights and biases for more diversity

        Args:
            sigma_w (float): The amount by which the weights should be altered.
            sigma_b (float): The amount by which the biases should be altered.
            p (float): The probability of mutating a weight or bias
        """
        rng = np.random.default_rng()
        for li, (W, b) in enumerate(zip(self.W, self.b)):
            if rng.random() < 0.9:
                mask_w = (rng.random(W.shape) < p)      # mask of 1s and 0s to decide what weights should be altered
                noise_w = rng.normal(0, sigma_w, size=W.shape)      # noise to be added to the weight
                self.W[li] = W + noise_w*mask_w     # mutated weight
                
                # similarly for bias
                mask_b = (rng.random(b.shape) < p)
                noise_b = rng.normal(0, sigma_b, size=b.shape)
                self.b[li] = b + noise_b*mask_b
    
    def forward(self, x):
        """Forward method for the network

        Args:
            x (List[float]): Observations from the game - [dx, dy, vy, gap_half]

        Returns:
            out(float): Output from the network lies between -1 and 1
        """
        h = x
        
        for (W, b) in list(zip(self.W, self.b))[:-1]:
            h = tanh(h @ W + b)     # Using tanh as the activation function
        out = h @ self.W[-1] + self.b[-1]
        return out
    
    def act(self, observations):
        """Agent's decision for whether bird should jump or not

        Args:
            observations (List[float]): Observations from the game - [dx, dy, vy, gap_half]

        Returns:
            int: Returns 1 for jump and 0 for not jump
        """
        y = self.forward(observations)
        
        return 1 if y[0] > 0.0 else 0

def tanh(x):
    """Performs tanh operation 

    Args:
        x (float): Input number

    Returns:
        float: Returns an answer between -1 and 1
    """
    return np.tanh(x).astype(np.float32)