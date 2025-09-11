import numpy as np
import copy

class Agent:
    def __init__(self, in_dim, hidden, out_dim):
        self.shapes = []
        last = in_dim
        
        for h in hidden:
            self.shapes.append((last, h))
            last = h
            
        self.shapes.append((last, out_dim))
        
        rng = np.random.default_rng()
        self.W = [rng.normal(0, 0.5, size = s) for s in self.shapes]
        self.b = [np.zeros((s[1],), dtype=np.float32) for s in self.shapes]
        
    def clone(self):
        m = object.__new__(Agent)     # Creates an empty object of Agent class, without triggering the init function, thus we don't need to give the function parameters      
        m.shapes = copy.deepcopy(self.shapes)
        m.W = [copy.deepcopy(w) for w in self.W]
        m.b = [copy.deepcopy(bias) for bias in self.b]
        return m
    
    def mutate(self, sigma_w, sigma_b, p):
        rng = np.random.default_rng()
        for li, (W, b) in enumerate(zip(self.W, self.b)):
            if rng.random() < 0.9:
                mask_w = (rng.random(W.shape) < p)
                noise_w = rng.normal(0, sigma_w, size=W.shape)
                self.W[li] = W + noise_w*mask_w
                
                mask_b = (rng.random(b.shape) < p)
                noise_b = rng.normal(0, sigma_b, size=b.shape)
                self.b[li] = b + noise_b*mask_b
    
    def forward(self, x):
        h = x
        
        for (W, b) in list(zip(self.W, self.b))[:-1]:
            h = tanh(h @ W + b)
        out = h @ self.W[-1] + self.b[-1]
        return out
    
    def act(self, observations):
        y = self.forward(observations)
        
        return 1 if y[0] > 0.0 else 0

def tanh(x):
    return np.tanh(x).astype(np.float32)