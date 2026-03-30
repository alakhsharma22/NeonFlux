import neonflux
import numpy as np

class NeonLayer:
    """Base class for layers"""
    def forward(self, x):
        pass

class Linear(NeonLayer):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.bias = np.zeros(out_features, dtype=np.float32)

    def forward(self, x):
        out = neonflux.matmul(x, self.weights)
        out += self.bias 
        return out

class ReLU(NeonLayer):
    def forward(self, x):
        return neonflux.relu(x)

class Sequential(NeonLayer):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
