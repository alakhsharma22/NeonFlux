import time
import numpy as np
from neon_nn import Linear, ReLU, Sequential

BATCH = 64
INPUT_DIM = 128
HIDDEN_DIM = 1024
OUTPUT_DIM = 10
N_ITERS = 100
WARMUP_ITERS = 10

print(f"Benchmarking NeonFlux MLP: [Input({INPUT_DIM}) -> Linear({HIDDEN_DIM}) -> ReLU -> Linear({HIDDEN_DIM}) -> ReLU -> Linear({OUTPUT_DIM})]")
print(f"Batch Size: {BATCH}")
print("-" * 60)

model_neon = Sequential([
    Linear(INPUT_DIM, HIDDEN_DIM),
    ReLU(),
    Linear(HIDDEN_DIM, HIDDEN_DIM),
    ReLU(),
    Linear(HIDDEN_DIM, OUTPUT_DIM)
])

input_data = np.random.randn(BATCH, INPUT_DIM).astype(np.float32)

print("Running NeonFlux Benchmark...")

for _ in range(WARMUP_ITERS):
    _ = model_neon.forward(input_data)

start = time.time()
for _ in range(N_ITERS):
    _ = model_neon.forward(input_data)
end = time.time()

neon_time = (end - start) * 1000.0 / N_ITERS
print(f"NeonFlux Avg Time: {neon_time:.3f} ms")
print("-" * 60)