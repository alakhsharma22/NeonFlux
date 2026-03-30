import time
import numpy as np
import torch

BATCH = 64
INPUT_DIM = 128
HIDDEN_DIM = 1024
OUTPUT_DIM = 10
N_ITERS = 100
WARMUP_ITERS = 10

print(f"Benchmarking PyTorch CPU MLP: [Input({INPUT_DIM}) -> Linear({HIDDEN_DIM}) -> ReLU -> Linear({HIDDEN_DIM}) -> ReLU -> Linear({OUTPUT_DIM})]")
print(f"Batch Size: {BATCH}")
print("-" * 60)

model_torch = torch.nn.Sequential(
    torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
).to("cpu")

input_data = np.random.randn(BATCH, INPUT_DIM).astype(np.float32)
input_tensor = torch.from_numpy(input_data)

print("Running PyTorch Benchmark...")

with torch.no_grad():
    for _ in range(WARMUP_ITERS):
        _ = model_torch(input_tensor)

start = time.time()
with torch.no_grad():
    for _ in range(N_ITERS):
        _ = model_torch(input_tensor)
end = time.time()

torch_time = (end - start) * 1000.0 / N_ITERS
print(f"PyTorch Avg Time: {torch_time:.3f} ms")
print("-" * 60)