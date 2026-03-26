"""Local test and benchmark for vecadd kernel."""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

here = Path(__file__).parent
CUDA_SRC = (here / "vecadd.cu").read_text()
CPP_SRC = (here / "vecadd.cpp").read_text()

module = load_inline(
    name='vecadd_module',
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=['vecadd'],
    verbose=True,
    extra_cuda_cflags=[],
)

SIZE = 16384

A = torch.randn(SIZE, SIZE, dtype=torch.float16, device='cuda')
B = torch.randn(SIZE, SIZE, dtype=torch.float16, device='cuda')
output = torch.empty_like(A)

# correctness
result = module.vecadd(A, B, output)
expected = A + B
assert torch.allclose(result, expected), f"FAIL: max diff {(result - expected).abs().max().item()}"
print("correctness: PASS")
