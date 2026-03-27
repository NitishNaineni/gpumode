"""Local test for vecsum kernel."""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

here = Path(__file__).parent
CUDA_SRC = (here / "vecsum.cu").read_text()
CPP_SRC = (here / "vecsum.cpp").read_text()

module = load_inline(
    name='vecsum_module',
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=['vecsum'],
    verbose=True,
    extra_cuda_cflags=[],
)

SIZE = 52428800

data = torch.randn(SIZE, dtype=torch.float32, device='cuda')
output = torch.zeros(1, dtype=torch.float32, device='cuda')

result = module.vecsum(data, output)
expected = data.sum()

if torch.allclose(result, expected, rtol=1e-3):
    print("correctness: PASS")
else:
    print(f"correctness: FAIL (got {result.item()}, expected {expected.item()}, diff {abs(result.item() - expected.item())})")
