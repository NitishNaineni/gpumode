"""Combines vecadd.cu and vecadd.cpp into submission.py for popcorn submit."""
from pathlib import Path

here = Path(__file__).parent
cuda_src = (here / "vecadd.cu").read_text()
cpp_src = (here / "vecadd.cpp").read_text()

submission = f'''#!POPCORN leaderboard vectoradd_v2
#!POPCORN gpu A100

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CUDA_SRC = """{cuda_src}"""

CPP_SRC = """{cpp_src}"""

module = load_inline(
    name='vecadd_module',
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=['vecadd'],
    verbose=True,
    extra_cuda_cflags=['-arch=sm_80', '--use_fast_math'],
)

def custom_kernel(data: input_t) -> output_t:
    A, B, output = data
    return module.vecadd(A, B, output)
'''

(here / "submission.py").write_text(submission)
print("submission.py built successfully")
