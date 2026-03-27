"""Combines vecsum.cu and vecsum.cpp into submission.py for popcorn submit."""
import sys
from pathlib import Path

here = Path(__file__).parent
cuda_src = (here / "vecsum.cu").read_text()
cpp_src = (here / "vecsum.cpp").read_text()

GPUS = {
    "A100": ("A100", "sm_80"),
    "B200": ("B200", "sm_100"),
    "H100": ("H100", "sm_90"),
    "L4":   ("L4",   "sm_89"),
}

gpu = sys.argv[1] if len(sys.argv) > 1 else "B200"
if gpu not in GPUS:
    print(f"Unknown GPU: {gpu}. Options: {', '.join(GPUS)}")
    sys.exit(1)

gpu_name, arch = GPUS[gpu]

submission = f'''#!POPCORN leaderboard vectorsum_v2
#!POPCORN gpu {gpu_name}

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CUDA_SRC = """{cuda_src}"""

CPP_SRC = """{cpp_src}"""

module = load_inline(
    name='vecsum_module',
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=['vecsum'],
    verbose=True,
    extra_cuda_cflags=['-arch={arch}', '--use_fast_math'],
)

def custom_kernel(data: input_t) -> output_t:
    data, output = data
    return module.vecsum(data, output)[0]
'''

(here / "submission.py").write_text(submission)
print(f"submission.py built for {gpu_name} ({arch})")
