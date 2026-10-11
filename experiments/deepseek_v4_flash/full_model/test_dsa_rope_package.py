import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # noqa: F401


torch.npu.set_device(0)
x = torch.randn((1, 1, 1, 576), device="npu", dtype=torch.bfloat16)
expected = x.clone()
cos = torch.ones((1, 1, 1, 64), device="npu", dtype=torch.float32)
sin = torch.zeros_like(cos)

torch.ops._C_ascend.inplace_partial_rotary_mul(
    x,
    cos,
    sin,
    rotary_mode="interleave",
    partial_slice=[512, 576],
)
torch.npu.synchronize()
torch.testing.assert_close(x, expected, rtol=0, atol=0)
print("InplacePartialRotaryMul Ascend950 package smoke passed")
