# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import patch

import torch

import vllm_ascend.profiler.moe_profile as moe_profile


def test_profile_range_is_noop_when_disabled():
    hidden_states = torch.empty(2, 2048)
    topk_ids = torch.empty(2, 8, dtype=torch.int32)

    with patch.object(moe_profile, "_MOE_PROFILE_RANGES_ENABLED", False):
        context = moe_profile.moe_profile_range("catccos", "kernel_enqueue", hidden_states, topk_ids)

    assert isinstance(context, type(nullcontext()))


def test_profile_range_contains_backend_stage_and_real_shape():
    hidden_states = torch.empty(138, 2048)
    topk_ids = torch.empty(138, 8, dtype=torch.int32)

    with (
        patch.object(moe_profile, "_MOE_PROFILE_RANGES_ENABLED", True),
        patch.object(torch.profiler, "record_function") as record_function,
    ):
        moe_profile.moe_profile_range("native_mc2", "dispatch_enqueue", hidden_states, topk_ids)

    record_function.assert_called_once_with("vllm_ascend.moe.native_mc2.dispatch_enqueue[M=138,H=2048,topK=8]")


def test_sync_boundaries_require_ranges_and_sync_switch():
    with (
        patch.object(moe_profile, "_MOE_PROFILE_RANGES_ENABLED", True),
        patch.object(moe_profile, "_MOE_PROFILE_SYNC_BOUNDARIES_ENABLED", True),
    ):
        assert moe_profile.moe_profile_sync_boundaries_enabled()

    with patch.object(moe_profile, "_MOE_PROFILE_RANGES_ENABLED", False):
        assert not moe_profile.moe_profile_sync_boundaries_enabled()
