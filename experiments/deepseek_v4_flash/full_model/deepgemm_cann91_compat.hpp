// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include <type_traits>
#include <c_api/asc_simd.h>

// CANN 9.2 names the hardware fields that CANN 9.1 accepts as integers.
// Preserve their encoding and fill the legacy post-processing defaults.
#if !__has_include(<c_api/defs/enum.h>)
enum class asc_load_l2_cache_mode : uint8_t {
    NORMAL_FIRST_VICTIM = 0, NORMAL_LAST_VICTIM = 1, NORMAL_PERSISTENT = 2,
    NOTALLOC_KEEP = 4, NOTALLOC_CLEAN = 5, NOTALLOC_DROP = 6
};
enum class asc_store_l2_cache_mode : uint8_t {
    NORMAL_FIRST_VICTIM = 0, NORMAL_LAST_VICTIM = 1, NORMAL_PERSISTENT = 2, NOTALLOC_CLEAN = 4
};
enum class asc_dual_dst_mode : uint8_t { DUAL_DST_DISABLE = 0, DUAL_DST_SPLIT_M = 1, DUAL_DST_SPLIT_N = 2 };
enum class asc_unit_flag_mode : uint8_t { DISABLE = 0, ENABLE_KEEP = 2, ENABLE_UPDATE = 3 };
enum class asc_relu_pre_mode : uint8_t { NONE = 0, NORMAL = 1, SCALAR = 2, VECTOR = 3 };
using asc_quant_mode = QuantMode_t;

// The old macros resolve unqualified names and collide with DeepGEMM's
// three-argument helpers in namespace deep_gemm.
#undef asc_sync_intra_arrive
#undef asc_sync_intra_wait
__aicore__ inline void dg_cann91_intra_arrive(pipe_t pipe, uint8_t id) {
    // Bisheng 9.1 requires a literal pipe enumerator at each builtin call.
    if ASCEND_IS_AIC { switch (pipe) {
        case PIPE_MTE1: set_intra_block(PIPE_MTE1, id); break;
        case PIPE_MTE2: set_intra_block(PIPE_MTE2, id); break;
        case PIPE_MTE3: set_intra_block(PIPE_MTE3, id); break;
        case PIPE_FIX: set_intra_block(PIPE_FIX, id); break;
        default: set_intra_block(PIPE_S, id); break;
    } } else { switch (pipe) {
        case PIPE_V: set_intra_block(PIPE_V, id); break;
        case PIPE_MTE2: set_intra_block(PIPE_MTE2, id); break;
        case PIPE_MTE3: set_intra_block(PIPE_MTE3, id); break;
        default: set_intra_block(PIPE_S, id); break;
    } }
}
__aicore__ inline void dg_cann91_intra_wait(pipe_t pipe, uint8_t id) {
    if ASCEND_IS_AIC { switch (pipe) {
        case PIPE_MTE1: wait_intra_block(PIPE_MTE1, id); break;
        case PIPE_MTE2: wait_intra_block(PIPE_MTE2, id); break;
        case PIPE_MTE3: wait_intra_block(PIPE_MTE3, id); break;
        case PIPE_FIX: wait_intra_block(PIPE_FIX, id); break;
        default: wait_intra_block(PIPE_S, id); break;
    } } else { switch (pipe) {
        case PIPE_V: wait_intra_block(PIPE_V, id); break;
        case PIPE_MTE2: wait_intra_block(PIPE_MTE2, id); break;
        case PIPE_MTE3: wait_intra_block(PIPE_MTE3, id); break;
        default: wait_intra_block(PIPE_S, id); break;
    } }
}
#define asc_sync_intra_arrive(pipe, sync_id) dg_cann91_intra_arrive((pipe), (sync_id))
#define asc_sync_intra_wait(pipe, sync_id) dg_cann91_intra_wait((pipe), (sync_id))

constexpr std::integral_constant<int, 0> ASC_POSITION_EVEN;
constexpr std::integral_constant<int, 0> ASC_DISPERSE_FIRST_QUARTER;
__simd_callee__ inline void asc_float2bfloat16_rn(
    vector_bfloat16_t& dst, vector_float src, vector_bool mask, std::integral_constant<int, 0>) {
    asc_float2bfloat16_rn(dst, src, mask);
}
__simd_callee__ inline void asc_float2e4m3_rn_sat(
    vector_fp8_e4m3fn_t& dst, vector_float src, vector_bool mask, std::integral_constant<int, 0>) {
    asc_float2e4m3_rn_sat(dst, src, mask);
}
__simd_callee__ inline void asc_storealign_pack_quarter(
    __ubuf__ uint32_t* dst, vector_uint32_t src, vector_bool mask) {
    vsts(src, dst, 0, PK4_B32, mask);
}
template <typename T>
__aicore__ inline T asc_load_dev(__gm__ T* src) {
    return static_cast<T>(ld_dev(reinterpret_cast<__gm__ std::make_unsigned_t<T>*>(src), 0));
}

template <typename T>
__aicore__ inline void asc_copy_gm2l1_nd2nz(
    __cbuf__ T* dst, __gm__ T* src, uint64_t stride, asc_load_l2_cache_mode cache,
    uint16_t n, uint16_t d, uint64_t batch_stride, bool small) {
    asc_copy_gm2l1_nd2nz(dst, src, stride, static_cast<uint8_t>(cache), n, d, batch_stride, small);
}
template <typename T>
__aicore__ inline void asc_copy_gm2l1_dn2nz(
    __cbuf__ T* dst, __gm__ T* src, uint64_t stride, asc_load_l2_cache_mode cache,
    uint16_t n, uint16_t d, uint64_t batch_stride, bool small) {
    asc_copy_gm2l1_dn2nz(dst, src, stride, static_cast<uint8_t>(cache), n, d, batch_stride, small);
}

template <typename T>
__aicore__ inline void asc_copy_ub2gm_align(
    __gm__ T* dst, __ubuf__ T* src, uint16_t rows, uint32_t bytes,
    asc_store_l2_cache_mode cache, uint64_t dst_stride, uint32_t src_stride) {
    asc_copy_ub2gm_align(dst, src, rows, bytes, static_cast<uint8_t>(cache), dst_stride, src_stride);
}

template <typename T>
__aicore__ inline void asc_copy_gm2ub_align(
    __ubuf__ T* dst, __gm__ T* src, uint16_t rows, uint32_t bytes,
    uint8_t left, uint8_t right, bool pad, asc_load_l2_cache_mode cache,
    uint64_t src_stride, uint32_t dst_stride) {
    asc_copy_gm2ub_align(dst, src, rows, bytes, left, right, pad, static_cast<uint8_t>(cache), src_stride, dst_stride);
}

template <typename T>
__aicore__ inline void asc_copy_gm2l1_align(
    __cbuf__ T* dst, __gm__ T* src, uint16_t rows, uint32_t bytes,
    uint8_t left, uint8_t right, bool pad, asc_load_l2_cache_mode cache,
    uint64_t src_stride, uint32_t dst_stride) {
    asc_copy_gm2l1_align(dst, src, rows, bytes, left, right, pad, static_cast<uint8_t>(cache), src_stride, dst_stride);
}

template <typename T, typename S>
__aicore__ inline void asc_copy_l0c2gm(
    __gm__ T* dst, __cc__ S* src, uint16_t n, uint16_t m, uint32_t dst_stride, uint16_t src_stride,
    asc_store_l2_cache_mode cache, asc_unit_flag_mode unit, asc_quant_mode quant, asc_relu_pre_mode relu,
    bool split, bool nz2nd, bool nz2dn, bool clip) {
    asc_copy_l0c2gm(dst, src, n, m, dst_stride, src_stride, static_cast<uint8_t>(cache), clip,
        static_cast<uint8_t>(unit), static_cast<uint64_t>(quant), static_cast<uint8_t>(relu),
        split, nz2nd, 0, 0, false, 0, false, false, false, nz2dn);
}

template <typename T, typename S>
__aicore__ inline void asc_copy_l0c2ub(
    __ubuf__ T* dst, __cc__ S* src, uint16_t n, uint16_t m, uint32_t dst_stride, uint16_t src_stride,
    uint32_t subblock, asc_dual_dst_mode dual, asc_unit_flag_mode unit, asc_quant_mode quant,
    asc_relu_pre_mode relu, bool split, bool nz2nd, bool nz2dn, bool clip) {
    // Keep T: the legacy void* overload emits S4, not BF16/FP8.
    asc_copy_l0c2ub(dst, src, n, m, dst_stride, src_stride,
        static_cast<uint8_t>(dual), static_cast<bool>(subblock), clip, static_cast<uint8_t>(unit),
        static_cast<uint64_t>(quant), static_cast<uint8_t>(relu), split, nz2nd,
        0, 0, false, 0, false, false, false, nz2dn);
}
#endif

__simd_callee__ inline void dg_storealign_postupdate(
    __ubuf__ uint8_t*& dst, vector_uint8_t src,
    uint16_t block_stride, uint16_t repeat_stride, vector_bool mask) {
#if __has_include(<c_api/defs/enum.h>)
    asc_storealign_postupdate(dst, src, block_stride, repeat_stride, mask);
#else
    // CANN9.1's five-argument public wrapper takes dst by value. Forward
    // directly to the same instruction so POST_UPDATE reaches the caller.
    vsstb(src, dst, (static_cast<uint32_t>(block_stride) << 16) | repeat_stride, mask, POST_UPDATE);
#endif
}
