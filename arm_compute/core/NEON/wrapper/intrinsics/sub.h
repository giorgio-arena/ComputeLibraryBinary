/*
 * Copyright (c) 2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef __ARM_COMPUTE_WRAPPER_SUB_H__
#define __ARM_COMPUTE_WRAPPER_SUB_H__

#include <arm_neon.h>

namespace arm_compute
{
namespace wrapper
{
#define VSUB_IMPL(stype, vtype, prefix, postfix)      \
    inline vtype vsub(const vtype &a, const vtype &b) \
    {                                                 \
        return prefix##_##postfix(a, b);              \
    }

VSUB_IMPL(uint8x8_t, uint8x8_t, vsub, u8)
VSUB_IMPL(int8x8_t, int8x8_t, vsub, s8)
VSUB_IMPL(uint16x4_t, uint16x4_t, vsub, u16)
VSUB_IMPL(int16x4_t, int16x4_t, vsub, s16)
VSUB_IMPL(uint32x2_t, uint32x2_t, vsub, u32)
VSUB_IMPL(int32x2_t, int32x2_t, vsub, s32)
VSUB_IMPL(uint64x1_t, uint64x1_t, vsub, u64)
VSUB_IMPL(int64x1_t, int64x1_t, vsub, s64)
VSUB_IMPL(float32x2_t, float32x2_t, vsub, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VSUB_IMPL(float16x4_t, float16x4_t, vsub, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

VSUB_IMPL(uint8x16_t, uint8x16_t, vsubq, u8)
VSUB_IMPL(int8x16_t, int8x16_t, vsubq, s8)
VSUB_IMPL(uint16x8_t, uint16x8_t, vsubq, u16)
VSUB_IMPL(int16x8_t, int16x8_t, vsubq, s16)
VSUB_IMPL(uint32x4_t, uint32x4_t, vsubq, u32)
VSUB_IMPL(int32x4_t, int32x4_t, vsubq, s32)
VSUB_IMPL(uint64x2_t, uint64x2_t, vsubq, u64)
VSUB_IMPL(int64x2_t, int64x2_t, vsubq, s64)
VSUB_IMPL(float32x4_t, float32x4_t, vsubq, f32)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
VSUB_IMPL(float16x8_t, float16x8_t, vsubq, f16)
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef vsub_IMPL
} // namespace wrapper
} // namespace arm_compute
#endif /* __ARM_COMPUTE_WRAPPER_SUB_H__ */
