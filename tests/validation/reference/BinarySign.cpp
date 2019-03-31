/*
 * Copyright (c) 2019 ARM Limited.
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
#include "BinarySign.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
using namespace arm_compute::misc::shape_calculator;

std::tuple<SimpleTensor<uint8_t>, SimpleTensor<float>, SimpleTensor<float>> binary_sign(const SimpleTensor<float> &src)
{
    TensorShape beta_shape { src.shape() };
    beta_shape.set(2, 1);
    
    SimpleTensor<uint8_t> dst(compute_binary_sign_shape(src.shape()), DataType::U8);
    SimpleTensor<float>   alpha(TensorShape(src.shape().total_size_upper(3)), DataType::F32);
    SimpleTensor<float>   beta(beta_shape, DataType::F32);
    
    // Initialize alpha and beta to zero
    for(int i = 0; i < alpha.num_elements(); ++i)
    {
        alpha[i] = 0;
    }
    for(int i = 0; i < beta.num_elements(); ++i)
    {
        beta[i] = 0;
    }
    
    const size_t num_batches = src.shape().total_size_upper(3);
    const size_t block_sz    = src.shape().total_size_lower(3);
    const size_t plane_sz    = src.shape().total_size_lower(2);
    const size_t row_sz      = src.shape().x();
    
    const size_t beta_plane_sz = beta.shape().total_size_lower(2);
    
    size_t dst_idx = 0;    
    for(size_t batch = 0; batch < num_batches; ++batch)
    {
        for(size_t plane = 0; plane < src.shape().z(); ++plane)
        {
            for(size_t row = 0; row < src.shape().y(); ++row)
            {
                for(size_t col = 0; col < row_sz; col += 8)
                {
                    const size_t num_elems = std::min(row_sz - col, 8UL); // 8 or up to padding
                    size_t plane_pos = row * row_sz + col;
                    
                    uint8_t out_val  = 0;
                    for(size_t i = 0; i < num_elems; ++i, ++plane_pos)
                    {
                        float src_val   = src[batch * block_sz + plane_pos + plane * plane_sz];
                        alpha[batch]   += std::abs(src_val);
                        beta[batch * beta_plane_sz + plane_pos] += std::abs(src_val);

                        uint8_t is_positive = (src_val > 0); // 1 or 0 in byte-form
                        out_val |= (is_positive << (7 - i)); // Store in bit-form
                    }

                    dst[dst_idx++] = out_val;
                }
            }
        }
        
        alpha[batch] /= block_sz;
        for(size_t i = 0; i < plane_sz; ++i)
        {
            beta[batch * plane_sz + i] /= src.shape().z();
        }
    }

    return std::tie(dst, alpha, beta);
}
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
