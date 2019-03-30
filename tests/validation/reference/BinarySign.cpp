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
#include "BitwiseAnd.h"

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

SimpleTensor<uint8_t> binary_sign(const SimpleTensor<float> &src)
{
    SimpleTensor<uint8_t> dst(compute_binary_sign_shape(src.shape()), DataType::U8);
    
    const size_t src_width = src.shape()[0];
    for(size_t dim = 0, dst_pos = 0; dim < src.shape().total_size_upper(1); ++dim)
    {
        for(size_t i = 0; i < src_width; i += 8, ++dst_pos)
        {
            size_t to_get = i + std::min((src_width - i), 8UL);

            uint8_t out_val = 0;
            for(size_t j = i, k = 0; j < to_get; ++j, ++k)
            {
                uint8_t val = (src[(dim * src_width) + j] > 0);                
                out_val |= (val << (7 - k));
            }
            
            dst[dst_pos] = out_val;
        }
    }

    return dst;
}
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
