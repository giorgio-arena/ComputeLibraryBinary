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
#include "BinaryConvolutionLayer.h"
#include "BinarySign.h"
#include "ConvolutionLayer.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "tests/SimpleTensorPrinter.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
    SimpleTensor<float> unbinarize(SimpleTensor<uint8_t> bin, TensorShape orig_shape, SimpleTensor<float> avg /* alpha or beta */, bool is_alpha)
    {
        SimpleTensor<float> unbin{ orig_shape, DataType::F32 };
    
        size_t dst_idx = 0;    
        for(size_t batch = 0; batch < bin.shape().total_size_upper(3); ++batch)
        {
            for(size_t plane = 0; plane < bin.shape().z(); ++plane)
            {
                for(size_t row = 0; row < bin.shape().y(); ++row)
                {
                    for(size_t col = 0; col < bin.shape().x(); ++col)
                    {
                        const Coordinates coords(col, row, plane, batch);
                        const size_t idx = coords2index(bin.shape(), coords);

                        const uint8_t bin_values = bin[idx];
                        const size_t  num_elems  = std::min(unbin.shape().x() - (col * 8UL), 8UL); // 8 or up to padding

                        for(size_t i = 0; i < num_elems; ++i)
                        {
                            uint8_t bin_val    = ((bin_values & (1 << (7 - i))) >> (7 - i));
                            int     positivity = bin_val == 0 ? -1 : 1;
                            
                            if(is_alpha)
                            {
                                unbin[dst_idx++] = positivity * avg[batch];
                            }
                            else
                            {
                                Coordinates beta_coords{ coords };
                                beta_coords.set(0, coords.x() + i);
                                beta_coords.set(2, 0);
                                                                
                                unbin[dst_idx++] = positivity * avg[coords2index(avg.shape(), beta_coords)];
                            }
                        }
                    }
                }
            }
        }
        
        return unbin;
    }
} // namespace
using namespace arm_compute::misc::shape_calculator;

SimpleTensor<float> binary_convolution(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &bias,
                                       const TensorShape &output_shape, const PadStrideInfo &info)
{
    SimpleTensor<uint8_t> bin_src{};
    SimpleTensor<uint8_t> bin_weights{};
    SimpleTensor<float>   alpha{};
    SimpleTensor<float>   beta{};
    SimpleTensor<float>   dummy{}; // For unused alpha/beta
    
    std::tie(bin_weights, alpha, dummy) = binary_sign(weights);
    std::tie(bin_src, dummy, beta)      = binary_sign(src);
    
    // Normalize beta
    SimpleTensor<float> k{ TensorShape(weights.shape().x(), weights.shape().y()), DataType::F32 };
    SimpleTensor<float> dummy_bias{ TensorShape(beta.shape()[3]), DataType::F32 };
    
    const float norm_factor = 1.f / (weights.shape().x() * weights.shape().y());
    for(int i = 0; i < k.num_elements(); ++i)
    {
        k[i] = norm_factor;
    }
    for(int i = 0; i < dummy_bias.num_elements(); ++i)
    {
        dummy_bias[i] = 0.f;
    }
    
    beta = convolution_layer(beta, k, dummy_bias, beta.shape(), info);
    
    // "Un-binarize" weights and src
    SimpleTensor<float> unbin_weights = unbinarize(bin_weights, weights.shape(), alpha, true);
    SimpleTensor<float> unbin_src     = unbinarize(bin_src, src.shape(), beta, false);
    
    // Perform approximate convolution    
    return convolution_layer(unbin_src, unbin_weights, bias, output_shape, info);
}
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
