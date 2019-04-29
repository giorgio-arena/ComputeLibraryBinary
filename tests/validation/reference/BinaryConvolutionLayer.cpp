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
#include "PadLayer.h"
#include "PoolingLayer.h"

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
    SimpleTensor<float> unbinarize(SimpleTensor<uint8_t> bin, TensorShape orig_shape)
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
                            unbin[dst_idx++] = static_cast<float>(positivity);
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
    
    const PaddingList padding { PaddingInfo(info.pad_left(), info.pad_right()), PaddingInfo(info.pad_top(), info.pad_bottom()) };
    SimpleTensor<float> padded_src = pad_layer(src, padding);
    
    std::tie(bin_weights, alpha, dummy) = binary_sign(weights);
    std::tie(bin_src, dummy, beta)      = binary_sign(padded_src);
    
    SimpleTensor<float> K = pooling_layer(beta, PoolingLayerInfo(PoolingType::AVG, weights.shape().x(), PadStrideInfo()));
    SimpleTensor<float> Ka { output_shape, DataType::F32 }; // 3D Ka
    
    for(size_t batch = 0; batch < Ka.shape().total_size_upper(3); ++batch)
    {
        for(size_t plane = 0; plane < Ka.shape().z(); ++plane)
        {
            for(size_t row = 0; row < Ka.shape().y(); ++row)
            {
                for(size_t col = 0; col < Ka.shape().x(); ++col)
                {                    
                    const size_t Ka_idx = coords2index(Ka.shape(), Coordinates(col, row, plane, batch));
                    const size_t K_idx  = coords2index(K.shape(), Coordinates(col, row, 0, batch));
                    
                    Ka[Ka_idx] = K[K_idx] * alpha[plane];
                }
            }
        }
    }
    
    
    SimpleTensor<float> dummy_bias { bias.shape(), DataType::F32 };
    for(int i = 0; i < dummy_bias.num_elements(); ++i)
    {
        dummy_bias[i] = 0.f;
    }
    
    // "Un-binarize" weights and src, perform convolution and apply normalization
    SimpleTensor<float> unbin_weights = unbinarize(bin_weights, weights.shape());
    SimpleTensor<float> unbin_src     = unbinarize(bin_src, padded_src.shape());
    SimpleTensor<float> binary_conv   = convolution_layer(unbin_src, unbin_weights, dummy_bias, output_shape, PadStrideInfo());
    
    SimpleTensor<float> dst { output_shape, DataType::F32 };
    for(int i = 0; i < dst.num_elements(); ++i)
    {
        dst[i] = binary_conv[i] * Ka[i] + bias[index2coords(dst.shape(), i)[2]];
    }
    
    return dst;
}
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
