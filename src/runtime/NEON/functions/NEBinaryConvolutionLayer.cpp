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
#include "arm_compute/runtime/NEON/functions/NEBinaryConvolutionLayer.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <cmath>
#include <memory>
#include <tuple>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

NEBinaryConvolutionLayer::NEBinaryConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _pad_input(), _binarize_input(), _binarize_weights(), _binary_convolution(), _normalize_beta(), _padded_input(), _binarized_input(),
      _binarized_weights(), _alpha(), _beta(), _K(), _is_prepared(false), _memory_manager(std::move(memory_manager))
{
}

void NEBinaryConvolutionLayer::configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(NEBinaryConvolutionLayer::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info));
    
    const PaddingList padding { PaddingInfo(conv_info.pad_left(), conv_info.pad_right()), PaddingInfo(conv_info.pad_top(), conv_info.pad_bottom()) };
    _pad_input.configure(input, &_padded_input, padding, PixelValue(0));
    _binarize_weights.configure(weights, &_binarized_weights, &_alpha);
    _binarize_input.configure(&_padded_input, &_binarized_input, nullptr, &_beta);
    _normalize_beta.configure(&_beta, &_K, PoolingLayerInfo(PoolingType::AVG, weights->info()->dimension(0), PadStrideInfo()));
    _binary_convolution.configure(&_binarized_input, &_binarized_weights, biases, output, PadStrideInfo(), &_alpha, &_K,
                                  Size2D(weights->info()->dimension(0), weights->info()->dimension(1)));
    
    _padded_input.allocator()->allocate();
    _binarized_weights.allocator()->allocate();
    _binarized_input.allocator()->allocate();
    _alpha.allocator()->allocate();
    _beta.allocator()->allocate();
    _K.allocator()->allocate();
}

Status NEBinaryConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    
    // TODO

    return Status{};
}

void NEBinaryConvolutionLayer::run()
{
    prepare();
    
    _pad_input.run();
    
    NEScheduler::get().schedule(&_binarize_input, Window::DimX);
    
    _normalize_beta.run();
    
    NEScheduler::get().schedule(&_binary_convolution, Window::DimX);
}

void NEBinaryConvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        NEScheduler::get().schedule(&_binarize_weights, Window::DimX);        
        _is_prepared = true;
    }
}
