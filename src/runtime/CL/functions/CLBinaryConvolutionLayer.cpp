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
#include "arm_compute/runtime/CL/functions/CLBinaryConvolutionLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <cmath>
#include <memory>
#include <tuple>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

CLBinaryConvolutionLayer::CLBinaryConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _binarize_input(), _binarize_weights(), _binary_convolution(), _normalize_beta(), _binarized_input(), _binarized_weights(), _alpha(), _beta(), _k(),
      _normalized_beta(), _is_prepared(false), _memory_manager(std::move(memory_manager))
{
}

void CLBinaryConvolutionLayer::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(CLBinaryConvolutionLayer::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr), output->info(), conv_info));
    
    TensorShape k_shape(weights->info()->dimension(0), weights->info()->dimension(1));
    auto_init_if_empty(*_k.info(), output->info()->clone()->set_tensor_shape(k_shape));
    
    _binarize_weights.configure(weights, &_binarized_weights, &_alpha);
    _binarize_input.configure(input, &_binarized_input, nullptr, &_beta);
    _normalize_beta.configure(&_beta, &_k, nullptr, &_normalized_beta, conv_info);
    
    _binarized_weights.allocator()->allocate();
    _binarized_input.allocator()->allocate();
    _alpha.allocator()->allocate();
    _beta.allocator()->allocate();
    _k.allocator()->allocate();
    _normalized_beta.allocator()->allocate();
}

Status CLBinaryConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    
    // TODO

    return Status{};
}

void CLBinaryConvolutionLayer::run()
{
    prepare();
    
    CLScheduler::get().enqueue(_binarize_input);
    
    _normalize_beta.run();
}

void CLBinaryConvolutionLayer::prepare()
{
    if(!_is_prepared)
    {
        CLScheduler::get().enqueue(_binarize_weights);
        
        // Fill k
        _k.map(CLScheduler::get().queue());
        
        const float norm_factor = 1.f / (_k.info()->dimension(0) * _k.info()->dimension(1));
        for(size_t y = 0; y < _k.info()->dimension(1); ++y)
        {
            for(size_t x = 0; x < _k.info()->dimension(0); ++x)
            {
                *reinterpret_cast<float *>(_k.ptr_to_element(Coordinates(x, y))) = norm_factor;
            }
        }
        
        _k.unmap(CLScheduler::get().queue());
        
        _is_prepared = true;
    }
}
