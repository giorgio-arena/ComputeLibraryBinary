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
#include "arm_compute/core/CL/kernels/CLBinaryConvolutionKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                          const PadStrideInfo &conv_info, const ITensorInfo *alpha, const ITensorInfo *beta, const Size2D &kernel_sz)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output, alpha, beta);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::U8);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(alpha, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(beta, 1, DataType::F32);

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *output,
                          const PadStrideInfo &conv_info, ITensorInfo *alpha, ITensorInfo *beta, const Size2D &kernel_sz)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output, alpha, beta);

    constexpr unsigned int num_elems_read_per_iteration = 1;
    constexpr unsigned int num_elems_written_per_iteration = num_elems_read_per_iteration * 8;

    // Configure window
    Window win = calculate_max_window(*input, Steps(num_elems_read_per_iteration));
    
    // Update window and padding
    AccessWindowHorizontal input_access(input, 0, num_elems_read_per_iteration);
    // Weights and biases don't need any padding
    AccessWindowHorizontal output_access(output, 0, num_elems_written_per_iteration);
    // Alpha doesn't need any padding
    AccessWindowHorizontal beta_access(beta, 0, (beta != nullptr) ? num_elems_written_per_iteration : 0);
    
    bool window_changed = update_window_and_padding(win, input_access, output_access, beta_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLBinaryConvolutionKernel::CLBinaryConvolutionKernel()
    : _input(nullptr),_weights(nullptr), _biases(nullptr), _output(nullptr), _alpha(nullptr), _beta(nullptr)
{
}

void CLBinaryConvolutionKernel::configure(ICLTensor *input, ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                                          const ICLTensor *alpha, ICLTensor *beta, const Size2D &kernel_sz)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output, alpha, beta);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr,
                                                  output->info(), conv_info, alpha->info(), beta->info(), kernel_sz));
    _input   = input;
    _weights = weights;
    _biases  = biases;
    _output  = output;
    _alpha   = alpha;
    _beta    = beta;

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option_if(biases != nullptr, "-DHAS_BIASES");
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("binary_sign", build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr,
                                                    output->info(), conv_info, alpha->info(), beta->info(), kernel_sz);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    
    ICLKernel::configure_internal(win_config.second);
}

Status CLBinaryConvolutionKernel::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                           const PadStrideInfo &conv_info, const ITensorInfo *alpha, const ITensorInfo *beta, const Size2D &kernel_sz)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output, alpha, beta);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, (biases != nullptr) ? biases : nullptr, output, conv_info, alpha, beta, kernel_sz));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), weights->clone().get(), (biases != nullptr) ? biases->clone().get() : nullptr,
                                                              output->clone().get(), conv_info, alpha->clone().get(), beta->clone().get(), kernel_sz).first);
    return Status{};
}

void CLBinaryConvolutionKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_3D();
    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_4D_tensor_argument(idx, _weights, slice);
        if(_biases != nullptr)
        {
            add_1D_tensor_argument(idx, _biases, slice);
        }
        add_3D_tensor_argument(idx, _output, slice);
        add_1D_tensor_argument(idx, _alpha, slice);
        add_2D_tensor_argument(idx, _beta, slice);        
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_3D(slice));
}
