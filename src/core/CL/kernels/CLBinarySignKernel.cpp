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
#include "arm_compute/core/CL/kernels/CLBinarySignKernel.h"

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
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *alpha)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), compute_binary_sign_shape(input->tensor_shape()));
    }
    
    // Checks performed when alpha is configured
    if(alpha != nullptr && alpha->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(alpha, 1, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(alpha->tensor_shape(), TensorShape(input->tensor_shape().total_size_upper(3)));
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, ITensorInfo *alpha)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_binary_sign_shape(input->tensor_shape())));
    if(alpha != nullptr)
    {
        auto_init_if_empty(*output, input->clone()->set_tensor_shape(TensorShape(input->tensor_shape().total_size_upper(3))));
    }

    constexpr unsigned int num_elems_read_per_iteration = 8;
    constexpr unsigned int num_elems_written_per_iteration = num_elems_read_per_iteration / 8;

    // Configure window
    Window win = calculate_max_window(*input, Steps(num_elems_read_per_iteration));
    
    // Update window and padding
    AccessWindowHorizontal input_access(input, 0, num_elems_read_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_written_per_iteration);
    // Alpha doesn't need any paddings
    bool                   window_changed = update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLBinarySignKernel::CLBinarySignKernel()
    : _input(nullptr), _output(nullptr), _alpha(nullptr)
{
}

void CLBinarySignKernel::configure(const ICLTensor *input, ICLTensor *output, ICLTensor *alpha)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), alpha ? alpha->info() : nullptr));

    _input  = input;
    _output = output;
    _alpha  = alpha;

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(_input->info()->tensor_shape()[0]));
    build_opts.add_option_if(alpha != nullptr, "-DCALCULATE_ALPHA");
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("binary_sign", build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), (alpha != nullptr) ? alpha->info() : nullptr);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    
    ICLKernel::configure_internal(win_config.second);
}

Status CLBinarySignKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *alpha)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, alpha));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), (alpha != nullptr) ? alpha->clone().get() : nullptr).first);
    return Status{};
}

void CLBinarySignKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_3D();
    
    unsigned int batch = 0;
    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        if(_alpha != nullptr)
        {
            add_1D_tensor_argument(idx, _alpha, slice);
            _kernel.setArg(idx++, batch++);
        }
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_3D(slice));
    
    // Normalize alpha values
    if(_alpha != nullptr)
    {
        _alpha->map(queue);
        for(size_t i =0; i < _alpha->info()->dimension(0); ++i)
        {
            *reinterpret_cast<float *>(_alpha->ptr_to_element(Coordinates(i))) /= _input->info()->tensor_shape().total_size_lower(3);
        }
        _alpha->unmap(queue);
    }
}
