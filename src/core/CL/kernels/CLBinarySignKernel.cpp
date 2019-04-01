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
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *alpha, const ITensorInfo *beta)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), compute_binary_sign_shape(input->tensor_shape()));
    }
    
    // Checks performed when alpha is configured
    if(alpha != nullptr && alpha->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(alpha, 1, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(alpha->tensor_shape(), TensorShape(input->tensor_shape().total_size_upper(3)));
    }
    
    // Checks performed when beta is configured
    if(beta != nullptr && beta->total_size() != 0)
    {
        TensorShape expected_shape = input->tensor_shape();
        expected_shape.set(2, 1);
        
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(beta, 1, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(beta->tensor_shape(), expected_shape);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, ITensorInfo *alpha, ITensorInfo *beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_binary_sign_shape(input->tensor_shape())).set_data_type(DataType::U8));
    if(alpha != nullptr)
    {
        auto_init_if_empty(*alpha, input->clone()->set_tensor_shape(TensorShape(input->tensor_shape().total_size_upper(3))));
    }
    if(beta != nullptr)
    {
        TensorShape beta_shape = input->tensor_shape();
        beta_shape.set(2, 1);
        
        auto_init_if_empty(*beta, input->clone()->set_tensor_shape(beta_shape));
    }

    constexpr unsigned int num_elems_read_per_iteration = 8;
    constexpr unsigned int num_elems_written_per_iteration = num_elems_read_per_iteration / 8;

    // Configure window
    Window win = calculate_max_window(*input, Steps(num_elems_read_per_iteration));
    
    // Update window and padding
    AccessWindowHorizontal input_access(input, 0, num_elems_read_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_written_per_iteration);
    // Alpha doesn't need any padding
    AccessWindowHorizontal beta_access(beta, 0, (beta != nullptr) ? num_elems_written_per_iteration : 0);
    
    bool window_changed = update_window_and_padding(win, input_access, output_access, beta_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLBinarySignKernel::CLBinarySignKernel()
    : _input(nullptr), _output(nullptr), _alpha(nullptr), _beta(nullptr)
{
}

void CLBinarySignKernel::configure(const ICLTensor *input, ICLTensor *output, ICLTensor *alpha, ICLTensor *beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), (alpha != nullptr) ? alpha->info() : nullptr,
                              (beta != nullptr) ? beta->info() : nullptr));

    _input  = input;
    _output = output;
    _alpha  = alpha;
    _beta  = beta;

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(_input->info()->tensor_shape()[0]));
    build_opts.add_option_if(alpha != nullptr, "-DCALCULATE_ALPHA");
    build_opts.add_option_if(beta != nullptr, "-DCALCULATE_BETA");
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("binary_sign", build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), (alpha != nullptr) ? alpha->info() : nullptr, (beta != nullptr) ? beta->info() : nullptr);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    
    ICLKernel::configure_internal(win_config.second);
}

Status CLBinarySignKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *alpha, const ITensorInfo *beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, alpha, beta));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), (alpha != nullptr) ? alpha->clone().get() : nullptr, (beta != nullptr) ? beta->clone().get() : nullptr).first);
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
            // Initialize alpha value for this batch to 0
            _alpha->map(queue);
            *reinterpret_cast<float *>(_alpha->ptr_to_element(Coordinates(batch))) = 0;
            _alpha->unmap(queue);
            
            add_1D_tensor_argument(idx, _alpha, slice);
            _kernel.setArg(idx++, batch);
        }
        
        if(_beta != nullptr)
        {
            // Initialize beta values for this batch to 0
            _beta->map(queue);
            
            Coordinates coords(0, 0);
            if(_beta->info()->num_dimensions() > 3)
            {
                coords.set(2, 0);
                coords.set(3, batch);
            }
            
            for(size_t y = 0; y < _beta->info()->dimension(1); ++y)
            {
                coords.set(1, y);
                for(size_t x = 0; x < _beta->info()->dimension(0); ++x)
                {
                    coords.set(0, x);
                    *reinterpret_cast<float *>(_beta->ptr_to_element(coords)) = 0;
                }
            }
            
            _beta->unmap(queue);
            
            add_2D_tensor_argument(idx, _beta, slice);
        }
        enqueue(queue, *this, slice);
        
        ++batch;
    }
    while(window.slide_window_slice_3D(slice));
    
    queue.finish();
    
    // Normalize alpha values
    if(_alpha != nullptr)
    {
        _alpha->map(queue);
        for(size_t i = 0; i < _alpha->info()->dimension(0); ++i)
        {
            *reinterpret_cast<float *>(_alpha->ptr_to_element(Coordinates(i))) /= _input->info()->tensor_shape().total_size_lower(3);
        }
        _alpha->unmap(queue);
    }
    
    // Normalize beta values
    if(_beta != nullptr)
    {
        _beta->map(queue);

        size_t num_batches = 1;
        Coordinates coords(0, 0);
        if(_beta->info()->num_dimensions() > 3)
        {
            coords.set(2, 0);
            coords.set(3, 0);
            num_batches = _beta->info()->dimension(3);
        }

        for(size_t b = 0; b < num_batches; ++b)
        {
            for(size_t y = 0; y < _beta->info()->dimension(1); ++y)
            {
                coords.set(1, y);
                for(size_t x = 0; x < _beta->info()->dimension(0); ++x)
                {
                    coords.set(0, x);
                    *reinterpret_cast<float *>(_beta->ptr_to_element(coords)) /= _input->info()->dimension(2);
                }
            }

            coords.set(3, b + 1);
        }

        _beta->unmap(queue);
    }
}
