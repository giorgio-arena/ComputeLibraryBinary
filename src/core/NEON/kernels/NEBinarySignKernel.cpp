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
#include "arm_compute/core/NEON/kernels/NEBinarySignKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <arm_neon.h>

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
    Window win = calculate_max_window(*output, Steps(num_elems_written_per_iteration));
    
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

NEBinarySignKernel::NEBinarySignKernel()
    : _input(nullptr), _output(nullptr), _alpha(nullptr), _beta(nullptr), _num_elems_read_per_iteration(8)
{
}

void NEBinarySignKernel::configure(const ITensor *input, ITensor *output, ITensor *alpha, ITensor *beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), (alpha != nullptr) ? alpha->info() : nullptr,
                              (beta != nullptr) ? beta->info() : nullptr));

    _input  = input;
    _output = output;
    _alpha  = alpha;
    _beta  = beta;

    // Configure kernel window    
    auto win_config = validate_and_configure_window(input->info(), output->info(), (alpha != nullptr) ? alpha->info() : nullptr, (beta != nullptr) ? beta->info() : nullptr);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    
    INEKernel::configure(win_config.second);
}

Status NEBinarySignKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *alpha, const ITensorInfo *beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, alpha, beta));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), (alpha != nullptr) ? alpha->clone().get() : nullptr, (beta != nullptr) ? beta->clone().get() : nullptr).first);
    return Status{};
}

void NEBinarySignKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    
    const unsigned int src_width      = _input->info()->dimension(0);
    const unsigned int src_stride_x   = _input->info()->strides_in_bytes().x() * _num_elems_read_per_iteration;
    const unsigned int src_stride_y   = _input->info()->strides_in_bytes().y();
    const unsigned int src_stride_z   = _input->info()->strides_in_bytes().z();
    const unsigned int src_stride_w   = _input->info()->strides_in_bytes()[3];
    const bool         calc_alpha     = _alpha != nullptr;
    const bool         calc_beta      = _beta != nullptr;
    const unsigned int alpha_stride_x = calc_alpha ? _alpha->info()->strides_in_bytes().x() : 0;
    const unsigned int beta_stride_y  = calc_beta ? _beta->info()->strides_in_bytes().y() : 0;
    const unsigned int beta_stride_w  = calc_beta ? _beta->info()->strides_in_bytes()[3] : 0;
    
    const float32x4_t zerof = vdupq_n_f32(0);
    const uint32x4_t zero   = vdupq_n_u32(0);
    const uint32x4_t one    = vdupq_n_u32(1);
    
    Iterator input(_input, window);
    Iterator output(_output, window);
    Iterator alpha{};
    Iterator beta{};
    
    if(calc_alpha)
    {
        Window alpha_win = window;
        
        alpha_win.set(Window::DimY, Window::Dimension(0, 0, 0));
        alpha_win.set(Window::DimZ, Window::Dimension(0, 0, 0));
        alpha_win.set(3, Window::Dimension(0, 0, 0));
        
        alpha = Iterator(_alpha, alpha_win);
    }
    if(calc_beta)
    {
        Window beta_win = window;
        
        beta_win.set(Window::DimZ, Window::Dimension(0, 0, 0));
        beta_win.set(3, Window::Dimension(0, 0, 0));
        
        beta = Iterator(_beta, beta_win);
    }
    
    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto in_ptr  = reinterpret_cast<float *>(input.ptr() + id.x() * src_stride_x + id.y() * src_stride_y + id.z() * src_stride_z + id[3] * src_stride_w);
        const auto out_ptr = reinterpret_cast<uint8_t *>(output.ptr());
        
        // Load input values
        float32x4_t src_vals0 = vld1q_f32(in_ptr);
        float32x4_t src_vals1 = vld1q_f32(in_ptr + 4);
        
        // Don't consider padding
        const uint32x4_t offs0   = {0, 1, 2, 3};
        const uint32x4_t offs1   = {4, 5, 6, 7};
        const uint32x4_t x_pos0 = vaddq_u32(vdupq_n_u32(id.x() * _num_elems_read_per_iteration), offs0);
        const uint32x4_t x_pos1 = vaddq_u32(vdupq_n_u32(id.x() * _num_elems_read_per_iteration), offs1);
        src_vals0 = vbslq_u32(vreinterpretq_u32_f32(x_pos0 < vdupq_n_u32(src_width)), src_vals0, zerof);
        src_vals1 = vbslq_u32(vreinterpretq_u32_f32(x_pos1 < vdupq_n_u32(src_width)), src_vals1, zerof);
        
        // Calculate sign
        const uint32x4_t signs0  = vbslq_u32(vreinterpretq_u32_f32(src_vals0 > zerof), one, zero);
        const uint32x4_t signs1  = vbslq_u32(vreinterpretq_u32_f32(src_vals1 > zerof), one, zero);
        
        // Write to one byte
        unsigned char dst_val = 0;
        dst_val |= (signs0[0] << 7);
        dst_val |= (signs0[1] << 6);
        dst_val |= (signs0[2] << 5);
        dst_val |= (signs0[3] << 4);
        dst_val |= (signs1[0] << 3);
        dst_val |= (signs1[1] << 2);
        dst_val |= (signs1[2] << 1);
        dst_val |= (signs1[3]);
        
        // Write to tensor
        *out_ptr = dst_val;
        
        // Calculate alpha and beta
        if(calc_alpha || calc_beta)
        {
            const float32x4_t abs_vals0 = vabsq_f32(src_vals0);
            const float32x4_t abs_vals1 = vabsq_f32(src_vals1);
            
            if(calc_alpha)
            {
                const auto alpha_ptr = reinterpret_cast<float *>(alpha.ptr() + id[3] * alpha_stride_x);
                float sum = abs_vals0[0] + abs_vals0[1] + abs_vals0[2] + abs_vals0[3] +
                            abs_vals1[0] + abs_vals1[1] + abs_vals1[2] + abs_vals1[3];
                *alpha_ptr += sum;
            }
            if(calc_beta)
            {
                const auto beta_ptr = reinterpret_cast<float *>(beta.ptr() + id.x() * src_stride_x + id.y() * beta_stride_y + id[3] * beta_stride_w);
                vst1q_f32(beta_ptr, vaddq_f32(vld1q_f32(beta_ptr), abs_vals0));
                vst1q_f32(beta_ptr + 4, vaddq_f32(vld1q_f32(beta_ptr + 4), abs_vals1));
            }
        }
    }, output);
    
    // Normalize alpha values
    if(_alpha != nullptr)
    {
        for(size_t i = 0; i < _alpha->info()->dimension(0); ++i)
        {
            *reinterpret_cast<float *>(_alpha->ptr_to_element(Coordinates(i))) /= _input->info()->tensor_shape().total_size_lower(3);
        }
    }
    
    // Normalize beta values
    if(_beta != nullptr)
    {
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
    }
}
