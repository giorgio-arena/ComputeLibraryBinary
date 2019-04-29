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
#include "arm_compute/core/NEON/kernels/NEBinaryConvolutionKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <arm_neon.h>
#include <bitset>

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

    constexpr unsigned int num_elems_read_per_iteration    = 1;
    const     unsigned int num_elems_written_per_iteration = num_elems_read_per_iteration * 8;

    // Configure window
    Window win = calculate_max_window(*output, Steps(num_elems_written_per_iteration));
    
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

NEBinaryConvolutionKernel::NEBinaryConvolutionKernel()
    : _input(nullptr),_weights(nullptr), _biases(nullptr), _output(nullptr), _alpha(nullptr), _beta(nullptr), _num_elems_written_per_iteration(8)
{
}

void NEBinaryConvolutionKernel::configure(ITensor *input, ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                                          const ITensor *alpha, ITensor *beta, const Size2D &kernel_sz)
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
    
    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr,
                                                    output->info(), conv_info, alpha->info(), beta->info(), kernel_sz);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    
    INEKernel::configure(win_config.second);
}

Status NEBinaryConvolutionKernel::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                           const PadStrideInfo &conv_info, const ITensorInfo *alpha, const ITensorInfo *beta, const Size2D &kernel_sz)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output, alpha, beta);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, (biases != nullptr) ? biases : nullptr, output, conv_info, alpha, beta, kernel_sz));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), weights->clone().get(), (biases != nullptr) ? biases->clone().get() : nullptr,
                                                              output->clone().get(), conv_info, alpha->clone().get(), beta->clone().get(), kernel_sz).first);
    return Status{};
}

void NEBinaryConvolutionKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    
    const unsigned int src_stride_x     = _input->info()->strides_in_bytes().x();
    const unsigned int src_stride_y     = _input->info()->strides_in_bytes().y();
    const unsigned int src_stride_z     = _input->info()->strides_in_bytes().z();
    const unsigned int src_stride_w     = _input->info()->strides_in_bytes()[3];
    const unsigned int weights_stride_y = _weights->info()->strides_in_bytes().y();
    const unsigned int weights_stride_z = _weights->info()->strides_in_bytes().z();
    const unsigned int weights_stride_w = _weights->info()->strides_in_bytes()[3];
    const unsigned int alpha_stride_x   = _alpha->info()->strides_in_bytes().x();
    const unsigned int beta_stride_x    = _beta->info()->strides_in_bytes().x();
    const unsigned int beta_stride_y    = _beta->info()->strides_in_bytes().y();
    const unsigned int beta_stride_w    = _beta->info()->strides_in_bytes()[3];
    const bool         has_bias         = _biases != nullptr;
    const unsigned int biases_stride_x  = has_bias ? _biases->info()->strides_in_bytes().x() : 0;
    const unsigned int weights_depth    = _weights->info()->dimension(2);
    
    Window alpha_win = window;
    Window beta_win  = window;
    alpha_win.set(Window::DimY, Window::Dimension(0, 0, 0));
    alpha_win.set(Window::DimZ, Window::Dimension(0, 0, 0));
    alpha_win.set(3, Window::Dimension(0, 0, 0));
    beta_win.set(Window::DimZ, Window::Dimension(0, 0, 0));
    beta_win.set(3, Window::Dimension(0, 0, 0));
    
    Iterator input(_input, window);
    Iterator output(_output, window);
    Iterator weights(_weights, window);
    Iterator biases(_biases, alpha_win); // TODO check nullptr
    Iterator alpha(_alpha, alpha_win);
    Iterator beta(_beta, beta_win);
    
    execute_window_loop(window, [&](const Coordinates & id)
    {
        auto       in_ptr      = input.ptr() + (id.x() / _num_elems_written_per_iteration) * src_stride_x + id.y() * src_stride_y + id[3] * src_stride_w; // TODO try remove cast in binarysign
        const auto out_ptr     = reinterpret_cast<float *>(output.ptr());
        auto       weights_ptr = weights.ptr() + id.z() * weights_stride_w;
        const auto alpha_ptr   = reinterpret_cast<float *>(alpha.ptr() + id.z() * alpha_stride_x);
        const auto beta_ptr    = reinterpret_cast<float *>(beta.ptr() + id.x() * beta_stride_x + id.y() * beta_stride_y + id[3] * beta_stride_w);
        
        float32x4_t out_vals0 = vdupq_n_f32(0);
        float32x4_t out_vals1 = vdupq_n_f32(0);
    
        for(unsigned int d = 0; d < weights_depth; ++d)
        {
            uint16x4_t src_vals;
            uint16x4_t next_byte;
            uint16x4_t tmp = vdup_n_u16(0);
            uint16x4_t weights_vals;
            
            src_vals[0] = *(in_ptr + 0 * src_stride_y);
            src_vals[1] = *(in_ptr + 1 * src_stride_y);
            src_vals[2] = *(in_ptr + 2 * src_stride_y);
            
            next_byte[0] = *(in_ptr + 0 * src_stride_y + src_stride_x);
            next_byte[1] = *(in_ptr + 1 * src_stride_y + src_stride_x);
            next_byte[2] = *(in_ptr + 2 * src_stride_y + src_stride_x);
            
            tmp = vshl_s16(vshl_s16(src_vals, vdup_n_u16(-1)), vdup_n_u16(7));
            tmp = vorr_u16(tmp, vshl_s16(vshl_u16(src_vals, vdup_n_u16(7)), vdup_n_u16(-1)));
            tmp = vorr_u16(tmp, vshl_u16(vshl_s16(next_byte, vdup_n_u16(-7)), vdup_n_u16(5)));
            tmp = vorr_u16(tmp, vshl_u16(vshl_s16(vshl_u16(next_byte, vdup_n_u16(1)), vdup_n_u16(-7)), vdup_n_u16(4)));
            
            weights_vals[0] = *(weights_ptr + 0 * weights_stride_y);
            weights_vals[1] = *(weights_ptr + 1 * weights_stride_y);
            weights_vals[2] = *(weights_ptr + 2 * weights_stride_y);
            
            weights_vals = vshl_u16(vshl_s16(weights_vals,vdup_n_u16(-5)), vdup_n_u16(5));
            
            uint16x8_t mask = vdupq_n_u16(255 >> 3);
            mask[1] = (mask[0] >> 1) | 128;
            mask[2] = (mask[1] >> 1) | 128;
            mask[3] = (mask[2] >> 1) | 128;
            mask[4] = (mask[3] >> 1) | 128;
            mask[5] = (mask[4] >> 1) | 128;
            
            out_vals0[0] += std::bitset<8>(~(src_vals[0] ^ ((~src_vals[0] & mask[0]) | weights_vals[0]))).count();
            out_vals0[0] += std::bitset<8>(~(src_vals[1] ^ ((~src_vals[1] & mask[0]) | weights_vals[1]))).count();
            out_vals0[0] += std::bitset<8>(~(src_vals[2] ^ ((~src_vals[2] & mask[0]) | weights_vals[2]))).count();

            out_vals1[2] += std::bitset<8>(~(tmp[0] ^ ((~tmp[0] & mask[0]) | weights_vals[0]))).count();
            out_vals1[2] += std::bitset<8>(~(tmp[1] ^ ((~tmp[1] & mask[0]) | weights_vals[1]))).count();
            out_vals1[2] += std::bitset<8>(~(tmp[2] ^ ((~tmp[2] & mask[0]) | weights_vals[2]))).count();
            weights_vals >>= 1;

            out_vals0[1] += std::bitset<8>(~(src_vals[0] ^ ((~src_vals[0] & mask[1]) | weights_vals[0]))).count();
            out_vals0[1] += std::bitset<8>(~(src_vals[1] ^ ((~src_vals[1] & mask[1]) | weights_vals[1]))).count();
            out_vals0[1] += std::bitset<8>(~(src_vals[2] ^ ((~src_vals[2] & mask[1]) | weights_vals[2]))).count();

            out_vals1[3] += std::bitset<8>(~(tmp[0] ^ ((~tmp[0] & mask[1]) | weights_vals[0]))).count();
            out_vals1[3] += std::bitset<8>(~(tmp[1] ^ ((~tmp[1] & mask[1]) | weights_vals[1]))).count();
            out_vals1[3] += std::bitset<8>(~(tmp[2] ^ ((~tmp[2] & mask[1]) | weights_vals[2]))).count();
            weights_vals >>= 1;

            out_vals0[2] += std::bitset<8>(~(src_vals[0] ^ ((~src_vals[0] & mask[2]) | weights_vals[0]))).count();
            out_vals0[2] += std::bitset<8>(~(src_vals[1] ^ ((~src_vals[1] & mask[2]) | weights_vals[1]))).count();
            out_vals0[2] += std::bitset<8>(~(src_vals[2] ^ ((~src_vals[2] & mask[2]) | weights_vals[2]))).count();
            weights_vals >>= 1;

            out_vals0[3] += std::bitset<8>(~(src_vals[0] ^ ((~src_vals[0] & mask[3]) | weights_vals[0]))).count();
            out_vals0[3] += std::bitset<8>(~(src_vals[1] ^ ((~src_vals[1] & mask[3]) | weights_vals[1]))).count();
            out_vals0[3] += std::bitset<8>(~(src_vals[2] ^ ((~src_vals[2] & mask[3]) | weights_vals[2]))).count();
            weights_vals >>= 1;

            out_vals1[0] += std::bitset<8>(~(src_vals[0] ^ ((~src_vals[0] & mask[4]) | weights_vals[0]))).count();
            out_vals1[0] += std::bitset<8>(~(src_vals[1] ^ ((~src_vals[1] & mask[4]) | weights_vals[1]))).count();
            out_vals1[0] += std::bitset<8>(~(src_vals[2] ^ ((~src_vals[2] & mask[4]) | weights_vals[2]))).count();
            weights_vals >>= 1;

            out_vals1[1] += std::bitset<8>(~(src_vals[0] ^ ((~src_vals[0] & mask[5]) | weights_vals[0]))).count();
            out_vals1[1] += std::bitset<8>(~(src_vals[1] ^ ((~src_vals[1] & mask[5]) | weights_vals[1]))).count();
            out_vals1[1] += std::bitset<8>(~(src_vals[2] ^ ((~src_vals[2] & mask[5]) | weights_vals[2]))).count();  

            in_ptr      += src_stride_z;
            weights_ptr += weights_stride_z;
        }
        
        float32x4_t tot_elems  = vdupq_n_f32(9 * weights_depth);
        float32x4_t beta_vals0 = vld1q_f32(beta_ptr);
        float32x4_t beta_vals1 = vld1q_f32(beta_ptr + 4);
        float32x4_t alpha_vals = vdupq_n_f32(*alpha_ptr);

        out_vals0 = vmulq_f32(vsubq_f32(vmulq_f32(vdupq_n_f32(2), out_vals0), tot_elems), vmulq_f32(alpha_vals, beta_vals0));
        out_vals1 = vmulq_f32(vsubq_f32(vmulq_f32(vdupq_n_f32(2), out_vals1), tot_elems), vmulq_f32(alpha_vals, beta_vals1));
        
        if(has_bias)
        {
            const auto biases_ptr = reinterpret_cast<float *>(biases.ptr() + id.z() * biases_stride_x);
            float32x4_t biases_val = vdupq_n_f32(*biases_ptr);
            
            out_vals0 = vaddq_f32(out_vals0, biases_val);
            out_vals1 = vaddq_f32(out_vals1, biases_val);
        }
        
        vst1q_f32(out_ptr, out_vals0);
        vst1q_f32(out_ptr + 4, out_vals1);
    }, output);
}
