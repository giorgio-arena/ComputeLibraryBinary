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
#ifndef __ARM_COMPUTE_CLBINARYCONVOLUTIONKERNEL_H__
#define __ARM_COMPUTE_CLBINARYCONVOLUTIONKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the binary convolution (XNOR popcount + normalization) operation kernel. */
class CLBinaryConvolutionKernel : public ICLKernel
{
public:
    /** Default constructor. */
    CLBinaryConvolutionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLBinaryConvolutionKernel(const CLBinaryConvolutionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLBinaryConvolutionKernel &operator=(const CLBinaryConvolutionKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLBinaryConvolutionKernel(CLBinaryConvolutionKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLBinaryConvolutionKernel &operator=(CLBinaryConvolutionKernel &&) = default;
    /** Default destructor */
    ~CLBinaryConvolutionKernel() = default;
    
    /** Set the inputs and output images
     *
     * @param[in]  input     Source tensor (binarized). Data types supported: U8.
     * @param[in]  weights   Weights tensor (binarized). Data type supported: Same as @p input.
     * @param[in]  biases    Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: F32
     * @param[out] output    Destination tensor. Data types supported: F32.
     * @param[in]  conv_info Contains padding and stride information described in @ref PadStrideInfo.
     * @param[out] alpha     Alpha tensor. Mean over absolute values of each original 3D weight.
     *                       Calculated using @ref CLBinarySignKernel. Data types supported: F32.
     * @param[out] beta      Beta tensor. Normalized mean over absolute values over channels of the original input.
     *                       Calculated using @ref CLBinarySignKernel. Data types supported: F32.
     * @param[in]  kernel_sz Size of the original (non-binirized) kernel
     * 
     */
    void configure(ICLTensor *input, ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   const ICLTensor *alpha, ICLTensor *beta, const Size2D &kernel_sz);
    
    /** Static function to check if given info will lead to a valid configuration of @ref CLBinaryConvolutionKernel
     *
     * @param[in] input     Source tensor (binarized). Data types supported: U8.
     * @param[in] weights   Weights tensor (binarized). Data type supported: Same as @p input.
     * @param[in] biases    Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: F32
     * @param[in] output    Destination tensor. Data types supported: F32.
     * @param[in] conv_info Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] alpha     Alpha tensor. Mean over absolute values of each original 3D weight.
     *                      Calculated using @ref CLBinarySignKernel. Data types supported: F32.
     * @param[in] beta      Beta tensor. Normalized mean over absolute values over channels of the original input.
     *                      Calculated using @ref CLBinarySignKernel. Data types supported: F32.
     * @param[in] kernel_sz Size of the original (non-binirized) kernel
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                           const PadStrideInfo &conv_info, const ITensorInfo *alpha, const ITensorInfo *beta, const Size2D &kernel_sz);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    ICLTensor       *_input;   /**< Source tensor */
    ICLTensor       *_weights; /**< Weights tensor */
    const ICLTensor *_biases;  /**< Biases tensor */
    ICLTensor       *_output;  /**< Destination tensor */
    const ICLTensor *_alpha;   /**< Alpha tensor */
    const ICLTensor *_beta;    /**< Beta tensor */
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLBINARYCONVOLUTIONKERNEL_H__ */
