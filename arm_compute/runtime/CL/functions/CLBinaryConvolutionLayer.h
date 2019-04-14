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
#ifndef __ARM_COMPUTE_CLBINARYCONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_CLBINARYCONVOLUTIONLAYER_H__

#include "arm_compute/core/CL/kernels/CLBinarySignKernel.h"
#include "arm_compute/core/CL/kernels/CLBinaryConvolutionKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLPadLayer.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;

/** Basic function to compute the binary convolution layer. This function calls the following OpenCL kernels/functions: TODO
 */
class CLBinaryConvolutionLayer : public IFunction
{   
public:
    /** Default constructor */
    CLBinaryConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLBinaryConvolutionLayer(const CLBinaryConvolutionLayer &) = delete;
    /** Default move constructor */
    CLBinaryConvolutionLayer(CLBinaryConvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLBinaryConvolutionLayer &operator=(const CLBinaryConvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLBinaryConvolutionLayer &operator=(CLBinaryConvolutionLayer &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: F32.
     * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                              Data type supported: Should match @p input data type
     * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * 
     */
    void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLBinaryConvolutionLayer
     *
     * @param[in] input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: F32.
     * @param[in] weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in] biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported:Same as @p input.
     * @param[in] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in] conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info);
    
    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    CLPadLayer                      _pad_input;
    CLBinarySignKernel              _binarize_input;
    CLBinarySignKernel              _binarize_weights;
    CLBinaryConvolutionKernel       _binary_convolution;
    CLConvolutionLayer              _normalize_beta;
    CLTensor                        _padded_input;
    CLTensor                        _binarized_input;
    CLTensor                        _binarized_weights;
    CLTensor                        _alpha;
    CLTensor                        _beta;
    CLTensor                        _k;
    CLTensor                        _normalized_beta;
    bool                            _is_prepared;
    std::shared_ptr<IMemoryManager> _memory_manager;
};
}
#endif /* __ARM_COMPUTE_CLCONVOLUTIONLAYER_H__ */
