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
#ifndef __ARM_COMPUTE_NEBINARYSIGNKERNEL_H__
#define __ARM_COMPUTE_NEBINARYSIGNKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the binary sign operation kernel.
 *
 *   Each value of the input tensor gets stored as a 0 bit in the destination tensor
 *   if it is 0.f or a negative value, it gets stored as a 1 bit otherwise.
 *   Every 8 input values will be stored in one single value of the output (8 bits per uint8_t value).
 *   Optionally, this kernel also calculates the alpha 1D tensor containing the mean over absolute values of each 3D input block.
 *   Optionally, this kernel also calculates the beta 2D tensor containing the normalized mean over absolute values over channels.
 */
class NEBinarySignKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEBinarySignKernel";
    }
    /** Default constructor. */
    NEBinarySignKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBinarySignKernel(const NEBinarySignKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEBinarySignKernel &operator=(const NEBinarySignKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEBinarySignKernel(NEBinarySignKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEBinarySignKernel &operator=(NEBinarySignKernel &&) = default;
    /** Default destructor */
    ~NEBinarySignKernel() = default;
    
    /** Set the inputs and output images
     *
     * @param[in]  input  Source tensor. Data types supported: F32.
     * @param[out] output Destination tensor. Data types supported: U8.
     * @param[out] alpha  (Optional) Alpha tensor. It contains the mean over absolute values of each 3D input block.
     *                    Data types supported: F32.
     * @param[out] beta  (Optional) Beta tensor. It contains the mean over absolute values over channels.
     *                    Data types supported: F32.
     */
    void configure(const ITensor *input, ITensor *output, ITensor *alpha = nullptr, ITensor *beta = nullptr);
    
    /** Static function to check if given info will lead to a valid configuration of @ref NEBinarySignKernel
     *
     * @param[in] input  Source tensor. Data types supported: F32.
     * @param[in] output Destination tensor. Data types supported: U8.
     * @param[in] alpha  (Optional) Alpha tensor. It contains the mean over absolute values of each 3D input block.
     *                    Data types supported: F32.
     * @param[in] beta  (Optional) Beta tensor. It contains the mean over absolute values over channels.
     *                    Data types supported: F32.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *alpha = nullptr, const ITensorInfo *beta = nullptr);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input;  /**< Source tensor */
    ITensor       *_output; /**< Destination tensor */
    ITensor       *_alpha;  /**< Alpha tensor */
    ITensor       *_beta;   /**< Beta tensor */
    unsigned int   _num_elems_read_per_iteration;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLBINARYSIGNKERNEL_H__ */
