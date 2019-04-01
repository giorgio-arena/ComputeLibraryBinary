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
#include "helpers.h"

/** This function computes the binary convolution operation.
 *
 * @param[in]  src_ptr                               Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                          Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                            src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                          Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                            src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                          Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                            src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes     The offset of the first element in the source tensor
 * @param[in]  weights_ptr                           Pointer to the weights tensor. Supported data types: U8
 * @param[in]  weights_stride_x                      Stride of the weights tensor in X dimension (in bytes)
 * @param[in]  weights_step_x                        weights_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  weights_stride_y                      Stride of the weights tensor in Y dimension (in bytes)
 * @param[in]  weights_step_y                        weights_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  weights_stride_z                      Stride of the weights tensor in Z dimension (in bytes)
 * @param[in]  weights_step_z                        weights_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  weights_stride_w                      Stride of the weights tensor in W dimension (in bytes)
 * @param[in]  weights_step_w                        weights_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  weights_offset_first_element_in_bytes The offset of the first element in the weights tensor
 * @param[in]  biases_ptr                            (Optional) Pointer to the biases tensor. Supported data types: U8
 * @param[in]  biases_stride_x                       (Optional) Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                         (Optional) biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes  The offset of the first element in the biases tensor
 * @param[out] dst_ptr                               Pointer to the destination tensor. Supported data types: U8
 * @param[in]  dst_stride_x                          Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                            dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                            dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                            dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in]  alpha_ptr                             Pointer to the alpha tensor. Supported data types: F32
 * @param[in]  alpha_stride_x                        Stride of the alpha tensor in X dimension (in bytes)
 * @param[in]  alpha_step_x                          alpha_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  alpha_offset_first_element_in_bytes   The offset of the first element in the alpha tensor
 * @param[in]  beta_ptr                              Pointer to the beta tensor. Supported data types: F32
 * @param[in]  beta_stride_x                         Stride of the beta tensor in X dimension (in bytes)
 * @param[in]  beta_step_x                           beta_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  beta_stride_y                         Stride of the beta tensor in Y dimension (in bytes)
 * @param[in]  beta_step_y                           beta_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  beta_offset_first_element_in_bytes    The offset of the first element in the beta tensor
 *
 */
__kernel void binary_convolution(
    TENSOR3D_DECLARATION(src),
    TENSOR4D_DECLARATION(weights),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(biases),
#endif // defined(HAS_BIAS)
    TENSOR3D_DECLARATION(dst),
    VECTOR_DECLARATION(alpha),
    IMAGE_DECLERATION(BETA))
{
    // Calculate input/output addresses
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    // TODO
}