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

#if defined(WEIGHTS_DEPTH)

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
__kernel void binary_convolution_3x3(
    TENSOR3D_DECLARATION(src),
    TENSOR4D_DECLARATION(weights),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(biases),
#endif // defined(HAS_BIAS)
    TENSOR3D_DECLARATION(dst),
    VECTOR_DECLARATION(alpha),
    IMAGE_DECLARATION(beta))
{
    // Calculate input/output addresses
    __global uchar *src_addr     = src_ptr + src_offset_first_element_in_bytes + get_global_id(0) * src_stride_x + get_global_id(1) * src_step_y;
    __global uchar *weights_addr = weights_ptr + weights_offset_first_element_in_bytes + get_global_id(2) * weights_stride_w;
    Tensor3D dst                 = CONVERT_TO_TENSOR3D_STRUCT(dst);
    
#if defined(HAS_BIAS)
    __global uchar *biases_addr = biases_ptr + biases_offset_first_element_in_bytes + get_global_id(2) * biases_stride_x;
#endif // defined(HAS_BIAS)
    
    __global uchar *alpha_addr = alpha_ptr + alpha_offset_first_element_in_bytes + get_global_id(2) * alpha_stride_x;
    Image           beta       = CONVERT_TO_IMAGE_STRUCT(beta);
    
    float8 out_vals = 0;
    for(volatile int d = 0; d < WEIGHTS_DEPTH; ++d)
    {
        uchar3 src_vals;
        uchar3 next_byte; //TODO check for padding
        uchar3 tmp = 0;
        uchar3 weights_vals;
        
        src_vals.s0 = *(src_addr + 0 * src_stride_y);
        src_vals.s1 = *(src_addr + 1 * src_stride_y);
        src_vals.s2 = *(src_addr + 2 * src_stride_y);
        
        next_byte.s0 = *(src_addr + 0 * src_stride_y + src_stride_x);
        next_byte.s1 = *(src_addr + 1 * src_stride_y + src_stride_x);
        next_byte.s2 = *(src_addr + 2 * src_stride_y + src_stride_x);
        
        tmp = ((src_vals >> 1) << 7) | ((src_vals << 7) >> 1) | ((next_byte >> 7) << 5) | (((next_byte << 1) >> 7) << 4);
        
        weights_vals.s0 = *(weights_addr + 0 * weights_stride_y);
        weights_vals.s1 = *(weights_addr + 1 * weights_stride_y);
        weights_vals.s2 = *(weights_addr + 2 * weights_stride_y);
        
        weights_vals = (weights_vals >> 5) << 5;
        
        uchar8 mask = (uchar8)(255 >> 3);
        mask.s1 = (mask.s0 >> 1) | 128;
        mask.s2 = (mask.s1 >> 1) | 128;
        mask.s3 = (mask.s2 >> 1) | 128;
        mask.s4 = (mask.s3 >> 1) | 128;
        mask.s5 = (mask.s4 >> 1) | 128;
        
        out_vals.s0 += popcount((uchar)(~(src_vals.s0 ^ ((~src_vals.s0 & mask.s0) | weights_vals.s0))));
        out_vals.s0 += popcount((uchar)(~(src_vals.s1 ^ ((~src_vals.s1 & mask.s0) | weights_vals.s1))));
        out_vals.s0 += popcount((uchar)(~(src_vals.s2 ^ ((~src_vals.s2 & mask.s0) | weights_vals.s2))));

        out_vals.s6 += popcount((uchar)(~(tmp.s0 ^ ((~tmp.s0 & mask.s0) | weights_vals.s0))));
        out_vals.s6 += popcount((uchar)(~(tmp.s1 ^ ((~tmp.s1 & mask.s0) | weights_vals.s1))));
        out_vals.s6 += popcount((uchar)(~(tmp.s2 ^ ((~tmp.s2 & mask.s0) | weights_vals.s2))));
        weights_vals >>= 1;
        
        out_vals.s1 += popcount((uchar)(~(src_vals.s0 ^ ((~src_vals.s0 & mask.s1) | weights_vals.s0))));
        out_vals.s1 += popcount((uchar)(~(src_vals.s1 ^ ((~src_vals.s1 & mask.s1) | weights_vals.s1))));
        out_vals.s1 += popcount((uchar)(~(src_vals.s2 ^ ((~src_vals.s2 & mask.s1) | weights_vals.s2))));
        
        out_vals.s7 += popcount((uchar)(~(tmp.s0 ^ ((~tmp.s0 & mask.s1) | weights_vals.s0))));
        out_vals.s7 += popcount((uchar)(~(tmp.s1 ^ ((~tmp.s1 & mask.s1) | weights_vals.s1))));
        out_vals.s7 += popcount((uchar)(~(tmp.s2 ^ ((~tmp.s2 & mask.s1) | weights_vals.s2))));
        weights_vals >>= 1;
        
        out_vals.s2 += popcount((uchar)(~(src_vals.s0 ^ ((~src_vals.s0 & mask.s2) | weights_vals.s0))));
        out_vals.s2 += popcount((uchar)(~(src_vals.s1 ^ ((~src_vals.s1 & mask.s2) | weights_vals.s1))));
        out_vals.s2 += popcount((uchar)(~(src_vals.s2 ^ ((~src_vals.s2 & mask.s2) | weights_vals.s2))));
        weights_vals >>= 1;
        
        out_vals.s3 += popcount((uchar)(~(src_vals.s0 ^ ((~src_vals.s0 & mask.s3) | weights_vals.s0))));
        out_vals.s3 += popcount((uchar)(~(src_vals.s1 ^ ((~src_vals.s1 & mask.s3) | weights_vals.s1))));
        out_vals.s3 += popcount((uchar)(~(src_vals.s2 ^ ((~src_vals.s2 & mask.s3) | weights_vals.s2))));
        weights_vals >>= 1;
        
        out_vals.s4 += popcount((uchar)(~(src_vals.s0 ^ ((~src_vals.s0 & mask.s4) | weights_vals.s0))));
        out_vals.s4 += popcount((uchar)(~(src_vals.s1 ^ ((~src_vals.s1 & mask.s4) | weights_vals.s1))));
        out_vals.s4 += popcount((uchar)(~(src_vals.s2 ^ ((~src_vals.s2 & mask.s4) | weights_vals.s2))));
        weights_vals >>= 1;
        
        out_vals.s5 += popcount((uchar)(~(src_vals.s0 ^ ((~src_vals.s0 & mask.s5) | weights_vals.s0))));
        out_vals.s5 += popcount((uchar)(~(src_vals.s1 ^ ((~src_vals.s1 & mask.s5) | weights_vals.s1))));
        out_vals.s5 += popcount((uchar)(~(src_vals.s2 ^ ((~src_vals.s2 & mask.s5) | weights_vals.s2))));        

        src_addr     += src_stride_z;
        weights_addr += weights_stride_z;
    }
    
    float8 tot_elems  = (float8)(9 * WEIGHTS_DEPTH);
    float8 beta_vals  = vload8(0, (__global float *)beta.ptr);
    float8 alpha_vals = (float8)(*((__global float *)alpha_addr));
    
    out_vals = ((float8)2 * out_vals - tot_elems) * (alpha_vals * beta_vals);
#if defined(HAS_BIAS)
    float8 biases_vals = (float8)(*((__global float *)biases_addr));
    out_vals += biases_vals;
#endif // defined(HAS_BIAS)
    
    vstore8(out_vals, 0, (__global float *)dst.ptr);
}

#endif // defined(WEIGHTS_DEPTH)
