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

#if defined(SRC_WIDTH)

/** This function computes the bitwise AND of two input images.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: U8
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void binary_sign(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    // Calculate input/output addresses
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + get_global_id(0) * dst_stride_x + get_global_id(1) * dst_step_y;
    
    // Load values
    float8 src_vals = vload8(0, (__global float *)src.ptr);
    
    // Don't consider padding
    int8 x_pos = (int8)(get_global_id(0) * 8) + (int8)(0, 1, 2, 3, 4, 5, 6, 7);
    src_vals = select((float8)0, src_vals, x_pos < SRC_WIDTH);
    
    // Calculate sign of elements
    uchar8 signs = select((uchar8)0, (uchar8)1, convert_uchar8(src_vals > (float8)0));
    
    // Write in a single byte
    uchar dst_val = 0;
    dst_val |= (signs.s0 << 7);
    dst_val |= (signs.s1 << 6);
    dst_val |= (signs.s2 << 5);
    dst_val |= (signs.s3 << 4);
    dst_val |= (signs.s4 << 3);
    dst_val |= (signs.s5 << 2);
    dst_val |= (signs.s6 << 1);
    dst_val |= (signs.s7);
    
    // Write to output tensor
    *dst_addr = dst_val;
}

#endif // defined(SRC_WIDTH)