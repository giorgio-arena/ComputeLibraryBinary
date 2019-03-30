/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#ifdef __ARM_FEATURE_SVE


#include "../../asmlib.hpp"

namespace arm_gemm {

void sve_interleaved_fp16_mla_3VLx8(const __fp16 *Apanel, const __fp16 *Bpanel, __fp16 *Cpanel, int ablocks, int bblocks, int K) {
    const __fp16 *a_ptr = Apanel;
    __fp16 *c_ptr = Cpanel;

    const long loops_count = (K / 2) - 1;
    const long tails_count = K % 2;

    for (int yb=0; yb<ablocks; yb++) {
        const __fp16 *a_ptr0 = a_ptr;
        const __fp16 *b_ptr = Bpanel;

        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;
            long loops = loops_count;
            long tails = tails_count;

            __asm __volatile (
                "mov z8.h, #0\n"
                "ptrue p0.h\n"
                "mov z9.h, #0\n"
                "mov z10.h, #0\n"
                "mov z11.h, #0\n"
                "mov z12.h, #0\n"
                "ld1rqh z0.h, p0/z, [%[a_ptr]]\n"
                "mov z13.h, #0\n"
                "ld1h z2.h, p0/z, [%[b_ptr]]\n"
                "mov z14.h, #0\n"
                "ld1h z3.h, p0/z, [%[b_ptr], #1, MUL VL]\n"
                "mov z15.h, #0\n"
                "ld1h z4.h, p0/z, [%[b_ptr], #2, MUL VL]\n"
                "mov z16.h, #0\n"
                "ld1h z5.h, p0/z, [%[b_ptr], #3, MUL VL]\n"
                "mov z17.h, #0\n"
                "ld1h z6.h, p0/z, [%[b_ptr], #4, MUL VL]\n"
                "mov z18.h, #0\n"
                "add %[a_ptr], %[a_ptr], #0x20\n"
                "mov z19.h, #0\n"
                "addvl %[b_ptr], %[b_ptr], #6\n"
                "mov z20.h, #0\n"
                "mov z21.h, #0\n"
                "mov z22.h, #0\n"
                "mov z23.h, #0\n"
                "mov z24.h, #0\n"
                "mov z25.h, #0\n"
                "mov z26.h, #0\n"
                "mov z27.h, #0\n"
                "mov z28.h, #0\n"
                "mov z29.h, #0\n"
                "mov z30.h, #0\n"
                "mov z31.h, #0\n"
                "cbz %[loops], 1f\n"
                "2:\n"
                "fmla z8.h, z2.h, z0.h[0]\n"
                "ld1h z7.h, p0/z, [%[b_ptr], #-1, MUL VL]\n"
                "fmla z9.h, z2.h, z0.h[1]\n"
                "ld1rqh z1.h, p0/z, [%[a_ptr], #-0x10]\n"
                "fmla z10.h, z2.h, z0.h[2]\n"
                "subs %[loops], %[loops], #0x1\n"
                "fmla z11.h, z2.h, z0.h[3]\n"
                "fmla z12.h, z2.h, z0.h[4]\n"
                "fmla z13.h, z2.h, z0.h[5]\n"
                "fmla z14.h, z2.h, z0.h[6]\n"
                "fmla z15.h, z2.h, z0.h[7]\n"
                "ld1h z2.h, p0/z, [%[b_ptr]]\n"
                "fmla z16.h, z3.h, z0.h[0]\n"
                "fmla z17.h, z3.h, z0.h[1]\n"
                "fmla z18.h, z3.h, z0.h[2]\n"
                "fmla z19.h, z3.h, z0.h[3]\n"
                "fmla z20.h, z3.h, z0.h[4]\n"
                "fmla z21.h, z3.h, z0.h[5]\n"
                "fmla z22.h, z3.h, z0.h[6]\n"
                "fmla z23.h, z3.h, z0.h[7]\n"
                "ld1h z3.h, p0/z, [%[b_ptr], #1, MUL VL]\n"
                "fmla z24.h, z4.h, z0.h[0]\n"
                "fmla z25.h, z4.h, z0.h[1]\n"
                "fmla z26.h, z4.h, z0.h[2]\n"
                "fmla z27.h, z4.h, z0.h[3]\n"
                "fmla z28.h, z4.h, z0.h[4]\n"
                "fmla z29.h, z4.h, z0.h[5]\n"
                "fmla z30.h, z4.h, z0.h[6]\n"
                "fmla z31.h, z4.h, z0.h[7]\n"
                "ld1h z4.h, p0/z, [%[b_ptr], #2, MUL VL]\n"
                "fmla z8.h, z5.h, z1.h[0]\n"
                "ld1rqh z0.h, p0/z, [%[a_ptr]]\n"
                "fmla z9.h, z5.h, z1.h[1]\n"
                "add %[a_ptr], %[a_ptr], #0x20\n"
                "fmla z10.h, z5.h, z1.h[2]\n"
                "addvl %[b_ptr], %[b_ptr], #6\n"
                "fmla z11.h, z5.h, z1.h[3]\n"
                "fmla z12.h, z5.h, z1.h[4]\n"
                "fmla z13.h, z5.h, z1.h[5]\n"
                "fmla z14.h, z5.h, z1.h[6]\n"
                "fmla z15.h, z5.h, z1.h[7]\n"
                "ld1h z5.h, p0/z, [%[b_ptr], #-3, MUL VL]\n"
                "fmla z16.h, z6.h, z1.h[0]\n"
                "fmla z17.h, z6.h, z1.h[1]\n"
                "fmla z18.h, z6.h, z1.h[2]\n"
                "fmla z19.h, z6.h, z1.h[3]\n"
                "fmla z20.h, z6.h, z1.h[4]\n"
                "fmla z21.h, z6.h, z1.h[5]\n"
                "fmla z22.h, z6.h, z1.h[6]\n"
                "fmla z23.h, z6.h, z1.h[7]\n"
                "ld1h z6.h, p0/z, [%[b_ptr], #-2, MUL VL]\n"
                "fmla z24.h, z7.h, z1.h[0]\n"
                "fmla z25.h, z7.h, z1.h[1]\n"
                "fmla z26.h, z7.h, z1.h[2]\n"
                "fmla z27.h, z7.h, z1.h[3]\n"
                "fmla z28.h, z7.h, z1.h[4]\n"
                "fmla z29.h, z7.h, z1.h[5]\n"
                "fmla z30.h, z7.h, z1.h[6]\n"
                "fmla z31.h, z7.h, z1.h[7]\n"
                "b.ne 2b\n"
                "1:\n"
                "cbz %[tails], 3f\n"
                "fmla z8.h, z2.h, z0.h[0]\n"
                "ld1h z7.h, p0/z, [%[b_ptr], #-1, MUL VL]\n"
                "fmla z9.h, z2.h, z0.h[1]\n"
                "ld1rqh z1.h, p0/z, [%[a_ptr], #-0x10]\n"
                "fmla z10.h, z2.h, z0.h[2]\n"
                "fmla z11.h, z2.h, z0.h[3]\n"
                "fmla z12.h, z2.h, z0.h[4]\n"
                "fmla z13.h, z2.h, z0.h[5]\n"
                "fmla z14.h, z2.h, z0.h[6]\n"
                "fmla z15.h, z2.h, z0.h[7]\n"
                "ld1h z2.h, p0/z, [%[b_ptr]]\n"
                "fmla z16.h, z3.h, z0.h[0]\n"
                "fmla z17.h, z3.h, z0.h[1]\n"
                "fmla z18.h, z3.h, z0.h[2]\n"
                "fmla z19.h, z3.h, z0.h[3]\n"
                "fmla z20.h, z3.h, z0.h[4]\n"
                "fmla z21.h, z3.h, z0.h[5]\n"
                "fmla z22.h, z3.h, z0.h[6]\n"
                "fmla z23.h, z3.h, z0.h[7]\n"
                "ld1h z3.h, p0/z, [%[b_ptr], #1, MUL VL]\n"
                "fmla z24.h, z4.h, z0.h[0]\n"
                "fmla z25.h, z4.h, z0.h[1]\n"
                "fmla z26.h, z4.h, z0.h[2]\n"
                "fmla z27.h, z4.h, z0.h[3]\n"
                "fmla z28.h, z4.h, z0.h[4]\n"
                "fmla z29.h, z4.h, z0.h[5]\n"
                "fmla z30.h, z4.h, z0.h[6]\n"
                "fmla z31.h, z4.h, z0.h[7]\n"
                "ld1h z4.h, p0/z, [%[b_ptr], #2, MUL VL]\n"
                "fmla z8.h, z5.h, z1.h[0]\n"
                "ld1rqh z0.h, p0/z, [%[a_ptr]]\n"
                "fmla z9.h, z5.h, z1.h[1]\n"
                "add %[a_ptr], %[a_ptr], #0x10\n"
                "fmla z10.h, z5.h, z1.h[2]\n"
                "addvl %[b_ptr], %[b_ptr], #3\n"
                "fmla z11.h, z5.h, z1.h[3]\n"
                "fmla z12.h, z5.h, z1.h[4]\n"
                "fmla z13.h, z5.h, z1.h[5]\n"
                "fmla z14.h, z5.h, z1.h[6]\n"
                "fmla z15.h, z5.h, z1.h[7]\n"
                "fmla z16.h, z6.h, z1.h[0]\n"
                "fmla z17.h, z6.h, z1.h[1]\n"
                "fmla z18.h, z6.h, z1.h[2]\n"
                "fmla z19.h, z6.h, z1.h[3]\n"
                "fmla z20.h, z6.h, z1.h[4]\n"
                "fmla z21.h, z6.h, z1.h[5]\n"
                "fmla z22.h, z6.h, z1.h[6]\n"
                "fmla z23.h, z6.h, z1.h[7]\n"
                "fmla z24.h, z7.h, z1.h[0]\n"
                "fmla z25.h, z7.h, z1.h[1]\n"
                "fmla z26.h, z7.h, z1.h[2]\n"
                "fmla z27.h, z7.h, z1.h[3]\n"
                "fmla z28.h, z7.h, z1.h[4]\n"
                "fmla z29.h, z7.h, z1.h[5]\n"
                "fmla z30.h, z7.h, z1.h[6]\n"
                "fmla z31.h, z7.h, z1.h[7]\n"
                "fmla z8.h, z2.h, z0.h[0]\n"
                "fmla z9.h, z2.h, z0.h[1]\n"
                "fmla z10.h, z2.h, z0.h[2]\n"
                "fmla z11.h, z2.h, z0.h[3]\n"
                "fmla z12.h, z2.h, z0.h[4]\n"
                "st1h z8.h, p0, [%[c_ptr]]\n"
                "fmla z13.h, z2.h, z0.h[5]\n"
                "fmla z14.h, z2.h, z0.h[6]\n"
                "fmla z15.h, z2.h, z0.h[7]\n"
                "fmla z16.h, z3.h, z0.h[0]\n"
                "fmla z17.h, z3.h, z0.h[1]\n"
                "fmla z18.h, z3.h, z0.h[2]\n"
                "fmla z19.h, z3.h, z0.h[3]\n"
                "fmla z20.h, z3.h, z0.h[4]\n"
                "st1h z16.h, p0, [%[c_ptr], #1, MUL VL]\n"
                "fmla z21.h, z3.h, z0.h[5]\n"
                "fmla z22.h, z3.h, z0.h[6]\n"
                "fmla z23.h, z3.h, z0.h[7]\n"
                "fmla z24.h, z4.h, z0.h[0]\n"
                "fmla z25.h, z4.h, z0.h[1]\n"
                "fmla z26.h, z4.h, z0.h[2]\n"
                "fmla z27.h, z4.h, z0.h[3]\n"
                "fmla z28.h, z4.h, z0.h[4]\n"
                "st1h z24.h, p0, [%[c_ptr], #2, MUL VL]\n"
                "fmla z29.h, z4.h, z0.h[5]\n"
                "fmla z30.h, z4.h, z0.h[6]\n"
                "fmla z31.h, z4.h, z0.h[7]\n"
                "b 4f\n"
                "3:\n"
                "fmla z8.h, z2.h, z0.h[0]\n"
                "ld1h z7.h, p0/z, [%[b_ptr], #-1, MUL VL]\n"
                "fmla z9.h, z2.h, z0.h[1]\n"
                "ld1rqh z1.h, p0/z, [%[a_ptr], #-0x10]\n"
                "fmla z10.h, z2.h, z0.h[2]\n"
                "fmla z11.h, z2.h, z0.h[3]\n"
                "fmla z12.h, z2.h, z0.h[4]\n"
                "fmla z13.h, z2.h, z0.h[5]\n"
                "fmla z14.h, z2.h, z0.h[6]\n"
                "fmla z15.h, z2.h, z0.h[7]\n"
                "fmla z16.h, z3.h, z0.h[0]\n"
                "fmla z17.h, z3.h, z0.h[1]\n"
                "fmla z18.h, z3.h, z0.h[2]\n"
                "fmla z19.h, z3.h, z0.h[3]\n"
                "fmla z20.h, z3.h, z0.h[4]\n"
                "fmla z21.h, z3.h, z0.h[5]\n"
                "fmla z22.h, z3.h, z0.h[6]\n"
                "fmla z23.h, z3.h, z0.h[7]\n"
                "fmla z24.h, z4.h, z0.h[0]\n"
                "fmla z25.h, z4.h, z0.h[1]\n"
                "fmla z26.h, z4.h, z0.h[2]\n"
                "fmla z27.h, z4.h, z0.h[3]\n"
                "fmla z28.h, z4.h, z0.h[4]\n"
                "fmla z29.h, z4.h, z0.h[5]\n"
                "fmla z30.h, z4.h, z0.h[6]\n"
                "fmla z31.h, z4.h, z0.h[7]\n"
                "fmla z8.h, z5.h, z1.h[0]\n"
                "fmla z9.h, z5.h, z1.h[1]\n"
                "fmla z10.h, z5.h, z1.h[2]\n"
                "fmla z11.h, z5.h, z1.h[3]\n"
                "fmla z12.h, z5.h, z1.h[4]\n"
                "st1h z8.h, p0, [%[c_ptr]]\n"
                "fmla z13.h, z5.h, z1.h[5]\n"
                "fmla z14.h, z5.h, z1.h[6]\n"
                "fmla z15.h, z5.h, z1.h[7]\n"
                "fmla z16.h, z6.h, z1.h[0]\n"
                "fmla z17.h, z6.h, z1.h[1]\n"
                "fmla z18.h, z6.h, z1.h[2]\n"
                "fmla z19.h, z6.h, z1.h[3]\n"
                "fmla z20.h, z6.h, z1.h[4]\n"
                "st1h z16.h, p0, [%[c_ptr], #1, MUL VL]\n"
                "fmla z21.h, z6.h, z1.h[5]\n"
                "fmla z22.h, z6.h, z1.h[6]\n"
                "fmla z23.h, z6.h, z1.h[7]\n"
                "fmla z24.h, z7.h, z1.h[0]\n"
                "fmla z25.h, z7.h, z1.h[1]\n"
                "fmla z26.h, z7.h, z1.h[2]\n"
                "fmla z27.h, z7.h, z1.h[3]\n"
                "fmla z28.h, z7.h, z1.h[4]\n"
                "st1h z24.h, p0, [%[c_ptr], #2, MUL VL]\n"
                "fmla z29.h, z7.h, z1.h[5]\n"
                "fmla z30.h, z7.h, z1.h[6]\n"
                "fmla z31.h, z7.h, z1.h[7]\n"
                "4:\n"
                "st1h z9.h, p0, [%[c_ptr], #3, MUL VL]\n"
                "st1h z17.h, p0, [%[c_ptr], #4, MUL VL]\n"
                "st1h z25.h, p0, [%[c_ptr], #5, MUL VL]\n"
                "st1h z10.h, p0, [%[c_ptr], #6, MUL VL]\n"
                "st1h z18.h, p0, [%[c_ptr], #7, MUL VL]\n"
                "addvl %[c_ptr], %[c_ptr], #16\n"
                "st1h z26.h, p0, [%[c_ptr], #-8, MUL VL]\n"
                "st1h z11.h, p0, [%[c_ptr], #-7, MUL VL]\n"
                "st1h z19.h, p0, [%[c_ptr], #-6, MUL VL]\n"
                "st1h z27.h, p0, [%[c_ptr], #-5, MUL VL]\n"
                "st1h z12.h, p0, [%[c_ptr], #-4, MUL VL]\n"
                "st1h z20.h, p0, [%[c_ptr], #-3, MUL VL]\n"
                "st1h z28.h, p0, [%[c_ptr], #-2, MUL VL]\n"
                "st1h z13.h, p0, [%[c_ptr], #-1, MUL VL]\n"
                "st1h z21.h, p0, [%[c_ptr]]\n"
                "st1h z29.h, p0, [%[c_ptr], #1, MUL VL]\n"
                "st1h z14.h, p0, [%[c_ptr], #2, MUL VL]\n"
                "st1h z22.h, p0, [%[c_ptr], #3, MUL VL]\n"
                "st1h z30.h, p0, [%[c_ptr], #4, MUL VL]\n"
                "st1h z15.h, p0, [%[c_ptr], #5, MUL VL]\n"
                "st1h z23.h, p0, [%[c_ptr], #6, MUL VL]\n"
                "st1h z31.h, p0, [%[c_ptr], #7, MUL VL]\n"
                "addvl %[c_ptr], %[c_ptr], #8\n"
            : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
              [loops] "+r" (loops), [tails] "+r" (tails)
            :
            : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "cc", "memory"
            );
        }
    }
}

} // namespace arm_gemm

#endif // __ARM_FEATURE_SVE
