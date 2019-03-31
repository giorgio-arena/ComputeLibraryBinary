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
#include "arm_compute/core/Types.h"
#include "tests/CL/Helper.h"

#include "tests/CL/CLAccessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/BinarySignFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr RelativeTolerance<float> tolerance_f32(0.01f); /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
} // namespace

using CLBinarySign        = CLSynthetizeFunction<CLBinarySignKernel>;
using CLBinarySignFixture = BinarySignValidationFixture<CLTensor, CLAccessor, CLBinarySign>;

TEST_SUITE(CL)
TEST_SUITE(BinarySign)

FIXTURE_DATA_TEST_CASE(RunSmall, CLBinarySignFixture, framework::DatasetMode::ALL, datasets::SmallShapes())
{
    // Validate output
    validate(CLAccessor(_target_out), _reference_out);
    // Validate alpha
    validate(CLAccessor(_target_alpha), _reference_alpha, tolerance_f32);
    // Validate beta
    validate(CLAccessor(_target_beta), _reference_beta, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLBinarySignFixture, framework::DatasetMode::NIGHTLY, datasets::LargeShapes())
{
    // Validate output
    validate(CLAccessor(_target_out), _reference_out);
    // Validate alpha
    validate(CLAccessor(_target_alpha), _reference_alpha, tolerance_f32);
    // Validate beta
    validate(CLAccessor(_target_beta), _reference_beta, tolerance_f32);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
