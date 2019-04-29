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
#include "arm_compute/core/Types.h"

#include "tests/NEON/Accessor.h"
#include "tests/NEON/Helper.h"
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

using NEBinarySign        = NESynthetizeFunction<NEBinarySignKernel>;
using NEBinarySignFixture = BinarySignValidationFixture<Tensor, Accessor, NEBinarySign>;

TEST_SUITE(NEON)
TEST_SUITE(BinarySign)

FIXTURE_DATA_TEST_CASE(RunSmall, NEBinarySignFixture, framework::DatasetMode::ALL, datasets::SmallShapes())
{
    // Validate output
    validate(Accessor(_target_out), _reference_out);
    // Validate alpha
    validate(Accessor(_target_alpha), _reference_alpha, tolerance_f32);
    // Validate beta
    validate(Accessor(_target_beta), _reference_beta, tolerance_f32);
}

FIXTURE_DATA_TEST_CASE(RunLarge, NEBinarySignFixture, framework::DatasetMode::NIGHTLY, datasets::LargeShapes())
{
    // Validate output
    validate(Accessor(_target_out), _reference_out);
    // Validate alpha
    validate(Accessor(_target_alpha), _reference_alpha, tolerance_f32);
    // Validate beta
    validate(Accessor(_target_beta), _reference_beta, tolerance_f32);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute
