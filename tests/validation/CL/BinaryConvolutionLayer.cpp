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
 * OUT OF OR IN CONCLCTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLBinaryConvolutionLayer.h"
#include "tests/CL/CLAccessor.h"
#include "tests/PaddingCalculator.h"
#include "tests/datasets/SmallConvolutionLayerDataset.h"
#include "tests/datasets/LargeConvolutionLayerDataset.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/BinaryConvolutionLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{

using CLBinaryConvolutionLayerFixture = BinaryConvolutionLayerValidationFixture<CLTensor, CLAccessor, CLBinaryConvolutionLayer>;

TEST_SUITE(CL)
TEST_SUITE(BinaryConvolutionLayer)

FIXTURE_DATA_TEST_CASE(RunSmall, CLBinaryConvolutionLayerFixture, framework::DatasetMode::ALL, combine(datasets::SmallConvolutionLayerDataset(),
                               framework::dataset::make("DataLayout", /*{*/ DataLayout::NCHW/*, DataLayout::NHWC }*/)))
{
    //validate(CLAccessor(_target), _reference);
}

FIXTURE_DATA_TEST_CASE(RunLarge, CLBinaryConvolutionLayerFixture, framework::DatasetMode::NIGHTLY, combine(datasets::LargeConvolutionLayerDataset(),
                               framework::dataset::make("DataLayout", /*{*/ DataLayout::NCHW/*, DataLayout::NHWC }*/)))
{
    //validate(CLAccessor(_target), _reference);
}

TEST_SUITE_END()
TEST_SUITE_END()
} // namespace validation
} // namespace test
} // namespace arm_compute