// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// #include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <scheduler/vectorize_helper.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <fstream>

namespace nvfuser {

using VectorizationAnalysisTest = NVFuserTest;

// Simple pad test
TEST_F(VectorizationAnalysisTest, ContigInnerDimsMapperResizeP2C) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({4, 8, 16});
  fusion.addInput(tv0);

  //////////////////////////////
  // resize on fastest dimension
  //////////////////////////////

  // positive resize (+2, +2)
  auto inner_pos = pad(tv0, {IrBuilder::create<Val>(2L), IrBuilder::create<Val>(2L)});

  // positive uneven resize (+4, +2)
  auto inner_pos_uneven = pad(tv0, {IrBuilder::create<Val>(4L), IrBuilder::create<Val>(2L)});

  // positive large resize (+32, +32)
  auto inner_pos_large = pad(tv0, {IrBuilder::create<Val>(32L), IrBuilder::create<Val>(32L)});

  // negative resize (-2, -2)
  auto inner_neg = pad(tv0, {IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(-2L)});

  // negative uneven resize (-2, -4)
  auto inner_neg_uneven = pad(tv0, {IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(-4L)});

  // uneven resize (-2, 4)
  auto inner_uneven = pad(tv0, {IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(4L)});

  // one side resize (0, 4)
  auto inner_one_size = pad(tv0, {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(4L)});

  // validate analysis
  std::unordered_map<TensorView*, Val*> projected_extent_map = ContiguousInnerDimensionsMapper::map(ref, logical_dom).getTvToContigMergeOfInnerSizeMap();

  std::cout << projected_extent_map[inner_pos]->evaluate() << std::endl;
}

} // nvfuser
