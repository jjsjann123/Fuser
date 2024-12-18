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

namespace {

void checkMappedVal(const std::unordered_map<TensorView*, Val*>& map, TensorView* tv_target, int64_t val) {
  auto iter = map.find(tv_target);
  EXPECT_TRUE(iter != map.end());
  if (iter != map.end()) {
    EXPECT_EQ(iter->evaluate(), val);
  }
}

} // namespace

using VectorizationAnalysisTest = NVFuserTest;

// Simple pad test
TEST_F(VectorizationAnalysisTest, ContigInnerDimsMapperResizeFastestDimensionP2C) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({4, 8, 16});
  fusion.addInput(tv0);

  std::vector<std::pair<TensorView*, int64_t>> expection_list;

  // positive resize (+2, +2)
  auto inner_pos = pad(tv0, {IrBuilder::create<Val>(2L), IrBuilder::create<Val>(2L)});
  expection_list.emplace_back(std::make_pair(inner_pos, 2));
  fusion.addOutput(inner_pos);

  // positive uneven resize (+4, +2)
  auto inner_pos_uneven = pad(tv0, {IrBuilder::create<Val>(4L), IrBuilder::create<Val>(2L)});
  expection_list.emplace_back(std::make_pair(inner_pos_uneven, 2));
  fusion.addOutput(inner_pos_uneven);

  // positive large resize (+32, +32)
  auto inner_pos_large = pad(tv0, {IrBuilder::create<Val>(32L), IrBuilder::create<Val>(32L)});
  expection_list.emplace_back(std::make_pair(inner_pos_large, 16));
  fusion.addOutput(inner_pos_large);

  // negative resize (-2, -2)
  auto inner_neg = pad(tv0, {IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(-2L)});
  expection_list.emplace_back(std::make_pair(inner_neg, 2));
  fusion.addOutput(inner_neg);

  // negative uneven resize (-2, -4)
  auto inner_neg_uneven = pad(tv0, {IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(-4L)});
  expection_list.emplace_back(std::make_pair(inner_neg_uneven, 2));
  fusion.addOutput(inner_neg_uneven);

  // negative large resize to zero (-8, -8)
  auto inner_neg = pad(tv0, {IrBuilder::create<Val>(-8L), IrBuilder::create<Val>(-8L)});
  expection_list.emplace_back(std::make_pair(inner_neg, 0));
  fusion.addOutput(inner_neg);

  // uneven resize (-2, 4)
  auto inner_uneven = pad(tv0, {IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(4L)});
  expection_list.emplace_back(std::make_pair(inner_uneven, 2));
  fusion.addOutput(inner_uneven);

  // one side resize (0, 4)
  auto inner_one_size = pad(tv0, {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(4L)});
  expection_list.emplace_back(std::make_pair(inner_one_size, 4));
  fusion.addOutput(inner_one_size);

  std::unordered_map<TensorView*, Val*> projected_extent_map = vectorize_helper::ContiguousInnerDimensionsMapper::map(tv0, tv0->getLogicalDomain()).getTvToContigMergeOfInnerSizeMap();

  for (const auto& [tv, val] : expection_list) {
    checkMappedVal(projected_extent_map, , val);
  }
}

TEST_F(VectorizationAnalysisTest, ContigInnerDimsMapperResizeMiddleDimension) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({4, 8, 16});
  fusion.addInput(tv0);

  std::vector<std::pair<TensorView*, int64_t>> expection_list;

  // positive resize (+2, +2)
  auto middle_pos = pad(tv0, {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(0L), IrBuilder::create<Val>(2L), IrBuilder::create<Val>(2L)});
  expection_list.emplace_back(std::make_pair(inner_one_size, 2*16));
  fusion.addOutput(middle_pos);

  // negative resize (-2, -2)
  auto middle_neg = pad(tv0, {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(0L), IrBuilder::create<Val>(-2L), IrBuilder::create<Val>(-2L)}); 
  expection_list.emplace_back(std::make_pair(inner_one_size, 2*16));
  fusion.addOutput(middle_neg);

  std::unordered_map<TensorView*, Val*> projected_extent_map = vectorize_helper::ContiguousInnerDimensionsMapper::map(tv0, tv0->getLogicalDomain()).getTvToContigMergeOfInnerSizeMap();
  for (const auto& [tv, val] : expection_list) {
    checkMappedVal(projected_extent_map, , val);
  }
}

// TEST_F(VectorizationAnalysisTest, ContigInnerDimsMapperResizeStacked) {
//   Fusion fusion;
//   FusionGuard fg(&fusion);
// 
//   auto tv0 = makeContigConcreteTensor({4, 8, 16});
//   fusion.addInput(tv0);
//   /////////////////
//   // stacked resize
//   /////////////////
//   // resize on different dimension
// 
//   // resize on the same dimension
// 
//   // resize on the same dimension, squeeze size to zero
// 
//   std::unordered_map<TensorView*, Val*> projected_extent_map = vectorize_helper::ContiguousInnerDimensionsMapper::map(tv0, tv0->getLogicalDomain()).getTvToContigMergeOfInnerSizeMap();
// 
//   std::cout << projected_extent_map[inner_pos]->evaluate() << std::endl;
// }

} // nvfuser
