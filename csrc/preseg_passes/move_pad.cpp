// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/move_pad.h>

#include <fusion.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <transform_replay.h>

namespace nvfuser::preseg_passes {

namespace {

struct Edge {
  Expr* expr_ = nullptr;
  size_t index_ = 0;

  Edge(Expr* expr, size_t index) : expr_(expr), index_(index) {}

  Val* val() const {
    return expr_->input(index_);
  }
};

using VecPadWidths = std::vector<std::vector<Val*>>;

// NOTE: this assumes all vec_pad_widths are positive entries so we don't need to consider accumulating them changing the output iter_type.
TensorView* replayConcretePad(TensorView* pad_tv, Val* pad_value, const VecPadWidths& vec_pad_widths, std::vector<IterDomain*> ref_iter_type) {
    NVF_ERROR(
        pad_tv->getDataType().has_value(), "pad source dtype is missing");
    const std::vector<IterDomain*> inp_dom =
        TensorDomain::noReductions(pad_tv->getLogicalDomain());
    const auto rank = inp_dom.size();

    NVF_ERROR(
        rank == ref_iter_type.size(),
        "ref_iter_type does not have compatible size regarding pad_tv");

    std::vector<Val*> merged_pad_widths;
    merged_pad_widths.reserve(rank*2);

    NVF_ERROR(
        !vec_pad_widths.empty(),
        "vec_pad_widths cannot be empty");
    if (vec_pad_widths.size() == 1) {
      merged_pad_widths = vec_pad_widths.at(0);
    } else {
      NVF_ERROR(false, "NOT IMPLEMENTED");
      for (const auto i : c10::irange(2*rank)) {
        Val* merged_pad_width = nullptr;
        for (const auto idx : c10::irange(vec_pad_widths.size())) {
          // skipping zero pad;
          Val* pad_width = vec_pad_widths[idx].at(i);
          if (pad_width->isZeroInt()) {
            continue;
          }
          merged_pad_width = merged_pad_width == nullptr ? pad_width : add(merged_pad_width, pad_width);
        }
        merged_pad_widths.push_back(merged_pad_width == nullptr? FusionGuard::getCurFusion()->zeroVal() : merged_pad_width);
      }
    }

    std::vector<IterDomain*> merged_root_ids;
    std::vector<IterDomain*> merged_logical_ids;
    for (const auto i : c10::irange(rank)) {
      Val* left_pad = merged_pad_widths.at(i * 2);
      Val* right_pad = merged_pad_widths.at(i * 2 + 1);
      IterDomain* inp_id = inp_dom.at(i);
      if (left_pad->isZeroInt() && right_pad->isZeroInt()) {
        merged_root_ids.push_back(inp_id->cloneWithoutRFactor());
        merged_logical_ids.push_back(merged_root_ids.back());
        continue;
      }
      // NOTE: nvfuser pad doesn't support negative padding, so we don't have to
      // worry about it cancelling out.
      IterDomain* merged_root_id =
          IterDomainBuilder(inp_id).is_rfactor_domain(true).build();
      merged_root_ids.push_back(merged_root_id);
      merged_logical_ids.push_back(IterDomain::resize(
          merged_root_id,
          left_pad,
          right_pad,
          true,
          ref_iter_type.at(i)->getIterType()));
    }

    auto* new_out = IrBuilder::create<TensorView>(
        IrBuilder::create<TensorDomain>(
            merged_root_ids, merged_logical_ids, merged_logical_ids),
        pad_tv->getDataType().value());
    IrBuilder::create<PadOp>(
        new_out, pad_tv, merged_pad_widths, pad_value);
    return new_out;
}

Val* propagatePadToProducer(PadOp* pad_op) {
  // TODO: wait, why do I need pad_dependencies?!
  std::vector<Val*> pad_dependencies;

  auto candidate_check = [&pad_dependencies](Val* val) {
    if (!val->isA<TensorView>()) {
      return false;
    }
    // TODO: refactor this check. We should totally support multiple uses here.
    // multiple uses should only block further back propagation when it applies. But we can replace the edge with a padded output. hint. we should count the encounter of each edge and check `encounter == # of uses` to proceed with propagation.
    // Get some use case with binary operation on this one.
    if (val->uses().size() > 1) {
      return false;
    }
    if (val->isFusionOutput()) {
      return false;
    }
    if (std::any_of(
            pad_dependencies.begin(),
            pad_dependencies.end(),
            [val](Val* pad_dependency) {
              return DependencyCheck::isDependencyOf(pad_dependency, val);
            })) {
      return false;
    }
    return true;
  };

  // NOTE: the optimization logic assumes a zero pad_op.
  // This is used for logic in handling binary operations, we should extend this
  // later.
  if (!pad_op->value()->isZero()) {
    return nullptr;
  }

  if (!candidate_check(pad_op->in())) {
    return nullptr;
  }

  for (Val* val : pad_op->inputs()) {
    if (val == pad_op->in() || val->isConst()) {
      continue;
    }
    pad_dependencies.push_back(val);
  }

  std::vector<Edge> frontier;
  // TODO: not sure if I need a stack if I need to keep a replay_sequence.
  std::stack<Edge> stack;
  std::vector<Expr*> replay_sequence;
  stack.emplace(pad_op, 0);

  // tvs in stack are:
  //   1. single use;
  //   2. not an output;
  //   3. cleared dependency of `pad_depdencies`;
  //   4. maybe also check aliases?!
  while (!stack.empty()) {
    Edge edge = stack.top();
    Expr* def = edge.val()->definition();
    stack.pop();

    if (def->isA<UnaryOp>()) {
      auto* uop = def->as<UnaryOp>();
      // TODO: exception to break propagation. i.e. check op type and exclude
      // division by 0
      if (candidate_check(uop->in())) {
        replay_sequence.push_back(uop);
        stack.emplace(uop, 0);
        continue;
        // TODO: this isn't right. I need to support a topology with fork, which
        // is what rope has. } else if (uop->in()->isA<TensorView>()) {
        //   // even though we cannot further propagate, we can still fork it
        //   here. replay_sequence.push_back(uop);
      }
      if (uop->in()->isA<TensorView>()) {
        // even though we cannot further propagate, we'd still want to replace the use here.
        replay_sequence.push_back(uop);
        frontier.emplace_back(uop, 0);
        continue;
        // replace producer here.
      }
      // This will require us having `replayExprWithNewInput` to support binary
      // ops.
      // TODO: adding pad_op
      // } else if (def->isA<PadOp>()) {
      //   if (candidate_check(def->input(0))) {
      //     // NOTE: stopping propagation, we'll merge it with its consumer
      //     padOp frontier.emplace_back(def, 0); continue;
      //   }
    } else if (def->isA<BinaryOp>()) {
      auto* bop = def->as<BinaryOp>();
      // TODO: exception to break propagation. i.e. check op type and exclude
      // division by 0; check for broadcast on padded axis.
      if (candidate_check(bop->lhs()) && candidate_check(bop->rhs())) {
        auto* lhs = bop->lhs()->as<TensorView>();
        auto* rhs = bop->rhs()->as<TensorView>();
        bool pad_on_broadcast = false;
        for (auto i : pad_op->getPaddedAxes()) {
          if (lhs->getLogicalDomain()[i]->isBroadcast() ||
              rhs->getLogicalDomain()[i]->isBroadcast()) {
            pad_on_broadcast = true;
            break;
          }
        }
        if (!pad_on_broadcast) {
          stack.emplace(bop, 0);
          stack.emplace(bop, 1);
          replay_sequence.push_back(bop);
          continue;
        }
      }
    }

    if (edge.val() != pad_op->in()) {
      // propagation stopped, push entry to frontier
      frontier.push_back(edge);
    }
  }

  if (frontier.empty()) {
    return nullptr;
  }

  std::unordered_map<Val*, Val*> replacement_map;
  // modify pad_op on frontier
  for (const Edge& edge : frontier) {
    // insert pad_op
    // Note: operation with multiple operand would require us to support partial
    // update in each iteration.

    // const auto width_size = pad_op->inputs().size() - 2;
    // const auto num_padded_dims = width_size / 2;
    // std::vector<Val*> pad_width;
    // pad_width.reserve(width_size);
    // for (auto i : c10::irange(num_padded_dims)) {
    //   pad_width.push_back(pad_op->input((num_padded_dims - i)*2));
    //   pad_width.push_back(pad_op->input((num_padded_dims - i)*2 + 1));
    // }
    // cannot use `pad` op, because it would give us symolic iter domain
    // replacement_map[edge.val()] = pad(edge.val()->as<TensorView>(),
    // pad_width, pad_op->value());

    auto pad_tv = edge.val()->as<TensorView>();
    const std::vector<IterDomain*> out_ids = TensorDomain::noReductions(
        pad_op->out()->as<TensorView>()->getLogicalDomain());

    TensorView* new_out = replayConcretePad(pad_tv, pad_op->value(), {pad_op->getPadWidths()}, out_ids);

    // TODO: test output from reduction here
    // TensorView* pad_out_tv = pad_op->out()->as<TensorView>();
    // std::vector<IterDomain*> new_root =
    // IterDomain::clone(TensorDomain::noReductions(edge.val()->as<TensorView>()->getMaybeRootDomain()),
    // true); NOTE: we use pad_out_tv instead of edge.val()->as<TensorView>()
    // since the input tensor doesn't have its root id marked with rfactor flag.
    // std::vector<IterDomain*> new_root =
    //     IterDomain::clone(edge.val()->as<TensorView>()->getMaybeRootDomain(), true);
        // should use edge.val() to ensure that we have the right broadcast marked on root.
        // IterDomain::clone(pad_out_tv->getMaybeRootDomain(), true);
    // NOTE: we cannot use the TensorDomain from fullSelfReplay, since it
    // doesn't keep root domain.

    // TODO: cannot use fullSelfReplay here since it requires matching broadcast between new root to old root in domain. I should merge the two `create<PadOp>` instances so we can basically have a `replayPadOnProducer` function.
    // std::vector<IterDomain*> new_logical =
    //     TransformReplay::fullSelfReplay(
    //         IrBuilder::create<TensorDomain>(new_root),
    //         pad_out_tv->domain(),
    //         true)
    //         ->logical();
    // auto new_out = IrBuilder::create<TensorView>(
    //     IrBuilder::create<TensorDomain>(new_root, new_logical, new_logical),
    //     edge.val()->getDataType().value());
    // IrBuilder::create<PadOp>(
    //     new_out,
    //     edge.val()->as<TensorView>(),
    //     pad_op->getPadWidths(),
    //     pad_op->value());

    replacement_map[edge.val()] = new_out;

    // TODO: modify existing pad_op, when its only consumer is a pad_op
  }

  // propagate to update TensorProxy
  // need to follow the reverse order from earlier stack traversal.
  std::reverse(replay_sequence.begin(), replay_sequence.end());
  for (Expr* e : replay_sequence) {
    if (e->isA<UnaryOp>()) {
      // TODO extend this for multiple inputs.
      // Expr* padded_e = replayExprWithNewInput(e,
      // replacement_map.at(e->input(0)));
      Val* out = ops::newValLike(
          replacement_map.at(e->input(0)), e->output(0)->getDataType().value());
      Expr* padded_e = IrBuilder::create<UnaryOp>(
          e->as<UnaryOp>()->getUnaryOpType(),
          out,
          replacement_map.at(e->input(0)));
      replacement_map[e->output(0)] = padded_e->output(0);
    } else if (e->isA<BinaryOp>()) {
      // Expr* padded_e = replayExprWithNewInput(e,
      // replacement_map.at(e->input(0)), replacement_map.at(e->input(1)));
      std::vector<Val*> vals = {
          replacement_map.at(e->input(0)), replacement_map.at(e->input(1))};
      Val* out = ops::newOutputTV(vals, e->output(0)->getDataType().value());
      Expr* padded_e = IrBuilder::create<BinaryOp>(
          e->as<BinaryOp>()->getBinaryOpType(), out, vals[0], vals[1]);
      replacement_map[e->output(0)] = padded_e->output(0);
    } else {
      NVF_ERROR(false, "expr type for propagation is not implemented");
    }
  }

  // return the replacement input to pad_op, since we have already padded
  // everything out.
  return replacement_map.at(pad_op->in());
}

void decomposeCatOp(Fusion* fusion) {
  // TODO: verify that no dead branch is traversed in exprs.
  std::vector<Expr*> exprs = fusion->exprs();

  // TODO: should we expand this optimization to general pad but not just pad
  // within cat?

  // is this traversing in topo order?
  for (auto* cat : ir_utils::filterByType<CatOp>(exprs)) {
    std::unordered_map<Val*, Val*> replacement_map;
    for (Val* in : cat->inputs()) {
      auto* pad_op = in->definition()->as<PadOp>();
      if (Val* new_pad_out = propagatePadToProducer(pad_op)) {
        replacement_map[in] = new_pad_out;
      }
    }
    if (replacement_map.empty()) {
      continue;
    }
    // NOTE: I'm hitting an index error with PadOp in
    // device_lower/pass/index.cpp:1944
    Val* res = nullptr;
    TensorView* cat_out_tv = cat->output(0)->as<TensorView>();
    bool is_boolean = isBooleanType(cat_out_tv->getDataType().value());
    for (Val* inp : cat->inputs()) {
      if (res == nullptr) {
        res = replacement_map.count(inp) == 0 ? inp : replacement_map.at(inp);
      } else {
        Val* rhs =
            replacement_map.count(inp) == 0 ? inp : replacement_map.at(inp);
        if (is_boolean) {
          res = bitwise_or(res, rhs);
        } else {
          res = add(res, rhs);
        }
      }
    }
    // restore data type if it's promoted by BinaryOp.
    res = maybeCastOp(cat_out_tv->getDataType().value(), res);

    // TODO: We won't have it in tests yet, but would replaceValue also replace
    // the outputs of the fusion?
    // TODO: does this invalidate the downstream exprs?
    // ir_utils::replaceValue(fusion, replacement_map);
    ir_utils::replaceValue(fusion, {{cat->output(0), res}});
    // Do we *have to* swap cat with pointwise add?
    if (cat->output(0)->isFusionOutput()) {
      fusion->replaceOutput(cat->output(0), res);
    }
  }
}

void mergeNeighboringPad(Fusion* fusion) {
  std::vector<Expr*> exprs = fusion->exprs();
  // traverse in topo order. We'll merge current pad into its one and only
  // consumer pad so we don't have to worry about interfering the traversal.
  for (auto* producer : ir_utils::filterByType<PadOp>(exprs)) {
    Val* pad_out = producer->out();
    if (pad_out->uses().size() != 1) {
      continue;
    }
    if (!pad_out->uses()[0]->isA<PadOp>()) {
      continue;
    }
    auto* consumer = pad_out->uses()[0]->as<PadOp>();
    // TODO: check for pad value being equal.
    if ((producer->value() != consumer->value()) &&
        (!producer->value()->isZero() || !consumer->value()->isZero())) {
      continue;
    }

    const std::vector<Val*> p_pad_widths = producer->getPadWidths();
    const std::vector<Val*> c_pad_widths = consumer->getPadWidths();

    // I think this should always hold, otherwise we can relax it and continue
    // instead.
    NVF_ERROR(
        p_pad_widths.size() == c_pad_widths.size(),
        "expect consecutive PadOp to have the same length of pad widths");

    auto* pad_inp = producer->in()->as<TensorView>();
    const std::vector<IterDomain*> inp_dom =
        TensorDomain::noReductions(pad_inp->getLogicalDomain());
    const std::vector<IterDomain*> out_dom = TensorDomain::noReductions(
        consumer->out()->as<TensorView>()->getLogicalDomain());

    NVF_ERROR(
        2 * inp_dom.size() == c_pad_widths.size(),
        "input rank is not compatible with pad widths");

    std::vector<Val*> merged_pad_width;
    merged_pad_width.reserve(p_pad_widths.size());

    std::vector<IterDomain*> merged_root_ids;
    std::vector<IterDomain*> merged_logical_ids;

    for (const auto i : c10::irange(inp_dom.size())) {
      Val* p_left_pad = p_pad_widths.at(i * 2);
      Val* p_right_pad = p_pad_widths.at(i * 2 + 1);
      Val* c_left_pad = c_pad_widths.at(i * 2);
      Val* c_right_pad = c_pad_widths.at(i * 2 + 1);
      IterDomain* inp_id = inp_dom.at(i);

      if (p_left_pad->isZeroInt() && p_right_pad->isZeroInt() &&
          c_left_pad->isZeroInt() && c_right_pad->isZeroInt()) {
        merged_root_ids.push_back(inp_id->cloneWithoutRFactor());
        merged_logical_ids.push_back(merged_root_ids.back());
        merged_pad_width.push_back(FusionGuard::getCurFusion()->zeroVal());
        merged_pad_width.push_back(FusionGuard::getCurFusion()->zeroVal());
        continue;
      }
      // TODO: Add test with merging pad on different dimensions
      // NOTE: should I worry about simplifying this by skipping zero?
      Val* merged_left_pad = add(p_left_pad, c_left_pad);
      Val* merged_right_pad = add(p_right_pad, c_right_pad);
      merged_pad_width.push_back(merged_left_pad);
      merged_pad_width.push_back(merged_right_pad);
      // NOTE: nvfuser pad doesn't support negative padding, so we don't have to
      // worry about it cancelling out.
      IterDomain* merged_root_id =
          IterDomainBuilder(inp_id).is_rfactor_domain(true).build();
      merged_root_ids.push_back(merged_root_id);
      merged_logical_ids.push_back(IterDomain::resize(
          merged_root_id,
          merged_left_pad,
          merged_right_pad,
          true,
          out_dom.at(i)->getIterType()));
    }

    auto* new_out = IrBuilder::create<TensorView>(
        IrBuilder::create<TensorDomain>(
            merged_root_ids, merged_logical_ids, merged_logical_ids),
        pad_inp->getDataType().value());
    IrBuilder::create<PadOp>(
        new_out, pad_inp, merged_pad_width, producer->value());

    ir_utils::replaceValue(
        fusion, {{consumer->out(), static_cast<Val*>(new_out)}});
    // Do we *have to* swap cat with pointwise add?
    if (consumer->out()->isFusionOutput()) {
      fusion->replaceOutput(consumer->out(), new_out);
    }
  }
}

} // namespace

void MovePadPass::runPass(Fusion* fusion) {
  std::cout << "=== input fusion ===" << std::endl;
  fusion->printMath();
  decomposeCatOp(fusion);
  std::cout << "=== after decompose cat fusion ===" << std::endl;
  fusion->printMath();
  mergeNeighboringPad(fusion);
  std::cout << "=== after merge pad fusion ===" << std::endl;
  fusion->printMath();
}

} // namespace nvfuser::preseg_passes
