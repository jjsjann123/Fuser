// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <debug.h>
#include <evaluator_common.h>
#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <polymorphic_value.h>

#include <functional>
#include <iostream>

namespace nvfuser {

namespace {

// Given a value, if it is not a fusion input, return the empty string. If it is
// a fusion input return a string like "input 2 ". This helper is used to
// provide more informative error messages when a malformed input is received.
std::string getInputPosString(const Val* val) {
  if (!val->isFusionInput()) {
    return "";
  }
  // Get position
  const std::vector<Val*>& inputs = val->fusion()->inputs();
  int64_t pos = -1;
  for (size_t i : c10::irange(inputs.size())) {
    if (inputs[i] == val) {
      pos = (int64_t)i;
      break;
    }
  }
  NVF_ERROR(
      pos != -1,
      "val->isFusionInput() is true but val cannot be found in fusion inputs: ",
      val->toString());
  std::stringstream ss;
  return "input " + std::to_string(pos) + ", ";
}

void validateValWithConcreteValue(
    const Val* value,
    const PolymorphicValue& concrete_value) {
  if (auto tv = dynamic_cast<const TensorView*>(value)) {
    NVF_CHECK(
        concrete_value.is<at::Tensor>(),
        "Expected ",
        getInputPosString(tv),
        tv->toString(),
        ", to be an at::Tensor but got scalar ",
        concrete_value);
    const auto& t = concrete_value.as<at::Tensor>();
    auto expect_dim =
        (int64_t)TensorDomain::noReductions(tv->getLogicalDomain()).size();
    NVF_CHECK(
        t.dim() == expect_dim,
        "Expected ",
        getInputPosString(tv),
        tv->toString(),
        ", to be bound to a tensor of rank ",
        expect_dim,
        ", but got a tensor of rank ",
        t.dim());
    auto actual_dtype = aten_to_data_type(t.scalar_type());
    NVF_CHECK(
        (value->dtype() == DataType::Index && isIntegralType(actual_dtype)) ||
            (value->dtype() == actual_dtype),
        "Expected ",
        getInputPosString(tv),
        tv->toString(),
        ", to be bound to a tensor of dtype ",
        value->dtype(),
        ", but got a tensor of dtype ",
        actual_dtype);
    // Intermediate tensorviews marked as CPU scalars will be created as meta
    // tensors during compilation. For example, for fusions containing SDPA fwd
    // and bwd, some outputs of the fwd op (philox seed, philox offset) are CPU
    // scalars.
    if (tv->isCpuScalar()) {
      NVF_CHECK(
          is_cpu_scalar(t) || is_meta_scalar(t),
          "Expected ",
          getInputPosString(tv),
          tv->toString(),
          ", to be bound to a CPU or meta scalar tensor "
          ", but got a tensor on device ",
          t.device(),
          " with ",
          t.numel(),
          " elements");
    } else {
      NVF_CHECK(
          !t.defined() || t.is_cuda() || t.is_meta(),
          "Expected ",
          getInputPosString(tv),
          tv->toString(),
          ", to be bound to a CUDA or meta tensor, but got a tensor on device ",
          t.device());
    }
  } else {
    NVF_CHECK(
        !concrete_value.is<at::Tensor>(),
        "Expected ",
        getInputPosString(value),
        value->toString(),
        ", to be a scalar but got ",
        aten_to_data_type(concrete_value.as<at::Tensor>().scalar_type()),
        " tensor of rank ",
        concrete_value.as<at::Tensor>().dim());

    NVF_CHECK(
        hasCompatibleDataType(concrete_value, value->dtype()),
        "Scalar value ",
        concrete_value,
        " is not compatible with the expected data type: ",
        value->dtype(),
        ".");
  }
}

} // namespace

void ExpressionEvaluator::bind_(
    const Val* value,
    PolymorphicValue concrete_value,
    bool evaluate_validate) {
  using namespace PolymorphicValue_functions;
  NVF_CHECK(concrete_value.hasValue(), "Cannot bind to undefined value");
  if (value->isConst()) {
    NVF_CHECK(
        value->value() == concrete_value,
        "Tried to bind to a constant value: ",
        toString(value->value()),
        " as ",
        toString(concrete_value));
    return;
  }
  validateValWithConcreteValue(value, concrete_value);
  if (evaluate_validate &&
      ir_utils::dependenciesSatisfied(value, known_values_)) {
    auto evaluated_value = evaluate(value);
    using namespace PolymorphicValue_functions;
    auto same = isSame(evaluated_value, concrete_value);
    NVF_CHECK(
        same,
        "Tried to bind to a value: ",
        getInputPosString(value),
        value->toInlineString(),
        "(which evaluated to ",
        toString(evaluated_value),
        ") as ",
        toString(concrete_value));
  }
  if (auto tv = dynamic_cast<const TensorView*>(value)) {
    const auto& t = concrete_value.as<at::Tensor>();
    auto logical_domain = TensorDomain::noReductions(tv->getLogicalDomain());
    NVF_ERROR(
        t.dim() == (int64_t)logical_domain.size(),
        "Expected ",
        getInputPosString(tv),
        tv->toString(),
        ", to be bound to a tensor of rank ",
        logical_domain.size(),
        ", but got a tensor of rank ",
        t.dim());
    for (auto i : c10::irange(t.dim())) {
      auto id = logical_domain[i];
      if (id->isBroadcast()) {
        // DIDs are ignored for broadcast.
        bind_(logical_domain[i]->extent(), 1, evaluate_validate);
        if (id->hasExpandedExtent()) {
          // Verify that t is also expanded
          NVF_ERROR(
              t.size(i) == 1 || t.stride(i) == 0,
              "IterDomain ",
              id->toString(),
              " in ",
              getInputPosString(tv),
              "TensorView ",
              tv->toString(),
              " has expanded extent but input tensor has size ",
              t.size(i),
              " and stride ",
              t.stride(i),
              " in dimension ",
              i);
          bind_(
              logical_domain[i]->expandedExtent(),
              t.size(i),
              evaluate_validate);
        }
      } else {
        if (logical_domain[i]->isDeviceDim()) {
          // Currently we have the restrictions:
          // (1) Devices parallelized axis extent == DeviceMesh's extent
          // (2) Device parallelized axis cannot be split or merged
          // Therefore, the device parallelized extents will always be allocated
          // with size 1, but the symbolic axis extent is binded with the extent
          // of the DeviceMesh
          NVF_CHECK(
              1 == t.size(i),
              "TensorView ",
              tv->toString(),
              getInputPosString(tv),
              " IterDomain ",
              id->toString(),
              "is sharded and must have size 1, but input tensor has size ",
              t.size(i));
          NVF_CHECK(
              tv->hasDeviceMesh(),
              "TV ",
              tv->toString(),
              getInputPosString(tv),
              " has an empty DeviceMesh with DID parallelization")
          bind_(
              logical_domain[i]->extent(),
              static_cast<int64_t>(tv->getDeviceMesh().size(
                  logical_domain[i]->getParallelType())),
              evaluate_validate);
        } else {
          bind_(logical_domain[i]->extent(), t.size(i), evaluate_validate);
        }
      }
    }
  }
  if (value->isA<NamedScalar>()) {
    known_named_scalars_[value->as<NamedScalar>()->name()] =
        std::move(concrete_value);
  } else {
    known_values_[value] = std::move(concrete_value);
  }
}

void ExpressionEvaluator::bind_(
    const std::string& name,
    PolymorphicValue concrete_value) {
  known_named_scalars_[name] = std::move(concrete_value);
}

void ExpressionEvaluator::bind(
    ParallelType pt,
    PolymorphicValue concrete_value) {
  NVF_ERROR(isParallelTypeThread(pt));
  if (precomputed_values_) {
    // Need to bind the thread value to integer machine
    //  in pre-computed mode.
    precomputed_values_->bindConcreteParallelTypeValue(
        pt, std::move(concrete_value));
  } else {
    bind(stringifyThreadSize(pt), std::move(concrete_value));
  }
}

const PolymorphicValue& ExpressionEvaluator::evaluate(ParallelType pt) {
  auto it = known_named_scalars_.find(stringifyThreadSize(pt));
  if (it != known_named_scalars_.end()) {
    return it->second;
  }
  return null_;
}

const PolymorphicValue& ExpressionEvaluator::evaluate(const Val* value) {
  return evaluate(value, known_values_);
}

PolymorphicValue ExpressionEvaluator::evaluate(const Val* value) const {
  std::unordered_map<const Val*, PolymorphicValue> known_values;
  return evaluate(value, known_values);
}

const PolymorphicValue& ExpressionEvaluator::evaluate(
    const Val* value,
    std::unordered_map<const Val*, PolymorphicValue>& known_values) const {
  if (precomputed_values_ && precomputed_values_->hasValidValues()) {
    if (precomputed_values_->getMaybeValueFor(value).hasValue()) {
      return precomputed_values_->getMaybeValueFor(value);
    }
  }

  std::reference_wrapper<const PolymorphicValue> maybe_concrete_value =
      getValue(value, known_values);
  if (!maybe_concrete_value.get().hasValue()) {
    if (auto def = value->definition()) {
      FUSER_PERF_SCOPE("ExpressionEvaluator::evaluate");
      auto outputs = def->evaluate(*this, known_values);
      for (auto i : c10::irange(def->outputs().size())) {
        known_values[def->output(i)] = std::move(outputs[i]);
      }
      maybe_concrete_value = getValue(value, known_values);
    }
  }
  return maybe_concrete_value;
}

const PolymorphicValue& ExpressionEvaluator::getValue(
    const Val* value,
    const std::unordered_map<const Val*, PolymorphicValue>&
        additional_known_values) const {
  if (value->isScalar() && value->isConst()) {
    return value->value();
  }

  if (value->isA<NamedScalar>()) {
    const auto it = known_named_scalars_.find(value->as<NamedScalar>()->name());
    if (it != known_named_scalars_.end()) {
      return it->second;
    }
  }

  auto it = known_values_.find(value);
  if (it != known_values_.end()) {
    return it->second;
  }

  if (&additional_known_values != &known_values_) {
    it = additional_known_values.find(value);
    return it != additional_known_values.end() ? it->second : null_;
  }

  return null_;
}

void ExpressionEvaluator::print() const {
  using namespace PolymorphicValue_functions;

  debug() << "\nEvaluation context\n";
  debug() << "--------------------\n";

  for (const auto& kv : known_values_) {
    NVF_ERROR(!kv.first->isConstScalar());
    debug() << kv.first << " = " << toString(kv.second) << " ; "
            << *kv.first->getValType() << "\n";
  }

  for (const auto& kv : known_named_scalars_) {
    debug() << kv.first << " = " << toString(kv.second) << " ;\n";
  }

  debug() << "\nPre-computed Values\n";
  if (precomputed_values_ != nullptr) {
    precomputed_values_->print();
  }
  debug() << "--------------------\n\n";
}

void ExpressionEvaluator::propagateBoundValuesThroughExactMaps(
    Fusion* fusion,
    ExactLogicalDomainMap* exact_map) {
  // We map Symbolic IterDomains here only if their extents match. This avoids
  // mapping between symbolic domains that might concretize to an (Iteration,
  // Broadcast) pair from a resolved broadcast.
  std::unique_ptr<ExactLogicalDomainMap> exact_map_ptr;
  if (exact_map == nullptr) {
    exact_map_ptr = std::make_unique<ExactLogicalDomainMap>(fusion);
    exact_map = exact_map_ptr.get();
  }
  const auto mapped_sets = exact_map->getMappedSets();

  for (const auto& set : mapped_sets.disjointSets()) {
    int64_t known_size = -1;
    std::vector<Val*> unknown_vals;
    for (const auto id : *set) {
      auto eval_val = evaluate(id->extent());
      if (eval_val.hasValue()) {
        NVF_ERROR(eval_val.is<int64_t>(), "Invalid extent value");
        int64_t this_size = eval_val.as<int64_t>();
        if (known_size != -1) {
          NVF_ERROR(
              known_size == this_size,
              "Conflicting sizes: ",
              known_size,
              ", ",
              this_size);
        } else {
          known_size = this_size;
        }
      } else {
        unknown_vals.push_back(id->extent());
      }
    }

    if (known_size == -1 || unknown_vals.empty()) {
      continue;
    }

    // Binding unknown vals to known_val
    for (auto unknown_val : unknown_vals) {
      bind(unknown_val, known_size);
    }
  }
}

ExpressionEvaluator ExpressionEvaluator::clone(IrCloner& ir_cloner) const {
  ExpressionEvaluator expr_eval;
  NVF_ERROR(
      !precomputed_values_,
      "Cannot clone ExpressionEvaluator with bound PrecomputedValues");
  for (const auto& kv : known_values_) {
    expr_eval.known_values_[ir_cloner.clone(kv.first)] = kv.second;
  }
  expr_eval.known_named_scalars_.insert(
      known_named_scalars_.begin(), known_named_scalars_.end());
  return expr_eval;
}

} // namespace nvfuser
