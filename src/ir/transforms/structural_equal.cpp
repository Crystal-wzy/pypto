/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/core.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/program.h"
#include "pypto/ir/reflection/field_visitor.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/printer.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

/**
 * @brief Unified structural equality checker for IR nodes
 *
 * Template parameter controls behavior on mismatch:
 * - AssertMode=false: Returns false (for structural_equal)
 * - AssertMode=true: Throws ValueError with detailed error message (for assert_structural_equal)
 *
 * This class is not part of the public API - use structural_equal() or assert_structural_equal().
 *
 * Implements the FieldIterator visitor interface for generic field-based comparison.
 * Uses the dual-node Visit overload which calls visitor methods with two field arguments.
 */
template <bool AssertMode>
class StructuralEqualImpl {
 public:
  using result_type = bool;

  explicit StructuralEqualImpl(bool enable_auto_mapping) : enable_auto_mapping_(enable_auto_mapping) {}

  // Returns bool for structural_equal, throws for assert_structural_equal
  bool operator()(const IRNodePtr& lhs, const IRNodePtr& rhs) {
    if constexpr (AssertMode) {
      Equal(lhs, rhs);
      return true;  // Only reached if no exception thrown
    } else {
      return Equal(lhs, rhs);
    }
  }

  bool operator()(const TypePtr& lhs, const TypePtr& rhs) {
    if constexpr (AssertMode) {
      EqualType(lhs, rhs);
      return true;  // Only reached if no exception thrown
    } else {
      return EqualType(lhs, rhs);
    }
  }

  // FieldIterator visitor interface (dual-node version - methods receive two fields)
  [[nodiscard]] result_type InitResult() const { return true; }

  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const IRNodePtrType& lhs, const IRNodePtrType& rhs) {
    INTERNAL_CHECK(lhs) << "structural_equal encountered null lhs IR node field";
    INTERNAL_CHECK(rhs) << "structural_equal encountered null rhs IR node field";
    return Equal(lhs, rhs);
  }

  // Specialization for std::optional<IRNodePtr>
  template <typename IRNodePtrType>
  result_type VisitIRNodeField(const std::optional<IRNodePtrType>& lhs,
                               const std::optional<IRNodePtrType>& rhs) {
    if (!lhs.has_value() && !rhs.has_value()) {
      return true;
    }
    if (!lhs.has_value() || !rhs.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("Optional field presence mismatch", lhs.has_value() ? *lhs : IRNodePtr(),
                      rhs.has_value() ? *rhs : IRNodePtr(), lhs.has_value() ? "has value" : "nullopt",
                      rhs.has_value() ? "has value" : "nullopt");
      }
      return false;
    }
    if (!*lhs && !*rhs) {
      return true;
    }
    if (!*lhs || !*rhs) {
      if constexpr (AssertMode) {
        ThrowMismatch("Optional field nullptr mismatch", *lhs, *rhs, *lhs ? "has value" : "nullptr",
                      *rhs ? "has value" : "nullptr");
      }
      return false;
    }
    return Equal(*lhs, *rhs);
  }

  template <typename IRNodePtrType>
  result_type VisitIRNodeVectorField(const std::vector<IRNodePtrType>& lhs,
                                     const std::vector<IRNodePtrType>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Vector size mismatch (" << lhs.size() << " items != " << rhs.size() << " items)";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs IR node in vector at index " << i;
      INTERNAL_CHECK(rhs[i]) << "structural_equal encountered null rhs IR node in vector at index " << i;

      if constexpr (AssertMode) {
        std::ostringstream index_str;
        index_str << "[" << i << "]";
        path_.emplace_back(index_str.str());
      }

      if (!Equal(lhs[i], rhs[i])) {
        if constexpr (AssertMode) {
          path_.pop_back();
        }
        return false;
      }

      if constexpr (AssertMode) {
        path_.pop_back();
      }
    }
    return true;
  }

  template <typename KeyType, typename ValueType, typename Compare>
  result_type VisitIRNodeMapField(const std::map<KeyType, ValueType, Compare>& lhs,
                                  const std::map<KeyType, ValueType, Compare>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Map size mismatch (" << lhs.size() << " items != " << rhs.size() << " items)";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    auto lhs_it = lhs.begin();
    auto rhs_it = rhs.begin();
    while (lhs_it != lhs.end()) {
      INTERNAL_CHECK(lhs_it->first) << "structural_equal encountered null lhs key in map";
      INTERNAL_CHECK(lhs_it->second) << "structural_equal encountered null lhs value in map";
      INTERNAL_CHECK(rhs_it->first) << "structural_equal encountered null rhs key in map";
      INTERNAL_CHECK(rhs_it->second) << "structural_equal encountered null rhs value in map";

      if (lhs_it->first->name_ != rhs_it->first->name_) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Map key mismatch ('" << lhs_it->first->name_ << "' != '" << rhs_it->first->name_ << "')";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }

      if constexpr (AssertMode) {
        std::ostringstream key_str;
        key_str << "['" << lhs_it->first->name_ << "']";
        path_.emplace_back(key_str.str());
      }

      if (!Equal(lhs_it->second, rhs_it->second)) {
        if constexpr (AssertMode) {
          path_.pop_back();
        }
        return false;
      }

      if constexpr (AssertMode) {
        path_.pop_back();
      }
      ++lhs_it;
      ++rhs_it;
    }
    return true;
  }

  // Leaf field comparisons (dual-node version)
  result_type VisitLeafField(const int& lhs, const int& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Integer value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const int64_t& lhs, const int64_t& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "int64_t value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const uint64_t& lhs, const uint64_t& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "uint64_t value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const double& lhs, const double& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "double value mismatch (" << lhs << " != " << rhs << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const std::string& lhs, const std::string& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "String value mismatch (\"" << lhs << "\" != \"" << rhs << "\")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const OpPtr& lhs, const OpPtr& rhs) {
    if (lhs->name_ != rhs->name_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Operator name mismatch ('" << lhs->name_ << "' != '" << rhs->name_ << "')";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const DataType& lhs, const DataType& rhs) {
    // INDEX and DEFAULT_CONST_INT (INT64) are treated as equivalent: the printer
    // emits raw integer literals for both, and the parser always creates INDEX.
    auto is_index_compat = [](const DataType& dt) {
      return dt == DataType::INDEX || dt == DataType::DEFAULT_CONST_INT;
    };
    if (is_index_compat(lhs) && is_index_compat(rhs)) return true;
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "DataType mismatch (" << lhs.ToString() << " != " << rhs.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const FunctionType& lhs, const FunctionType& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "FunctionType mismatch (" << FunctionTypeToString(lhs) << " != " << FunctionTypeToString(rhs)
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const ForKind& lhs, const ForKind& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ForKind mismatch (" << ForKindToString(lhs) << " != " << ForKindToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const ChunkPolicy& lhs, const ChunkPolicy& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ChunkPolicy mismatch (" << ChunkPolicyToString(lhs) << " != " << ChunkPolicyToString(rhs)
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  [[nodiscard]] result_type VisitLeafField(const LoopOrigin& lhs, const LoopOrigin& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "LoopOrigin mismatch (" << LoopOriginToString(lhs) << " != " << LoopOriginToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  [[nodiscard]] result_type VisitLeafField(const ScopeKind& lhs, const ScopeKind& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ScopeKind mismatch (" << ScopeKindToString(lhs) << " != " << ScopeKindToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const ParamDirection& lhs, const ParamDirection& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ParamDirection mismatch (" << ParamDirectionToString(lhs)
            << " != " << ParamDirectionToString(rhs) << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const std::vector<ParamDirection>& lhs, const std::vector<ParamDirection>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ParamDirection vector size mismatch (" << lhs.size() << " != " << rhs.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!VisitLeafField(lhs[i], rhs[i])) {
        return false;
      }
    }
    return true;
  }

  // Compare kwargs (vector of pairs to preserve order)
  result_type VisitLeafField(const std::vector<std::pair<std::string, std::any>>& lhs,
                             const std::vector<std::pair<std::string, std::any>>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Kwargs size mismatch (" << lhs.size() << " != " << rhs.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (lhs[i].first != rhs[i].first) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Kwargs key mismatch at index " << i << " ('" << lhs[i].first << "' != '" << rhs[i].first
              << "')";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Compare std::any values by type and content
      const auto& lhs_val = lhs[i].second;
      const auto& rhs_val = rhs[i].second;
      if (lhs_val.type() != rhs_val.type()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Kwargs value type mismatch for key '" << lhs[i].first << "'";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Type-specific comparison
      bool values_equal = true;
      if (lhs_val.type() == typeid(int)) {
        values_equal = (AnyCast<int>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<int>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(bool)) {
        values_equal = (AnyCast<bool>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<bool>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(std::string)) {
        values_equal = (AnyCast<std::string>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<std::string>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(double)) {
        values_equal = (AnyCast<double>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<double>(rhs_val, "comparing kwarg: " + lhs[i].first));
      } else if (lhs_val.type() == typeid(DataType)) {
        values_equal = (AnyCast<DataType>(lhs_val, "comparing kwarg: " + lhs[i].first) ==
                        AnyCast<DataType>(rhs_val, "comparing kwarg: " + lhs[i].first));
      }
      if (!values_equal) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Kwargs value mismatch for key '" << lhs[i].first << "'";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
    }
    return true;
  }

  result_type VisitLeafField(const MemorySpace& lhs, const MemorySpace& rhs) {
    if (lhs != rhs) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "MemorySpace mismatch (" << MemorySpaceToString(lhs) << " != " << MemorySpaceToString(rhs)
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  }

  result_type VisitLeafField(const TypePtr& lhs, const TypePtr& rhs) { return EqualType(lhs, rhs); }

  result_type VisitLeafField(const std::vector<TypePtr>& lhs, const std::vector<TypePtr>& rhs) {
    if (lhs.size() != rhs.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Type vector size mismatch (" << lhs.size() << " types != " << rhs.size() << " types)";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
      INTERNAL_CHECK(lhs[i]) << "structural_equal encountered null lhs TypePtr in vector at index " << i;
      INTERNAL_CHECK(rhs[i]) << "structural_equal encountered null rhs TypePtr in vector at index " << i;
      if (!EqualType(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  [[nodiscard]] result_type VisitLeafField(const Span& lhs, const Span& rhs) const {
    INTERNAL_UNREACHABLE << "structural_equal should not visit Span field";
    return true;  // Never reached
  }

  // Field kind hooks
  template <typename FVisitOp>
  void VisitIgnoreField([[maybe_unused]] FVisitOp&& visit_op) {
    // Ignored fields are always considered equal
  }

  template <typename FVisitOp>
  void VisitDefField(FVisitOp&& visit_op) {
    bool enable_auto_mapping = true;
    std::swap(enable_auto_mapping, enable_auto_mapping_);
    visit_op();
    std::swap(enable_auto_mapping, enable_auto_mapping_);
  }

  template <typename FVisitOp>
  void VisitUsualField(FVisitOp&& visit_op) {
    visit_op();
  }

  // Path tracking hooks called by FieldIterator::VisitFieldImpl for each field.
  // PushFieldName pushes ".name" only when not inside a transparent container.
  // Transparent containers (Program, SeqStmts, OpStmts) suppress their own field
  // names so that their vector/map element accessors ([i] / ['key']) attach directly
  // to the parent field name, producing paths like body[1] instead of body.stmts[1].
  void PushFieldName(const char* name) {
    if constexpr (AssertMode) {
      if (transparent_depth_ == 0) {
        path_.emplace_back(name);  // No dot prefix — ThrowMismatch adds '.' separators
      }
    }
  }

  void PopFieldName() {
    if constexpr (AssertMode) {
      if (transparent_depth_ == 0) {
        path_.pop_back();
      }
    }
  }

  // Combine results (AND logic)
  template <typename Desc>
  void CombineResult(result_type& accumulator, result_type field_result, [[maybe_unused]] const Desc& desc) {
    accumulator = accumulator && field_result;
  }

 private:
  bool Equal(const IRNodePtr& lhs, const IRNodePtr& rhs);
  bool EqualVar(const VarPtr& lhs, const VarPtr& rhs);
  bool EqualMemRef(const MemRefPtr& lhs, const MemRefPtr& rhs);
  bool EqualIterArg(const IterArgPtr& lhs, const IterArgPtr& rhs);
  bool EqualType(const TypePtr& lhs, const TypePtr& rhs);

  /**
   * @brief Generic field-based equality check for IR nodes using FieldIterator
   *
   * Uses the dual-node Visit overload which passes two fields to each visitor method.
   *
   * @tparam NodePtr Shared pointer type to the node
   * @param lhs_op Left-hand side node
   * @param rhs_op Right-hand side node
   * @return true if all fields are equal
   */
  template <typename NodePtr>
  bool EqualWithFields(const NodePtr& lhs_op, const NodePtr& rhs_op) {
    using NodeType = typename NodePtr::element_type;
    auto descriptors = NodeType::GetFieldDescriptors();

    return std::apply(
        [&](auto&&... descs) {
          return reflection::FieldIterator<NodeType, StructuralEqualImpl<AssertMode>,
                                           decltype(descs)...>::Visit(*lhs_op, *rhs_op, *this, descs...);
        },
        descriptors);
  }

  // Only used in assert mode for error messages
  void ThrowMismatch(const std::string& reason, const IRNodePtr& lhs, const IRNodePtr& rhs,
                     const std::string& lhs_desc = "", const std::string& rhs_desc = "") {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Structural equality assertion failed";

      if (!path_.empty()) {
        msg << " at: ";
        for (size_t i = 0; i < path_.size(); ++i) {
          msg << path_[i];
          if (i < path_.size() - 1 && path_[i + 1][0] != '[') {
            msg << ".";
          }
        }
      }
      msg << "\n\n";

      if (lhs || rhs) {
        msg << "Left-hand side:\n";
        if (lhs) {
          std::string lhs_str = PythonPrint(lhs, "pl");
          std::istringstream iss(lhs_str);
          std::string line;
          while (std::getline(iss, line)) {
            msg << "  " << line << "\n";
          }
        } else {
          msg << "  (null)\n";
        }

        msg << "\nRight-hand side:\n";
        if (rhs) {
          std::string rhs_str = PythonPrint(rhs, "pl");
          std::istringstream iss(rhs_str);
          std::string line;
          while (std::getline(iss, line)) {
            msg << "  " << line << "\n";
          }
        } else {
          msg << "  (null)\n";
        }
        msg << "\n";
      } else if (!lhs_desc.empty() || !rhs_desc.empty()) {
        msg << "Left: " << lhs_desc << "\n";
        msg << "Right: " << rhs_desc << "\n\n";
      }

      msg << "Reason: " << reason;
      throw pypto::ValueError(msg.str());
    }
  }

  bool enable_auto_mapping_;
  std::unordered_map<VarPtr, VarPtr> lhs_to_rhs_var_map_;
  std::unordered_map<VarPtr, VarPtr> rhs_to_lhs_var_map_;
  std::vector<std::string> path_;  // Only used in assert mode
  int transparent_depth_ = 0;      // Depth inside transparent containers (Program/SeqStmts/OpStmts)
};

// Type dispatch macro for generic field-based comparison.
// Saves and resets transparent_depth_ to 0 before entering EqualWithFields so that
// field names of this (non-transparent) node are always pushed into the path, even
// when Equal() is called recursively from within a transparent container's field visit.
#define EQUAL_DISPATCH(Type)                                               \
  if (auto lhs_##Type = As<Type>(lhs)) {                                   \
    auto rhs_##Type = As<Type>(rhs);                                       \
    if constexpr (AssertMode) {                                            \
      int saved_depth = transparent_depth_;                                \
      transparent_depth_ = 0;                                              \
      bool result = rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type); \
      transparent_depth_ = saved_depth;                                    \
      return result;                                                       \
    } else {                                                               \
      return rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type);        \
    }                                                                      \
  }

// Dispatch macro for transparent container nodes (Program, SeqStmts, OpStmts).
// Increments transparent_depth_ so that their field names are suppressed in the path,
// allowing vector/map element accessors ([i] / ['key']) to attach directly to the
// parent field name: e.g., body[1] instead of body.stmts[1].
#define EQUAL_DISPATCH_TRANSPARENT(Type)                                 \
  if (auto lhs_##Type = As<Type>(lhs)) {                                 \
    if constexpr (AssertMode) transparent_depth_++;                      \
    auto rhs_##Type = As<Type>(rhs);                                     \
    bool result = rhs_##Type && EqualWithFields(lhs_##Type, rhs_##Type); \
    if constexpr (AssertMode) transparent_depth_--;                      \
    return result;                                                       \
  }

// Helper: recursively flatten SeqStmts/OpStmts (transparent containers) into a flat list.
// Used to normalize IR before structural comparison.
static void CollectFlatStmts(const StmtPtr& s, std::vector<StmtPtr>& out) {
  if (auto seq = As<SeqStmts>(s)) {
    for (const auto& child : seq->stmts_) CollectFlatStmts(child, out);
  } else if (auto ops = As<OpStmts>(s)) {
    for (const auto& child : ops->stmts_) CollectFlatStmts(child, out);
  } else {
    out.push_back(s);
  }
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::Equal(const IRNodePtr& lhs, const IRNodePtr& rhs) {
  if (lhs.get() == rhs.get()) return true;

  if (!lhs || !rhs) {
    if constexpr (AssertMode) ThrowMismatch("One node is null, the other is not", lhs, rhs);
    return false;
  }

  // Normalize SeqStmts/OpStmts (transparent containers) before comparing.
  // Handles: SeqStmts([OpStmts([a,b]),c]) == SeqStmts([a,b,c]) (flattening)
  //          SeqStmts([X]) == X                                 (single-element unwrap)
  {
    auto lhs_stmt = std::dynamic_pointer_cast<const Stmt>(lhs);
    auto rhs_stmt = std::dynamic_pointer_cast<const Stmt>(rhs);
    bool lhs_transparent = lhs_stmt && (As<SeqStmts>(lhs) || As<OpStmts>(lhs));
    bool rhs_transparent = rhs_stmt && (As<SeqStmts>(rhs) || As<OpStmts>(rhs));
    if (lhs_transparent || rhs_transparent) {
      std::vector<StmtPtr> lhs_flat, rhs_flat;
      if (lhs_transparent) {
        CollectFlatStmts(lhs_stmt, lhs_flat);
      }
      if (rhs_transparent) {
        CollectFlatStmts(rhs_stmt, rhs_flat);
      }
      // One transparent, one not: check single-element unwrap
      if (lhs_transparent && !rhs_transparent) {
        if (lhs_flat.size() == 1) return Equal(lhs_flat[0], rhs);
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Node type mismatch (" << lhs->TypeName() << " != " << rhs->TypeName() << ")";
          ThrowMismatch(msg.str(), lhs, rhs);
        }
        return false;
      }
      if (rhs_transparent && !lhs_transparent) {
        if (rhs_flat.size() == 1) return Equal(lhs, rhs_flat[0]);
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Node type mismatch (" << lhs->TypeName() << " != " << rhs->TypeName() << ")";
          ThrowMismatch(msg.str(), lhs, rhs);
        }
        return false;
      }
      // Both transparent: compare flat vectors
      if (lhs_flat.size() != rhs_flat.size()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "Vector size mismatch (" << lhs_flat.size() << " items != " << rhs_flat.size() << " items)";
          ThrowMismatch(msg.str(), lhs, rhs);
        }
        return false;
      }
      for (size_t i = 0; i < lhs_flat.size(); ++i) {
        if constexpr (AssertMode) {
          std::ostringstream idx;
          idx << "[" << i << "]";
          path_.emplace_back(idx.str());
        }
        bool ok = Equal(lhs_flat[i], rhs_flat[i]);
        if constexpr (AssertMode) path_.pop_back();
        if (!ok) return false;
      }
      return true;
    }
  }

  if (lhs->TypeName() != rhs->TypeName()) {
    // Allow IterArg <-> Var cross-comparison: after pass transformations (e.g., outline_incore_scopes),
    // a captured IterArg may become a plain Var function parameter. Compare by name/type only.
    auto is_var_like = [](const std::string& name) {
      return name == "IterArg" || name == "Var" || name == "MemRef";
    };
    if (enable_auto_mapping_ && is_var_like(lhs->TypeName()) && is_var_like(rhs->TypeName())) {
      return EqualVar(std::static_pointer_cast<const Var>(lhs), std::static_pointer_cast<const Var>(rhs));
    }
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Node type mismatch (" << lhs->TypeName() << " != " << rhs->TypeName() << ")";
      ThrowMismatch(msg.str(), lhs, rhs);
    }
    return false;
  }

  // Check MemRef before IterArg and Var (MemRef inherits from Var)
  if (auto lhs_memref = As<MemRef>(lhs)) {
    auto rhs_memref = std::static_pointer_cast<const MemRef>(rhs);
    bool result = rhs_memref && EqualMemRef(lhs_memref, rhs_memref);
    return result;
  }

  // Check IterArg before Var (IterArg inherits from Var)
  if (auto lhs_iter = As<IterArg>(lhs)) {
    bool result = EqualIterArg(lhs_iter, std::static_pointer_cast<const IterArg>(rhs));
    return result;
  }

  if (auto lhs_var = As<Var>(lhs)) {
    bool result = EqualVar(lhs_var, std::static_pointer_cast<const Var>(rhs));
    return result;
  }

  // All other types use generic field-based comparison
  EQUAL_DISPATCH(ConstInt)
  EQUAL_DISPATCH(ConstFloat)
  EQUAL_DISPATCH(ConstBool)
  EQUAL_DISPATCH(Call)
  EQUAL_DISPATCH(MakeTuple)
  EQUAL_DISPATCH(TupleGetItemExpr)

  // BinaryExpr and UnaryExpr are abstract base classes matching multiple kinds
  EQUAL_DISPATCH(BinaryExpr)
  EQUAL_DISPATCH(UnaryExpr)

  EQUAL_DISPATCH(AssignStmt)
  EQUAL_DISPATCH(IfStmt)
  EQUAL_DISPATCH(YieldStmt)
  EQUAL_DISPATCH(ReturnStmt)
  EQUAL_DISPATCH(ForStmt)
  EQUAL_DISPATCH(WhileStmt)
  EQUAL_DISPATCH(ScopeStmt)
  EQUAL_DISPATCH_TRANSPARENT(SeqStmts)
  EQUAL_DISPATCH_TRANSPARENT(OpStmts)
  EQUAL_DISPATCH(EvalStmt)
  EQUAL_DISPATCH(BreakStmt)
  EQUAL_DISPATCH(ContinueStmt)
  EQUAL_DISPATCH(Function)
  EQUAL_DISPATCH_TRANSPARENT(Program)

  throw pypto::TypeError("Unknown IR node type in StructuralEqualImpl::Equal: " + lhs->TypeName());
}

#undef EQUAL_DISPATCH
#undef EQUAL_DISPATCH_TRANSPARENT

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualType(const TypePtr& lhs, const TypePtr& rhs) {
  if (lhs->TypeName() != rhs->TypeName()) {
    // Allow UnknownType <-> MemRefType: tile.alloc() return type may be UnknownType in
    // original IR but MemRefType after print/parse with a type annotation.
    auto is_memref_compat = [](const TypePtr& t) { return IsA<UnknownType>(t) || IsA<MemRefType>(t); };
    if (is_memref_compat(lhs) && is_memref_compat(rhs)) {
      return true;
    }
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Type name mismatch (" << lhs->TypeName() << " != " << rhs->TypeName() << ")";
      ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  if (auto lhs_scalar = As<ScalarType>(lhs)) {
    auto rhs_scalar = As<ScalarType>(rhs);
    if (!rhs_scalar) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for ScalarType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_scalar->dtype_ != rhs_scalar->dtype_) {
      // INDEX and DEFAULT_CONST_INT (INT64) are treated as equivalent: the printer
      // emits raw integer literals for both, and the parser always creates INDEX.
      auto is_index_compat = [](const DataType& dt) {
        return dt == DataType::INDEX || dt == DataType::DEFAULT_CONST_INT;
      };
      if (is_index_compat(lhs_scalar->dtype_) && is_index_compat(rhs_scalar->dtype_)) {
        return true;
      }
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "ScalarType dtype mismatch (" << lhs_scalar->dtype_.ToString()
            << " != " << rhs_scalar->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    return true;
  } else if (auto lhs_tensor = As<TensorType>(lhs)) {
    auto rhs_tensor = As<TensorType>(rhs);
    if (!rhs_tensor) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for TensorType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tensor->dtype_ != rhs_tensor->dtype_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TensorType dtype mismatch (" << lhs_tensor->dtype_.ToString()
            << " != " << rhs_tensor->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tensor->shape_.size() != rhs_tensor->shape_.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TensorType shape rank mismatch (" << lhs_tensor->shape_.size()
            << " != " << rhs_tensor->shape_.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tensor->shape_.size(); ++i) {
      if (!Equal(lhs_tensor->shape_[i], rhs_tensor->shape_[i])) return false;
    }
    // Compare tensor_view
    if (lhs_tensor->tensor_view_.has_value() != rhs_tensor->tensor_view_.has_value()) {
      if constexpr (AssertMode) {
        ThrowMismatch("TensorType tensor_view presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tensor->tensor_view_.has_value()) {
      const auto& lhs_tv = lhs_tensor->tensor_view_.value();
      const auto& rhs_tv = rhs_tensor->tensor_view_.value();
      // Compare valid_shape
      if (lhs_tv.valid_shape.size() != rhs_tv.valid_shape.size()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "TensorView valid_shape size mismatch (" << lhs_tv.valid_shape.size()
              << " != " << rhs_tv.valid_shape.size() << ")";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      for (size_t i = 0; i < lhs_tv.valid_shape.size(); ++i) {
        if (!Equal(lhs_tv.valid_shape[i], rhs_tv.valid_shape[i])) return false;
      }
      // Compare stride
      if (lhs_tv.stride.size() != rhs_tv.stride.size()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "TensorView stride size mismatch (" << lhs_tv.stride.size() << " != " << rhs_tv.stride.size()
              << ")";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      for (size_t i = 0; i < lhs_tv.stride.size(); ++i) {
        if (!Equal(lhs_tv.stride[i], rhs_tv.stride[i])) return false;
      }
      // Compare layout
      if (lhs_tv.layout != rhs_tv.layout) {
        if constexpr (AssertMode) {
          ThrowMismatch("TensorView layout mismatch", IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
    }
    return true;
  } else if (auto lhs_tile = As<TileType>(lhs)) {
    auto rhs_tile = As<TileType>(rhs);
    if (!rhs_tile) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for TileType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    // Compare dtype
    if (lhs_tile->dtype_ != rhs_tile->dtype_) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TileType dtype mismatch (" << lhs_tile->dtype_.ToString()
            << " != " << rhs_tile->dtype_.ToString() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    // Compare shape size and dimensions
    if (lhs_tile->shape_.size() != rhs_tile->shape_.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TileType shape rank mismatch (" << lhs_tile->shape_.size()
            << " != " << rhs_tile->shape_.size() << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tile->shape_.size(); ++i) {
      if (!Equal(lhs_tile->shape_[i], rhs_tile->shape_[i])) return false;
    }
    // Compare tile_view (normalize layout-only tile_view == no tile_view).
    // A "layout-only" tile_view has valid_shape==tile_shape, empty stride, no start_offset.
    // The tile_view may carry blayout/slayout/fractal/pad hints (e.g. from tile.move) that
    // are not preserved through variable type annotations in the printer, so roundtrip drops
    // them.  Treat such a tile_view as equivalent to no tile_view for roundtrip purposes.
    if (lhs_tile->tile_view_.has_value() != rhs_tile->tile_view_.has_value()) {
      const TileView& present_tv =
          lhs_tile->tile_view_.has_value() ? lhs_tile->tile_view_.value() : rhs_tile->tile_view_.value();
      const std::vector<ExprPtr>& tile_shape = lhs_tile->shape_;
      // Check: valid_shape == tile_shape AND stride empty AND no start_offset.
      bool layout_only = (present_tv.valid_shape.size() == tile_shape.size()) && present_tv.stride.empty() &&
                         !present_tv.start_offset;
      if (layout_only) {
        for (size_t i = 0; i < tile_shape.size(); ++i) {
          if (!Equal(present_tv.valid_shape[i], tile_shape[i])) {
            layout_only = false;
            break;
          }
        }
      }
      if (!layout_only) {
        if constexpr (AssertMode) {
          ThrowMismatch("TileType tile_view presence mismatch", IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Layout-only tile_view is semantically equivalent to no tile_view — skip inner comparison.
      return true;
    }
    if (lhs_tile->tile_view_.has_value()) {
      const auto& lhs_tv = lhs_tile->tile_view_.value();
      const auto& rhs_tv = rhs_tile->tile_view_.value();
      // Compare valid_shape
      if (lhs_tv.valid_shape.size() != rhs_tv.valid_shape.size()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "TileView valid_shape size mismatch (" << lhs_tv.valid_shape.size()
              << " != " << rhs_tv.valid_shape.size() << ")";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      for (size_t i = 0; i < lhs_tv.valid_shape.size(); ++i) {
        if (!Equal(lhs_tv.valid_shape[i], rhs_tv.valid_shape[i])) return false;
      }
      // Compare stride
      if (lhs_tv.stride.size() != rhs_tv.stride.size()) {
        if constexpr (AssertMode) {
          std::ostringstream msg;
          msg << "TileView stride size mismatch (" << lhs_tv.stride.size() << " != " << rhs_tv.stride.size()
              << ")";
          ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      for (size_t i = 0; i < lhs_tv.stride.size(); ++i) {
        if (!Equal(lhs_tv.stride[i], rhs_tv.stride[i])) return false;
      }
      // Compare start_offset
      if (!Equal(lhs_tv.start_offset, rhs_tv.start_offset)) return false;
      // Compare blayout
      if (lhs_tv.blayout != rhs_tv.blayout) {
        if constexpr (AssertMode) {
          ThrowMismatch("TileView blayout mismatch", IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Compare slayout
      if (lhs_tv.slayout != rhs_tv.slayout) {
        if constexpr (AssertMode) {
          ThrowMismatch("TileView slayout mismatch", IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Compare fractal
      if (lhs_tv.fractal != rhs_tv.fractal) {
        if constexpr (AssertMode) {
          ThrowMismatch("TileView fractal mismatch", IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
      // Compare pad
      if (lhs_tv.pad != rhs_tv.pad) {
        if constexpr (AssertMode) {
          ThrowMismatch("TileView pad mismatch", IRNodePtr(), IRNodePtr(), "", "");
        }
        return false;
      }
    }
    return true;
  } else if (auto lhs_tuple = As<TupleType>(lhs)) {
    auto rhs_tuple = As<TupleType>(rhs);
    if (!rhs_tuple) {
      if constexpr (AssertMode) {
        ThrowMismatch("Type cast failed for TupleType", IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    if (lhs_tuple->types_.size() != rhs_tuple->types_.size()) {
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "TupleType size mismatch (" << lhs_tuple->types_.size() << " != " << rhs_tuple->types_.size()
            << ")";
        ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
      }
      return false;
    }
    for (size_t i = 0; i < lhs_tuple->types_.size(); ++i) {
      if (!EqualType(lhs_tuple->types_[i], rhs_tuple->types_[i])) return false;
    }
    return true;
  } else if (IsA<MemRefType>(lhs) || IsA<UnknownType>(lhs)) {
    return true;  // Singleton type, both being MemRefType or UnknownType is sufficient
  }

  INTERNAL_UNREACHABLE << "EqualType encountered unhandled Type: " << lhs->TypeName();
  return false;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualVar(const VarPtr& lhs, const VarPtr& rhs) {
  if (!enable_auto_mapping_) {
    auto lhs_it = lhs_to_rhs_var_map_.find(lhs);
    auto rhs_it = rhs_to_lhs_var_map_.find(rhs);
    // Case 1: already mapped to the same variable
    if (lhs_it != lhs_to_rhs_var_map_.end() && rhs_it != rhs_to_lhs_var_map_.end()) {
      if (lhs_it->second == rhs && rhs_it->second == lhs) return true;
      // Pre-SSA: the same lhs Var object appears at multiple def-sites (e.g., loop variable reused
      // in all unrolled copies). Reprinted/reparsed program creates fresh Var objects per def-site,
      // so rhs_map[rhs]=lhs but lhs_map[lhs]!=rhs. Accept when names agree.
      if (rhs_it->second == lhs && lhs->name_ == rhs->name_) return true;
      if constexpr (AssertMode) {
        ThrowMismatch("Variable mapping inconsistent (without auto-mapping)",
                      std::static_pointer_cast<const IRNode>(lhs),
                      std::static_pointer_cast<const IRNode>(rhs), "var " + lhs->name_, "var " + rhs->name_);
      }
      return false;
    }
    // Case 1.5: rhs in map but lhs not — handles pre-SSA passes that create a new def VarPtr
    // for an iter_arg but leave the body referencing the old VarPtr with the same name.
    // The reparsed IR always uses one VarPtr per def-site, so rhs is already registered but
    // lhs (the old body-use VarPtr) is not. Accept if the registered lhs has the same name.
    if (rhs_it != rhs_to_lhs_var_map_.end() && lhs_it == lhs_to_rhs_var_map_.end()) {
      if (rhs_it->second->name_ == lhs->name_) {
        lhs_to_rhs_var_map_[lhs] = rhs;
        return true;
      }
    }
    // Case 1.6: lhs in map but rhs not — e.g. roundtrip where the parser creates a fresh VarPtr
    // for each occurrence of the same dynamic var (M, N) in type annotations. The first occurrence
    // registers lhs->M_rhs_1 during DefField; later occurrences (e.g. return_types_) produce
    // M_rhs_n not in rhs_map. Accept and register rhs->lhs if the already-mapped rhs has the same name.
    if (lhs_it != lhs_to_rhs_var_map_.end() && rhs_it == rhs_to_lhs_var_map_.end()) {
      if (lhs_it->second->name_ == rhs->name_) {
        rhs_to_lhs_var_map_[rhs] = lhs;
        return true;
      }
    }
    // Case 2: different variables not yet in either map — strict pointer identity required
    // when enable_auto_mapping=false.
    if (lhs.get() != rhs.get()) {
      if constexpr (AssertMode) {
        ThrowMismatch("Variable pointer mismatch (without auto-mapping)",
                      std::static_pointer_cast<const IRNode>(lhs),
                      std::static_pointer_cast<const IRNode>(rhs), "var " + lhs->name_, "var " + rhs->name_);
      }
      return false;
    }
    return true;
  }

  if (!EqualType(lhs->GetType(), rhs->GetType())) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Variable type mismatch (" << lhs->GetType()->TypeName() << " != " << rhs->GetType()->TypeName()
          << ")";
      ThrowMismatch(msg.str(), IRNodePtr(), IRNodePtr(), "", "");
    }
    return false;
  }

  auto it = lhs_to_rhs_var_map_.find(lhs);
  if (it != lhs_to_rhs_var_map_.end()) {
    if (it->second != rhs) {
      // Pre-SSA programs may reuse the same Var object at multiple def-sites (e.g., loop variable
      // assigned in each unrolled iteration). The reprinted/reparsed program creates fresh Var
      // objects at each position, so lhs maps to different rhs objects.
      // Accept if names match: register rhs -> lhs so UsualField lookups work.
      if (lhs->name_ == rhs->name_) {
        rhs_to_lhs_var_map_[rhs] = lhs;
        return true;
      }
      if constexpr (AssertMode) {
        std::ostringstream msg;
        msg << "Variable mapping inconsistent ('" << lhs->name_ << "' cannot map to both '"
            << it->second->name_ << "' and '" << rhs->name_ << "')";
        ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                      std::static_pointer_cast<const IRNode>(rhs));
      }
      return false;
    }
    return true;
  }

  auto rhs_it = rhs_to_lhs_var_map_.find(rhs);
  if (rhs_it != rhs_to_lhs_var_map_.end() && rhs_it->second != lhs) {
    // rhs is already mapped to a different lhs. If names match, allow it (pre-SSA reuse).
    if (lhs->name_ == rhs->name_) {
      return true;
    }
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "Variable mapping inconsistent ('" << rhs->name_ << "' is already mapped from '"
          << rhs_it->second->name_ << "', cannot map from '" << lhs->name_ << "')";
      ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  lhs_to_rhs_var_map_[lhs] = rhs;
  rhs_to_lhs_var_map_[rhs] = lhs;
  return true;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualMemRef(const MemRefPtr& lhs, const MemRefPtr& rhs) {
  // 1. First, compare as Var (handles variable mapping and type comparison)
  if (!EqualVar(lhs, rhs)) {
    return false;
  }

  // 2. Then, compare MemRef-specific fields (except id_ which is a naming counter)
  if (lhs->memory_space_ != rhs->memory_space_) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "MemRef memory_space mismatch (" << MemorySpaceToString(lhs->memory_space_)
          << " != " << MemorySpaceToString(rhs->memory_space_) << ")";
      ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  if (!Equal(lhs->addr_, rhs->addr_)) {
    if constexpr (AssertMode) {
      ThrowMismatch("MemRef addr mismatch", std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  if (lhs->size_ != rhs->size_) {
    if constexpr (AssertMode) {
      std::ostringstream msg;
      msg << "MemRef size mismatch (" << lhs->size_ << " != " << rhs->size_ << ")";
      ThrowMismatch(msg.str(), std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  return true;
}

template <bool AssertMode>
bool StructuralEqualImpl<AssertMode>::EqualIterArg(const IterArgPtr& lhs, const IterArgPtr& rhs) {
  // 1. First, compare as Var (handles variable mapping)
  if (!EqualVar(lhs, rhs)) {
    return false;
  }

  // 2. Then, compare IterArg-specific field: initValue_
  if (!Equal(lhs->initValue_, rhs->initValue_)) {
    if constexpr (AssertMode) {
      ThrowMismatch("IterArg initValue mismatch", std::static_pointer_cast<const IRNode>(lhs),
                    std::static_pointer_cast<const IRNode>(rhs));
    }
    return false;
  }

  return true;
}

// Explicit template instantiations
template class StructuralEqualImpl<false>;  // For structural_equal
template class StructuralEqualImpl<true>;   // For assert_structural_equal

// Type aliases for cleaner code
using StructuralEqual = StructuralEqualImpl<false>;
using StructuralEqualAssert = StructuralEqualImpl<true>;

// Public API implementation
bool structural_equal(const IRNodePtr& lhs, const IRNodePtr& rhs, bool enable_auto_mapping) {
  StructuralEqual checker(enable_auto_mapping);
  return checker(lhs, rhs);
}

bool structural_equal(const TypePtr& lhs, const TypePtr& rhs, bool enable_auto_mapping) {
  StructuralEqual checker(enable_auto_mapping);
  return checker(lhs, rhs);
}

// Public assert API
void assert_structural_equal(const IRNodePtr& lhs, const IRNodePtr& rhs, bool enable_auto_mapping) {
  StructuralEqualAssert checker(enable_auto_mapping);
  checker(lhs, rhs);
}

void assert_structural_equal(const TypePtr& lhs, const TypePtr& rhs, bool enable_auto_mapping) {
  StructuralEqualAssert checker(enable_auto_mapping);
  checker(lhs, rhs);
}

}  // namespace ir
}  // namespace pypto
