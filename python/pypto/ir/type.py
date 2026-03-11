# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type utilities and wrappers for PyPTO IR."""

from collections.abc import Sequence

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    ConstInt,
    Expr,
    MemorySpace,
    MemRef,
    Span,
    TensorType,
    TensorView,
    TileType,
    TileView,
)

from .utils import _normalize_shape

# Store the original native __init__
_native_tensor_type_init = TensorType.__init__
_native_tile_type_init = TileType.__init__


def _tensor_type_init_wrapper(
    self,
    shape: Sequence[int | Expr],
    dtype: DataType,
    memref: MemRef | None = None,
    tensor_view: TensorView | None = None,
):
    """Wrapped __init__ for TensorType that supports integer shapes, optional MemRef and TensorView.

    Args:
        shape: Shape dimensions as a sequence of integers or Expr nodes.
               Integers are automatically converted to ConstInt(dim, DataType.INT64, Span.unknown()).
        dtype: Element data type
        memref: Optional memory reference
        tensor_view: Optional tensor view information
    """
    shape_exprs = _normalize_shape(shape)
    # Always pass all 4 arguments to native constructor (memref and tensor_view can be None)
    _native_tensor_type_init(self, shape_exprs, dtype, memref, tensor_view)


def _tile_type_init_wrapper(
    self,
    shape: Sequence[int | Expr],
    dtype: DataType,
    memref: MemRef | None = None,
    tile_view: TileView | None = None,
):
    """Wrapped __init__ for TileType that supports integer shapes, optional MemRef and TileView.

    Args:
        shape: Shape dimensions as a sequence of integers or Expr nodes.
               Integers are automatically converted to ConstInt(dim, DataType.INT64, Span.unknown()).
        dtype: Element data type
        memref: Optional memory reference
        tile_view: Optional tile view information
    """
    shape_exprs = _normalize_shape(shape)
    # Always pass all 4 arguments to native constructor (memref and tile_view can be None)
    _native_tile_type_init(self, shape_exprs, dtype, memref, tile_view)


# Monkey-patch the native TensorType.__init__ to support integer shapes
TensorType.__init__ = _tensor_type_init_wrapper

# Monkey-patch the native TileType.__init__ to support integer shapes
TileType.__init__ = _tile_type_init_wrapper

# Store the original native MemRef.__init__
_native_memref_init = MemRef.__init__


def _memref_init_wrapper(
    self,
    memory_space: MemorySpace,
    addr: int | Expr,
    size: int,
    id: int,
    span: Span | None = None,
) -> None:
    """Wrapped __init__ for MemRef that accepts int or Expr for addr.

    Args:
        memory_space: Memory space for this reference
        addr: Starting address (int or Expr; int is converted to ConstInt)
        size: Size in bytes
        id: Unique identifier
        span: Optional source span
    """
    if isinstance(addr, int):
        addr = ConstInt(addr, DataType.INT64, Span.unknown())
    if span is None:
        _native_memref_init(self, memory_space, addr, size, id)
    else:
        _native_memref_init(self, memory_space, addr, size, id, span)


# Monkey-patch MemRef.__init__ to support integer addresses
MemRef.__init__ = _memref_init_wrapper


__all__ = ["TensorType", "TileType"]
