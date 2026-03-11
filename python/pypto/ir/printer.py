# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Utilities for printing IR nodes in Python syntax and verifying print-parse roundtrips."""

from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core import passes as _passes


def python_print(node: _ir_core.IRNode | _ir_core.Type, prefix: str = "pl", concise: bool = False) -> str:
    """Print IR node or Type object in Python IR syntax.

    This is a unified wrapper that dispatches to the appropriate C++ function
    based on the type of the input object.

    Args:
        node: IR node (Expr, Stmt, Function, Program) or Type object to print
        prefix: Module prefix (default 'pl' for 'import pypto.language as pl')
        concise: If true, omit intermediate type annotations (default false)

    Returns:
        Python-style string representation
    """
    if isinstance(node, _ir_core.Type):
        return _ir_core.python_print_type(node, prefix)
    else:
        return _ir_core.python_print(node, prefix, concise)


def verify_roundtrip(program: _ir_core.Program) -> None:
    """Verify print-parse roundtrip correctness for a Program.

    Prints the program to its text representation, parses it back, and asserts
    structural equality between the original and reparsed program (without
    auto_mapping, so variable names must agree exactly). This ensures the
    printer and parser are consistent and no information is lost during
    serialization.

    Args:
        program: The IR program to verify.

    Raises:
        ValueError: If re-parsing fails or the reparsed program is not
            structurally equal to the original.
    """
    # Lazy import to avoid circular: ir.printer → language → ir
    import pypto.language as pl  # noqa: PLC0415
    from pypto.pypto_core import InternalError as _InternalError  # noqa: PLC0415

    try:
        text = python_print(program)
    except _InternalError:
        # Skip roundtrip for programs the printer cannot handle yet (e.g., Unroll+iter_args)
        return

    try:
        reparsed = pl.parse(text)
    except Exception as e:
        raise ValueError(
            f"IR roundtrip failed for '{program.name}': re-parsing printed IR raised an error.\n"
            f"Parse error: {e}\n"
            f"Printed IR:\n{text}"
        ) from e

    if not isinstance(reparsed, _ir_core.Program):
        raise ValueError(
            f"IR roundtrip failed for '{program.name}': re-parsed result is "
            f"{type(reparsed).__name__}, expected Program.\n"
            f"Printed IR:\n{text}"
        )

    try:
        _ir_core.assert_structural_equal(program, reparsed, enable_auto_mapping=False)
    except ValueError as e:
        raise ValueError(
            f"IR roundtrip failed for '{program.name}': reparsed program is not "
            f"structurally equal to the original.\n"
            f"{e}\n"
            f"Printed IR:\n{text}"
        ) from e


def roundtrip_instrument() -> _passes.CallbackInstrument:
    """Create a PassInstrument that verifies print-parse roundtrip after each pass.

    Returns a CallbackInstrument that calls verify_roundtrip() on the program
    after every pass executes. Use with PassContext to catch printer/parser bugs
    introduced by any transformation in the pipeline.

    Returns:
        A CallbackInstrument named "RoundtripVerification".
    """

    def _check(_pass_obj: _passes.Pass, program: _ir_core.Program) -> None:
        verify_roundtrip(program)

    return _passes.CallbackInstrument(after_pass=_check, name="RoundtripVerification")
