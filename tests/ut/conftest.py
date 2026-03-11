# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared fixtures for unit tests."""

import pytest
from pypto.ir.printer import roundtrip_instrument
from pypto.pypto_core import passes


@pytest.fixture(autouse=True)
def pass_verification_context():
    """Enable BEFORE_AND_AFTER property verification and roundtrip checking for all pass executions.

    This ensures that for every pass run in a test:
    - Before execution, its required properties are verified.
    - After execution, its produced properties are verified.
    - After execution, the IR is verified to roundtrip through print/parse correctly.

    This helps keep pass property declarations accurate and catches printer/parser bugs.
    """
    with passes.PassContext(
        [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER), roundtrip_instrument()]
    ):
        yield
