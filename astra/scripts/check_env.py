# Copyright 2025 Project Astra Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry-point shim for the ``astra-check-env`` console script.

Delegates to ``scripts/check_env.py`` at the project root so that
``pip install -e .`` wires up the ``astra-check-env`` command correctly.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


def main() -> int:
    script = pathlib.Path(__file__).parent.parent.parent / "scripts" / "check_env.py"
    spec = importlib.util.spec_from_file_location("_check_env_script", script)
    if spec is None or spec.loader is None:
        print(f"ERROR: check_env.py not found at {script}", file=sys.stderr)
        return 1
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.main()  # type: ignore[attr-defined]
