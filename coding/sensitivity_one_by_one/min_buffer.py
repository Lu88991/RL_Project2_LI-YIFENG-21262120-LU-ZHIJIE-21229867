#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单参数敏感性分析入口：仅运行轴「min_buffer」。

等价命令:
  python sensitivity_analysis_random.py --only-param min_buffer

请在项目根目录执行本脚本，或从任意目录:
  python path/to/sensitivity_params/min_buffer.py
"""
from __future__ import annotations

import os
import sys

_ONLY = "min_buffer"


def main() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    extra = sys.argv[1:]
    sys.argv = [os.path.join(root, "sensitivity_analysis_random.py"), "--only-param", _ONLY, *extra]
    import sensitivity_analysis_random as _sar

    _sar.main()


if __name__ == "__main__":
    main()
