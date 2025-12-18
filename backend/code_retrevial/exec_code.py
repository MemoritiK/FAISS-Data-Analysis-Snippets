import io
import sys
import ast
import base64
import traceback
from contextlib import redirect_stdout
from types import CodeType

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import math
import statistics
import random
import datetime
import json
from pathlib import Path



def _wrap_last_expression(code: str) -> str|CodeType:
    """
    Rewrites code so the last expression is assigned to __last_expr__
    """
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tree.body[-1] = ast.Assign(
                targets=[ast.Name(id="__last_expr__", ctx=ast.Store())],
                value=tree.body[-1].value,
            )
        ast.fix_missing_locations(tree)
        return compile(tree, "<snippet>", "exec")
    except Exception:
        return compile(code, "<snippet>", "exec")


def run_python_code(code: str):
    stdout_buffer = io.StringIO()
    figures = []
    last_expr = None

    safe_globals = {        
            # Core scientific stack
            "np": np,
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "scipy": scipy,
        
            # Common stdlib utilities
            "math": math,
            "statistics": statistics,
            "random": random,
            "datetime": datetime,
            "json": json,
            "Path": Path,
        }

    try:
        compiled = _wrap_last_expression(code)

        with redirect_stdout(stdout_buffer):
            exec(compiled, safe_globals)

        last_expr = safe_globals.get("__last_expr__")

        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight")
            plt.close(fig)

            img_buf.seek(0)
            figures.append(
                base64.b64encode(img_buf.read()).decode("utf-8")
            )
        return {
            "stdout": stdout_buffer.getvalue(),
            "last_expression": repr(last_expr) if last_expr is not None else None,
            "plots": figures,
            "error": None,
        }

    except Exception:
        return {
            "stdout": stdout_buffer.getvalue(),
            "last_expression": None,
            "plots": [],
            "error": traceback.format_exc(),
        }
