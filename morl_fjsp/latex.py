"""Small helpers to write LaTeX number macros shared by the analysis scripts."""
from __future__ import annotations

import os


def pfmt(p: float) -> str:
    """p-value for use inside math mode: 3 decimals from 0.01 on, two significant digits below,
    and m\\times10^{e} below 0.001."""
    if p != p:
        return "--"
    if p >= 0.01:
        return f"{p:.3f}"
    if p >= 0.001:
        return f"{p:.2g}"
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def merge_macros(path: str, macros: dict) -> None:
    """Add/replace \\newcommand lines in `path`, keeping macros written by other scripts."""
    lines: dict[str, str] = {}
    if os.path.exists(path):
        for ln in open(path):
            if ln.startswith("\\newcommand{\\"):
                lines[ln[len("\\newcommand{\\"):ln.index("}")]] = ln
    for k, v in macros.items():
        lines[k] = f"\\newcommand{{\\{k}}}{{{v}}}\n"
    with open(path, "w") as f:
        f.write("".join(lines.values()))
