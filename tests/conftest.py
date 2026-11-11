"""Pytest configuration: report each Part as PASS / FAIL / INCOMPLETE."""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                      # the simulator reads data/ relative to the repository

_status: "OrderedDict[str, list]" = OrderedDict()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    # an unimplemented student function counts as "incomplete", not as a failure
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        rep.outcome = "skipped"
        rep.longrepr = (str(item.fspath), 0, f"INCOMPLETE: {call.excinfo.value}")
    if rep.when == "call" or rep.outcome != "passed":
        fn = getattr(item, "function", None)
        part = getattr(fn, "part", "other")
        pts = getattr(fn, "points", 0)
        _status.setdefault(part, []).append((item.name, rep.outcome, pts))


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_sep("=", "Project 3 public checks")
    for part, items in _status.items():
        seen = {}
        for name, outcome, pts in items:
            seen[name] = (outcome, pts)
        n_pass = sum(o == "passed" for o, _ in seen.values())
        n_inc = sum(o == "skipped" for o, _ in seen.values())
        n_fail = sum(o == "failed" for o, _ in seen.values())
        terminalreporter.write_line(f"{part:<28} {n_pass} passed, {n_fail} failed, {n_inc} incomplete")
    terminalreporter.write_line("(The Gradescope autograder also runs additional hidden tests.)")
