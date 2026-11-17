"""The cohorts-without-graphs control (owner 2026-11-09, reviewer qN7Z).

One declared difference from the captured cohort control `integrated_denseprefix_fd`: the coverage stamp is off,
so the arm's uncaptured decode passes run the routed cohort body instead of the dense fall-through. Every other
arm keeps the stamp, and the two post-run gates accept the reversed evidence (routed steps all eager, no graph
replay, work counted by the route tape) only when the server attests that declaration.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from vskipper.runtime import coverage, design

VP = Path(__file__).resolve().parents[3] / "vskipper/src/vskipper/experiments"
NEW = "vskipper_ctrl_cohort_nograph"


def _load(name: str, path: Path):
    if str(VP) not in sys.path:
        sys.path.insert(0, str(VP))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Arm:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        self.saved = design._ARM_CACHE.get("name")
        design._ARM_CACHE["name"] = self.name
        coverage.reset_coverage_dense_state()
        return self

    def __exit__(self, *exc) -> None:
        if self.saved is None:
            design._ARM_CACHE.pop("name", None)
        else:
            design._ARM_CACHE["name"] = self.saved
        coverage.reset_coverage_dense_state()


def test_the_new_arm_is_the_captured_cohort_control_with_the_stamp_off():
    new, base = design.ARMS[NEW], design.ARMS["integrated_denseprefix_fd"]
    assert {key: value for key, value in new.items() if key != "coverage_dense"} == base
    assert new["coverage_dense"] is False


def test_only_the_new_arm_turns_the_stamp_off():
    for name in design.ARMS:
        with _Arm(name):
            assert design.arm_coverage_dense() is (name != NEW), name


def test_coverage_dense_enabled_follows_the_arm():
    with _Arm("vskipper"):
        assert coverage.coverage_dense_enabled() is True
    with _Arm("integrated_denseprefix_fd"):
        assert coverage.coverage_dense_enabled() is True
    with _Arm(NEW):
        assert coverage.coverage_dense_enabled() is False


def test_served_design_carries_the_declaration_and_model_design_does_not():
    # the gates read served_design; model.design stays as it was, so the served arm's rc1 identity check is unchanged
    from vskipper.runtime.common import resolved_design_attestation
    for name, want in (("vskipper", True), ("integrated_denseprefix_fd", True), (NEW, False)):
        with _Arm(name):
            assert resolved_design_attestation()["coverage_dense"] is want
            assert "coverage_dense" not in design.design_attestation()


def _runtime(declared, dispatches: int, graphed: int, eager: int, with_c3: bool = True, backend: str = "full",
             layer_rows: int = 0) -> dict:
    runtime = {
        "v3_full_graph": {"decode_backend": backend,
                          "counters": {"decode_dispatches_total": dispatches, "whole_step_graph_replays": graphed,
                                       "eager_dispatches": eager}},
        "served_design": {} if declared is None else {"coverage_dense": declared},
        "model": {"flexidepth": {"full_graph_routes": {"layer_rows": layer_rows}}},
    }
    if with_c3:
        runtime["fd_c3"] = {"counters": {"eager_skip_decode_layer_calls": 0}}
    return runtime


def test_runtime_gate_takes_the_reversed_evidence_only_when_declared():
    gate = _load("validate_runtime_after_under_test", VP / "validate_runtime_after.py")
    good = _runtime(False, 0, 0, 0, with_c3=False, backend="disabled", layer_rows=64)
    activity = gate.validate_v3_graph_activity([good])
    assert activity[0]["uncaptured_by_design"] is True and activity[0]["routed_layer_rows"] == 64
    for bad in (_runtime(False, 0, 0, 0, with_c3=False, backend="full", layer_rows=64),     # booted with decode graphs
                _runtime(False, 0, 0, 0, with_c3=False, backend="disabled", layer_rows=0)):  # nothing routed
        with pytest.raises(ValueError):
            gate.validate_v3_graph_activity([bad])
    # every other arm must still run its routed steps captured, declared or not
    for undeclared in (True, None):
        with pytest.raises(ValueError):
            gate.validate_v3_graph_activity([_runtime(undeclared, 10, 0, 10, backend="disabled", layer_rows=64)])
    captured = gate.validate_v3_graph_activity([_runtime(True, 10, 10, 0)])
    assert "uncaptured_by_design" not in captured[0]


def test_boot_refuses_the_uncaptured_control_with_decode_graphs():
    from vskipper.runtime.validation import validate_full_graph_model_configuration
    routed = list(range(16, 32))

    def call(graphs: bool) -> None:
        validate_full_graph_model_configuration(loaded_layers=routed, loaded_flexidepth_layers=routed, tp_size=1,
                                                pp_size=1, quant_config=None, environ={}, cuda_graph_enabled=graphs)

    def not_this_guard(graphs: bool) -> None:
        try:
            call(graphs)
        except ValueError as exc:   # another check may refuse a bare CPU environment, never this guard
            assert "uncaptured decode" not in str(exc)

    with _Arm(NEW):
        with pytest.raises(ValueError, match="uncaptured decode"):
            call(True)
        not_this_guard(False)
    with _Arm("integrated_denseprefix_fd"):
        not_this_guard(True)


def test_skipping_gate_reads_the_route_tape_only_when_declared(tmp_path):
    gate = _load("verify_skipping_executed_under_test", VP / "gates" / "verify_skipping_executed.py")

    def snapshot(vp: dict) -> Path:
        path = tmp_path / "server_info.json"
        path.write_text(json.dumps({"internal_states": [{"vp_runtime": vp}]}))
        return path

    routes = {"model": {"flexidepth": {"full_graph_routes": {"project_rows": 5, "run_rows": 15}}}}
    declared = {"regime_switch": {"enabled": False}, "served_design": {"coverage_dense": False}}
    assert gate.skip_and_total(snapshot({**declared, **routes})) == (5, 20, "model.flexidepth.full_graph_routes (rows)")
    with pytest.raises(SystemExit):   # declared, but no route-tape counters: refuse rather than assume
        gate.skip_and_total(snapshot(declared))
    # an always-route arm WITHOUT the declaration still needs fd_c3: the route tape alone does not pass it
    plain = {"regime_switch": {"enabled": False}, "served_design": {"coverage_dense": True}, **routes}
    with pytest.raises(SystemExit):
        gate.skip_and_total(snapshot(plain))
