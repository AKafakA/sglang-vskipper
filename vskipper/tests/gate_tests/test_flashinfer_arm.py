"""Owner 2026-10-26 (rebuttal: "if support flash attention is possible"; option 1, FlashInfer with prefill routing only).

The FlashInfer arm `vskipper_fi` is the served design with every request decoding on the dense body (FlashInfer has no
row-gated decode kernel), served under launch profile `paper_flashinfer` on SGLang's own substrate; the paper profile's
launch is unchanged byte for byte. CPU only: nothing is booted.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import msgspec
import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("rpc_fi", ROOT / "vskipper/src/vskipper/experiments/run_paired_campaign.py")
rpc = importlib.util.module_from_spec(_spec)
sys.modules["rpc_fi"] = rpc
_spec.loader.exec_module(rpc)


@contextlib.contextmanager
def _serving(name):
    from vskipper.runtime import design

    saved = design._ARM_CACHE.get("name")
    design._ARM_CACHE["name"] = name
    try:
        yield
    finally:
        if saved is None:
            design._ARM_CACHE.pop("name", None)
        else:
            design._ARM_CACHE["name"] = saved


def test_arm_is_the_served_design_routing_prefill_only():
    from vskipper.runtime import design

    fi, served = design.ARMS["vskipper_fi"], design.ARMS["vskipper"]
    differ = ("phases", "masked_decode_attention", "admission_overrides")
    assert {k: v for k, v in fi.items() if k not in differ} == {k: v for k, v in served.items() if k not in differ}
    assert fi["phases"] == "prefill" and fi["masked_decode_attention"] is False
    assert fi["admission_overrides"] == {"decode_after_fd_prefill": "stock", "decode_after_stock_prefill": "stock",
                                         "decode_upgrade": "none"}


def test_arm_resolves_prefill_routing_with_pinning_and_a_dense_standard_decode_graph():
    """The served prefill pinning stays (band-free admission block, no decode leg); decode is not routed, so the decode
    graph runner captures ONE regime-free body on SGLang's own backend (no FD conditional graph, no masked decode)."""
    from vskipper.runtime.attestation import full_graph_conditional_graph_enabled
    from vskipper.runtime.common import (REQUEST_BODY_FD, REQUEST_BODY_STOCK, flexidepth_active_phases,
                                         regime_switch_config)
    from vskipper.runtime.config import full_graph_masked_decode_attention_enabled
    from vskipper.runtime.regime import AdmissionPinner, DecodeRegimeDispatch

    with _serving("vskipper_fi"):
        assert flexidepth_active_phases() == frozenset({"prefill"})
        assert full_graph_conditional_graph_enabled() is False
        assert full_graph_masked_decode_attention_enabled() is False
        cfg = regime_switch_config()
        assert cfg.prefill.enabled and not cfg.decode.enabled
        assert cfg.admission.enabled and cfg.admission.band_free and cfg.pinning
        pinner = AdmissionPinner(cfg)
        assert pinner.active and pinner.phase_sticky and not pinner.upgrades_enabled
        # the served prefill decision: the round's prompt tokens against the 1,536-token bracket
        assert pinner.prefill_pin(1535) == REQUEST_BODY_STOCK and pinner.prefill_pin(1536) == REQUEST_BODY_FD
        assert pinner.decode_pin_at_boundary(REQUEST_BODY_FD) == REQUEST_BODY_STOCK
        assert pinner.decode_pin_at_boundary(REQUEST_BODY_STOCK) == REQUEST_BODY_STOCK
        pinner.observe_decode_step(256, 262_144)   # no band to mirror: nothing moves
        pinner.observe_idle()
        assert pinner.counters()["band_flips"] == 0 and pinner.counters()["steps_observed"] == 0
        dispatch = DecodeRegimeDispatch(cfg)
        assert not dispatch.active and dispatch.capture_bodies() == [None]   # one regime-free capture per bucket
        assert dispatch.observe_or_pin(256, 262_144, "prod_allrun") is None  # model_runner accounts the pin instead


def test_existing_arms_resolve_as_before():
    from vskipper.runtime.common import regime_switch_config

    with _serving("vpre_binarycohort"):           # prefill-only, band-reading block: no pinning (version-1 prefill)
        cfg = regime_switch_config()
        assert not cfg.decode.enabled and not cfg.admission.enabled and not cfg.pinning
    with _serving("vskipper"):                    # the served design
        cfg = regime_switch_config()
        assert cfg.decode.enabled and cfg.admission.enabled and cfg.pinning and not cfg.admission.band_free
        assert cfg.admission.decode_after_fd_prefill == "band"
    with _serving("vskipper_noupgrade"):          # an override arm with a decode leg
        assert regime_switch_config().pinning


def test_band_reading_admission_still_requires_a_decode_leg():
    from vskipper.runtime import design
    from vskipper.runtime.common import RegimeSwitchConfig

    cfg = dict(design.SERVED_REGIME_SWITCH)
    cfg["decode"] = {**cfg["decode"], "enabled": False}
    with pytest.raises(ValueError):                # the served block reads the band
        msgspec.json.decode(json.dumps(cfg), type=RegimeSwitchConfig).validate()
    cfg["admission"] = {**cfg["admission"], **design.ARMS["vskipper_fi"]["admission_overrides"]}
    msgspec.json.decode(json.dumps(cfg), type=RegimeSwitchConfig).validate()   # band-free: valid


def test_fd_prefill_decodes_stock_whatever_the_band():
    """The new boundary value on its own, with a decode leg present: an FD prefill decodes stock even at band HIGH."""
    from vskipper.runtime import design
    from vskipper.runtime.common import REQUEST_BODY_FD, REQUEST_BODY_STOCK, RegimeSwitchConfig
    from vskipper.runtime.regime import AdmissionPinner

    cfg = dict(design.SERVED_REGIME_SWITCH)
    cfg["admission"] = {**cfg["admission"], **design.ARMS["vskipper_fi"]["admission_overrides"]}
    config = msgspec.json.decode(json.dumps(cfg), type=RegimeSwitchConfig)
    config.validate()
    pinner = AdmissionPinner(config)
    for _ in range(2):
        assert pinner.decode_pin_at_boundary(REQUEST_BODY_FD) == REQUEST_BODY_STOCK
        assert pinner.decode_pin_at_boundary(REQUEST_BODY_STOCK) == REQUEST_BODY_STOCK
        pinner.observe_decode_step(256, 262_144)   # band HIGH: still dense


def test_served_boundary_rules_are_unchanged():
    from vskipper.runtime import design
    from vskipper.runtime.common import REQUEST_BODY_FD, RegimeSwitchConfig
    from vskipper.runtime.regime import AdmissionPinner

    config = msgspec.json.decode(json.dumps(design.SERVED_REGIME_SWITCH), type=RegimeSwitchConfig)
    config.validate()
    pinner = AdmissionPinner(config)
    pinner.observe_decode_step(256, 262_144)       # band HIGH: the served FD prefill follows the band
    assert pinner.decode_pin_at_boundary(REQUEST_BODY_FD) == REQUEST_BODY_FD


def test_triton_requirement_keys_on_the_run_mask_reader():
    """The boot guard requires triton only where the FlexiDepth run mask is read (masked decode attention): on for the served
    arm, off for the FlashInfer arm, whose rows all attend through the resolved backend (attention.py zeroes the non-RUN rows)."""
    from vskipper.runtime.config import full_graph_masked_decode_attention_enabled

    with _serving("vskipper"):
        assert full_graph_masked_decode_attention_enabled() is True
    with _serving("vskipper_fi"):
        assert full_graph_masked_decode_attention_enabled() is False
    src = (ROOT / "vskipper/src/vskipper/integration/sglang/model_runner.py").read_text()     # where ref keeps the boot guard
    guard = src[src.index("only triton reads the FlexiDepth run mask") - 2500:src.index("only triton reads the FlexiDepth run mask")]
    assert "and full_graph_masked_decode_attention_enabled()" in guard


def test_masked_decode_attention_is_an_arm_field():
    from vskipper.runtime.config import full_graph_masked_decode_attention_enabled

    with _serving("vskipper"):
        assert full_graph_masked_decode_attention_enabled() is True
    with _serving("vpre_binarycohort"):           # unchanged: the field defaults to the design
        assert full_graph_masked_decode_attention_enabled() is True
    with _serving("vskipper_fi"):
        assert full_graph_masked_decode_attention_enabled() is False


def test_flashinfer_profile_is_declared_to_gate_e():
    from vskipper.runtime import design

    assert design.SERVED_LAUNCH_PROFILES["paper_flashinfer"] == {"dtype": "float16", "mem_fraction_static": 0.8}
    assert design.SERVED_ATTENTION_SUBSTRATE_BY_PROFILE == {"paper_flashinfer": "flashinfer"}
    gate_spec = importlib.util.spec_from_file_location("vdc_fi", ROOT / "vskipper/src/vskipper/experiments/gates/verify_default_conformance.py")
    vdc = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(vdc)
    resolved = vdc.resolve_exemptions(design.SERVED_LAUNCH_EXEMPTIONS, "paper_flashinfer")
    assert resolved["dtype"]["value"] == "float16" and resolved["mem_fraction_static"]["value"] == 0.8
    # a FlashInfer server at SGLang's defaults needs no backend exemption: Gate E skips fields at their default
    defaults = {"attention_backend": "flashinfer", "sampling_backend": "flashinfer", "dtype": "auto", "mem_fraction_static": 0.83}
    served = {"attention_backend": "flashinfer", "sampling_backend": "flashinfer", "dtype": "float16", "mem_fraction_static": 0.8}
    declared, undeclared = vdc.conformance(served, defaults, resolved)
    assert undeclared == [] and len(declared) == 2


def _spec(profile):
    spec = {"tree": str(ROOT), "python": sys.executable, "model_path": "/m", "upstream_tree": "/u",
            "upstream_arms": {"upstream_fi_g1024": ["--cuda-graph-max-bs", "1024"]}}
    if profile:
        spec["launch_profile"] = profile
    return spec


EXPECTED_PAPER = ["--host", "127.0.0.1", "--port", "30000", "--dtype", "float16", "--mem-fraction-static", "0.8",
                  "--attention-backend=triton", "--prefill-attention-backend=triton", "--decode-attention-backend=triton"]


@pytest.mark.parametrize("profile", [None, "paper"])
def test_paper_profile_launch_is_unchanged(profile):
    server = rpc.Server(_spec(profile), "vskipper", 30000, Path("/tmp/s.log"))
    argv = server.command()
    assert argv[argv.index("--host"):] == EXPECTED_PAPER
    assert rpc.substrate_environment(_spec(profile), {"X": "1"}) == {"X": "1", "SGLANG_IS_FLASHINFER_AVAILABLE": "false"}


def test_flashinfer_profile_launches_on_sglangs_own_substrate():
    spec = _spec("paper_flashinfer")
    fork = rpc.Server(spec, "vskipper_fi", 30000, Path("/tmp/s.log")).command()
    assert fork[fork.index("--host"):] == ["--host", "127.0.0.1", "--port", "30000", "--dtype", "float16",
                                           "--mem-fraction-static", "0.8"]
    up = rpc.Server(spec, "upstream_fi_g1024", 30000, Path("/tmp/s.log")).command()
    assert up[up.index("--host"):] == ["--host", "127.0.0.1", "--port", "30000", "--dtype", "float16",
                                       "--mem-fraction-static", "0.8", "--cuda-graph-max-bs", "1024"]
    assert rpc.substrate_environment(spec, {"X": "1", "SGLANG_IS_FLASHINFER_AVAILABLE": "false"}) == {"X": "1"}


def test_flashinfer_declaration_names_both_arms():
    declared = json.loads((ROOT / "vskipper/configs/deploy/execution_differences.a100_80gb_flashinfer.json").read_text())["declared"]
    assert declared["upstream_fi_g1024"]["decode_ladder_top"] == declared["vskipper_fi"]["decode_ladder_top"] == 1024
    assert declared["vskipper_fi"]["kv_capacity_tokens"] == "smaller_ok" and declared["vskipper_fi"]["decode_kv_band"] == "rule"
