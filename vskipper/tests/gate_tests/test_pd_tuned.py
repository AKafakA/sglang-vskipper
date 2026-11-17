"""Owner 2026-10-27: P/D config 2 -- the pd_tuned decode band (method A). CPU only: nothing is booted."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_pd_tuned_arms_differ_from_vskipper_only_in_the_declared_band():
    from vskipper.runtime import design

    assert design.PD_TUNED_BANDS, "no pd_tuned band declared"
    for layout, band in design.PD_TUNED_BANDS.items():
        arm = design.ARMS[f"vskipper_pdtuned_{layout}"]
        assert arm["decode_kv_band"] == {"NVIDIA_A100": band} and arm["decode_kv_band_policy"] == "pd_tuned"
        assert 0 < band[0] < band[1] and band[0] == int(round(0.8 * band[1] / 1000.0)) * 1000
        rest = {k: v for k, v in arm.items() if k not in ("decode_kv_band", "decode_kv_band_policy")}
        assert rest == {k: v for k, v in design.ARMS["vskipper"].items() if k not in ("decode_kv_band", "decode_kv_band_policy")}


def test_declaration_names_the_exact_pd_tuned_band_on_every_decode_instance():
    from vskipper.runtime import design

    declared = json.loads((ROOT / "vskipper/configs/deploy/execution_differences.a100_80gb_pd_tp1.json").read_text())["declared"]
    for layout, band in design.PD_TUNED_BANDS.items():
        arm = f"vskipper_pdtuned_{layout}"
        for key in [k for k in declared if k == arm or k.startswith(arm + "__")]:
            assert declared[key]["decode_kv_band"] == {"pd_tuned": list(band)}, key
        # every instance serves the tuned arm (one served design per P/D group); a prefill instance never reads the band
        rest = lambda d: {k: v for k, v in d.items() if k != "decode_kv_band"}
        assert rest(declared[arm + "__prefill"]) == rest(declared["vskipper_pd__prefill"])


def test_boot_and_gate_accept_pd_tuned_only_as_a_declared_policy():
    runner = (ROOT / "vskipper/src/vskipper/integration/sglang/model_runner.py").read_text()
    assert 'elif _band_policy == "pd_tuned":' in runner and 'raise ValueError(f"unknown decode_kv_band_policy' in runner
    gate = (ROOT / "vskipper/src/vskipper/experiments/gates/verify_execution_differences.py").read_text()
    assert 'e["decode_kv_band_policy"] == "pd_tuned"' in gate and 'expected.get("pd_tuned")' in gate


def test_lane_b_prefill_profile_is_paper_pd_prefill_with_its_own_bootstrap_port():
    from vskipper.runtime import design

    lb, base = design.SERVED_LAUNCH_PROFILES["paper_pd_prefill_lb"], design.SERVED_LAUNCH_PROFILES["paper_pd_prefill"]
    assert lb == {**base, "disaggregation_bootstrap_port": 8999} and lb["base_gpu_id"] == 0
    for field, rule in design.SERVED_LAUNCH_EXEMPTIONS.items():
        profiles = rule.get("profiles")
        if isinstance(profiles, dict) and "paper_pd_prefill" in profiles:
            want = 8999 if field == "disaggregation_bootstrap_port" else profiles["paper_pd_prefill"]
            assert profiles["paper_pd_prefill_lb"] == want, field
