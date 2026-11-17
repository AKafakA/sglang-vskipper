"""P/D groups with several prefill and decode instances (owner 2026-10-25: 2P1D = two TP1 prefill instances + one TP2
decode instance; 2026-10-26: 1P1D and 2P2D at TP1).

The router names every prefill instance with its own bootstrap port and every decode instance, each instance gets its own
port, the first decode instance stays the deployment identity on port+2, the other instances are recorded beside it, and a
role list without 'prefill' and 'decode' (or with a repeated or unknown role) is refused. CPU-only: nothing is booted.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("rpc", ROOT / "vskipper/src/vskipper/experiments/run_paired_campaign.py")
rpc = importlib.util.module_from_spec(spec)
sys.modules["rpc"] = rpc
spec.loader.exec_module(rpc)


def _spec(roles):
    return {"tree": str(ROOT), "python": "py", "upstream_tree": "/u",
            "arms": {"baseline": "upstream_pd_g512", "treatments": ["vskipper_pd"]},
            "upstream_arms": {"upstream_pd_g512": {r: [] for r in roles}},
            "pd": {"roles": roles, "bootstrap_port": 8998, "visible_devices": "0,1,2,3",
                   "prefill": {"launch_profile": "paper_pd_prefill"},
                   "prefill_b": {"launch_profile": "paper_pd_prefill_b", "bootstrap_port": 8999},
                   "decode": {"launch_profile": "paper_pd_decode_tp2"},
                   "decode_b": {"launch_profile": "paper_pd_decode_b"}}}


def test_two_prefill_instances_ports_and_router():
    server = rpc.Server(_spec(["prefill", "prefill_b", "decode"]), "upstream_pd_g512", 32051, Path("/tmp/r.log"))
    assert server.role_ports == {"prefill": 32052, "decode": 32053, "prefill_b": 32054}
    assert server.server_port == 32053 and server.prefill_roles == ("prefill", "prefill_b")
    router = server.router_command()
    assert router[router.index("--decode") + 1] == "http://127.0.0.1:32053"
    prefills = [router[i + 1:i + 3] for i, a in enumerate(router) if a == "--prefill"]
    assert prefills == [["http://127.0.0.1:32052", "8998"], ["http://127.0.0.1:32054", "8999"]]


def test_two_prefill_two_decode_ports_router_and_sidecars():
    server = rpc.Server(_spec(["prefill", "prefill_b", "decode", "decode_b"]), "upstream_pd_g512", 32051, Path("/tmp/r.log"))
    assert server.role_ports == {"prefill": 32052, "decode": 32053, "prefill_b": 32054, "decode_b": 32055}
    assert server.server_port == 32053 and server.decode_roles == ("decode", "decode_b")
    assert server.sidecar_roles == ("prefill", "prefill_b", "decode_b")
    router = server.router_command()
    decodes = [router[i + 1] for i, a in enumerate(router) if a == "--decode"]
    assert decodes == ["http://127.0.0.1:32053", "http://127.0.0.1:32055"]
    assert [router[i + 2] for i, a in enumerate(router) if a == "--prefill"] == ["8998", "8999"]
    argv = rpc.runner_command({"python": "py", "tree": "/t", "suites_dir": "/s", "source_revision": "x",
                               "upstream_tree": "/u", "arms": {"baseline": "upstream_pd_g512", "treatments": []}},
                              "upstream_pd_g512", Path("/m"), Path("/q"), Path("/c"), 32051, server_port=32053,
                              prefill_ports=(32052, 32054), decode_ports=(32055,))
    assert argv[argv.index("--server-port") + 1] == "32053"
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--prefill-server-port"] == ["32052", "32054"]
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--decode-server-port"] == ["32055"]


def test_default_group_is_one_prefill_one_decode():
    spec = _spec(["prefill", "decode"])
    del spec["pd"]["roles"]
    server = rpc.Server(spec, "upstream_pd_g512", 32051, Path("/tmp/r.log"))
    assert server.roles == ("prefill", "decode") and server.role_ports == {"prefill": 32052, "decode": 32053}
    assert server.sidecar_roles == ("prefill",)
    router = server.router_command()
    assert router.count("--prefill") == 1 and router[router.index("--prefill") + 2] == "8998"
    assert router.count("--decode") == 1


@pytest.mark.parametrize("roles", [["prefill"], ["decode"], ["prefill", "decode", "decode"], ["decode", "other"],
                                   ["prefill_b", "decode"], ["prefill", "decode_b"], ["prefill", "decode", "router"]])
def test_bad_role_lists_refused(roles):
    with pytest.raises(ValueError):
        rpc.pd_roles({"roles": roles})


def _group(tmp_path, monkeypatch, outcomes):
    """A P/D group whose boots follow `outcomes` ('clash' | 'fail' | 'ok'); nothing is launched."""
    server = rpc.Server(_spec(["prefill", "prefill_b", "decode"]), "upstream_pd_g512", 32051, tmp_path / "r.log")
    boots = []

    def enter_pd():
        outcome = outcomes[len(boots)]
        boots.append(outcome)
        if outcome == "clash":
            server.role_log("decode").write_text("torch.distributed.DistNetworkError: ... code: -98, name: EADDRINUSE\n")
            raise RuntimeError("upstream_pd_g512 decode exited during boot")
        if outcome == "fail":
            server.role_log("decode").write_text("torch.OutOfMemoryError: CUDA out of memory\n")
            raise RuntimeError("upstream_pd_g512 decode exited during boot")
        return server

    monkeypatch.setattr(server, "_enter_pd", enter_pd)
    monkeypatch.setattr(server, "__exit__", lambda *a: None)
    monkeypatch.setattr(rpc.time, "sleep", lambda s: None)
    return server, boots


def test_port_clash_boot_is_repeated(tmp_path, monkeypatch):
    server, boots = _group(tmp_path, monkeypatch, ["clash", "ok"])
    assert server.__enter__() is server and boots == ["clash", "ok"]
    assert (tmp_path / "r.decode.log.port-clash-boot1").is_file()   # the lost boot's log is kept


def test_other_boot_failure_is_not_repeated(tmp_path, monkeypatch):
    server, boots = _group(tmp_path, monkeypatch, ["fail", "ok"])
    with pytest.raises(RuntimeError):
        server.__enter__()
    assert boots == ["fail"]


def test_port_clash_boots_are_bounded(tmp_path, monkeypatch):
    server, boots = _group(tmp_path, monkeypatch, ["clash"] * 5)
    with pytest.raises(RuntimeError):
        server.__enter__()
    assert len(boots) == rpc.Server.PD_PORT_CLASH_BOOTS


def test_preflight_reads_the_decode_instances_expectation(tmp_path):
    """A P/D fork arm is attested by its decode instance (vskipper_tp2), as every cell's manifest is: the preflight looks for
    that expectation, never for one named after the group (2026-10-25: expect_vskipper_pd.json refused the chain)."""
    for name in ("expect_upstream_pd_g512.json", "expect_vskipper_tp2.json"):
        (tmp_path / name).write_text("{}\n")
    (tmp_path / "vskipper/configs/deploy/hosts").mkdir(parents=True)
    (tmp_path / "vskipper/configs/deploy/hosts/h.json").write_text("{}\n")
    for suffix in ("requests.jsonl", "metadata.jsonl"):
        (tmp_path / f"gsm8k_eqw_r18.{suffix}").write_text("")
    spec = _spec(["prefill", "prefill_b", "decode"])
    spec["pd"]["decode"]["fork_arm"] = "vskipper_tp2"
    spec.update({"tree": str(tmp_path), "python": str(tmp_path), "model_path": str(tmp_path), "suites_dir": str(tmp_path),
                 "staging_root": str(tmp_path), "expect_dir": str(tmp_path), "host_config": "vskipper/configs/deploy/hosts/h.json",
                 "datasets": {"gsm8k": {"r18": 18.0}}, "source_revision": "x"})
    assert rpc.preflight(spec) == []
    (tmp_path / "expect_vskipper_tp2.json").unlink()
    assert any("expect_vskipper_tp2.json" in p for p in rpc.preflight(spec))


@pytest.mark.parametrize("profiles", [
    ("paper_pd_prefill", "paper_pd_decode"),                                               # 1P1D, TP1 (owner 2026-10-26)
    ("paper_pd_prefill", "paper_pd_prefill_b", "paper_pd_decode_g2", "paper_pd_decode_b"),  # 2P2D, TP1
])
def test_tp1_layout_profiles_are_declared(profiles):
    """Every role's launch profile exists, its instances sit on distinct GPUs, and Gate E can account for every value the
    profile sets: each non-substrate field carries a declared exemption under that profile with the same value."""
    from vskipper.runtime import design

    gpus = [design.SERVED_LAUNCH_PROFILES[p]["base_gpu_id"] for p in profiles]
    assert len(set(gpus)) == len(gpus)
    for p in profiles:
        for field, value in design.SERVED_LAUNCH_PROFILES[p].items():
            assert design.SERVED_LAUNCH_EXEMPTIONS[field]["profiles"][p] == value, (p, field)
        assert "tp_size" not in design.SERVED_LAUNCH_PROFILES[p]


def test_80gb_tp1_declaration_covers_every_role():
    """The A100-80GB TP1 P/D declaration names both arms for every instance role of 1P1D and 2P2D."""
    import json

    declared = json.loads((ROOT / "vskipper/configs/deploy/execution_differences.a100_80gb_pd_tp1.json").read_text())["declared"]
    for arm in ("upstream_pd_g1024", "vskipper_pd"):
        for suffix in ("", "__prefill", "__prefill_b", "__decode_b"):
            assert arm + suffix in declared, arm + suffix
    assert declared["upstream_pd_g1024"]["decode_ladder_top"] == declared["vskipper_pd"]["decode_ladder_top"] == 1024
