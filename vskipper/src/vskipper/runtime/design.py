"""The served design and the canonical arms, as code.

[D-609] WHY THIS FILE EXISTS
---------------------------
On 2026-09-09 roughly eighteen hours of A100 time measured a design the owner had
rejected. The intended configuration was pushed in through the environment
(``EXTRA_SERVER_ENV="SGLANG_FD_FULL_GRAPH_LOW_ROW_POLICY=off"``); the variable never
reached the server; every deployment manifest recorded the old default; and every
output gate -- work identity, GR-2, Accounting-v5, bank audits -- passed while the
wrong system was measured. **A green campaign on the wrong system is
indistinguishable from a green campaign on the right one.**

The owner's ruling: *"remove the knobs and record the decisions and then run it"*, and
*"never allowed to use the env to run any experiments"* -- including arm selection.

So: **deploying this tree IS configuring the experiment.** There is no variable to fail
to apply. The design is constants here; an arm is a named, frozen definition here; host
paths come from a committed per-host file. The environment is read in exactly one
remaining place -- to *refuse* stale configuration (``validation.py``) -- because
reading env to refuse is safe and reading env to configure is not.

Changing anything in this file is a design change and belongs in the decision log.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final, Mapping

# ---------------------------------------------------------------------------
# THE SERVED DESIGN
# ---------------------------------------------------------------------------
# Both admission legs. The prefill leg's lower bound stands at 1536 -- raising it to
# 3072 was tested under the same paired protocol and refuted (D-603: gsm8k TTFT +11.4 %
# at the knee, +12.9 % in overload). There is deliberately NO ``max_tokens``: the band's
# upper gate is deleted (D-596), because a hand-set token threshold pre-empted the
# engagement gate that makes the same decision from measurement.
SERVED_REGIME_SWITCH: Final[dict[str, Any]] = {
    # [D-849, 2026-09-21] version 2 = per-request body pinning (the `admission`
    # block). The bands, thresholds and bodies below are unchanged from version 1;
    # what changed is WHEN the decision is taken: once per request at admission,
    # from the scheduler's mirror of the decode band state, and kept for the
    # request's prefill and every decode step. Version 1 selected the body per
    # pass, so a request could receive both bodies; in the natural-lane GSM8K
    # knee cell 3,163 of 3,600 served requests matched neither model's token
    # sequence and ran away at 4.7 % against the checkpoint's own 2.2 %
    # (always-route 77/3,600, upstream 4/3,600). Requests that reproduced either
    # model's sequence ran away at that model's own rate (3/292, 0/145).
    "version": 2,
    "admission": {
        "enabled": True,
        "criterion": "phase_sticky",
        "cold_start": "stock",
        "prefill_demotion": "admission",
        "mixed_step": "forced_run",
        "decode_after_fd_prefill": "band",
        # [D-849 add. 28] A dense-prefilled request decodes dense for its whole life and
        # nothing promotes it: routed generation over a dense-computed prompt is the
        # checkpoint's loop mode (served Full->FD leg on the natural gsm8k knee suite:
        # 382 runaways of 3,600 vs always-route 83, mean output 884 vs 301 tokens). The
        # legal plans are fd->fd, fd->stock and stock->stock. The prefill body is the
        # version-1 bracket on the request's prompt tokens (with the engagement
        # demotion); the uncached-token count made warm prompts dense and thereby
        # decode-dense, which is the wrong trade once dense->routed is illegal.
        # [D-849 add. 31/32, owner 07:4xZ] The served design is the dd41daaf06 line: a
        # dense-prefilled request follows the band at the boundary and may be promoted
        # once at band HIGH; prefill body from the round's PROMPT tokens (before cache
        # hits, so long-prompt workloads prefill routed and never serve dense->routed in
        # volume; gsm8k natural gate 72/69/68 vs 76/83/78 on dd41); one config for every
        # workload. The v3 restriction (add. 28) is kept as the loop-safe struct default
        # and the reference arms, not as the served design.
        "decode_after_stock_prefill": "band",
        "decode_upgrade": "band_high",
        "decode_downgrade": "none",
        "prefill_tokens": "prompt",
    },
    "prefill": {
        "enabled": True,
        "min_tokens": 1536,
        "row_correction_alpha": 0.0,
        "include_mixed": True,
        "engagement_min": 0.35,
        "engagement_probe_every": 64,
    },
    "decode": {
        "enabled": True,
        "enter_rows": 176,
        "exit_rows": 144,
        "low_body": "prod_allrun",
        "high_body": "skip",
        # [D-738] The live criterion. Declared here; asserted at boot against
        # roofline.derived_kv_band (V* = tau*BW/(s*L_r*b) = 158k on the A100-80GB PCIe ->
        # exit 160k, enter 1.25 V* = 200k). Set from the lane-2 cut1 crossover ladder
        # (131k parity / 262k win) before the rule was written; the rule reproduces it.
        "enter_kv_tokens": 200000,
        "exit_kv_tokens": 160000,
    },
}

# Routed-MLP posture. ``off`` is the design (D-589) -- the low-row body swap was a third
# occupancy threshold outside the two admission legs, and its removal measured as parity
# on gsm8k. This is the exact setting whose env override silently did nothing.
# [D-738] The decode K/V band PER DEVICE. Each entry equals roofline.derived_kv_band(key)
# (V* = tau*BW/(s*L_r*b); exit = V*, enter = 1.25 V*, 10k grid) and is asserted against the
# rule at boot; the A100 entry is the served headline design, the H100 entries are the
# rule's PREDICTIONS for the transfer row. regime_switch_config() selects the entry for the
# device it boots on; an unknown device keeps the A100 declaration and the boot assertion
# refuses it (device_roofline.json has no entry), so nothing serves silently on a new card.
SERVED_DECODE_KV_BAND_BY_DEVICE: Final[dict[str, tuple[int, int]]] = {
    "NVIDIA_A100": (160000, 200000),
    "NVIDIA_H100_HBM3": (270000, 340000),
    "NVIDIA_H100_NVL": (320000, 400000),
    # [plan v3, 2026-09-16] hardware-generality rows, each the rule's output for the device's
    # published peaks (device_roofline.json), asserted at boot and by the execution-difference
    # gate: RTX 5880 Ada (960 GB/s -> V* = 78.2k) and the A100 40 GB SXM4 part (1,555 GB/s ->
    # V* = 126.7k; keyed by memory class, roofline.band_device_key).
    "NVIDIA_RTX_5880_Ada_Generation": (80000, 100000),
    "NVIDIA_A100_40GB": (130000, 160000),
    # [plan v4, 2026-09-16 22:4xZ] RTX A6000 (768 GB/s GDDR6 -> V* = 62.6k): the fourth hardware point.
    "NVIDIA_RTX_A6000": (60000, 80000),
    # [plan v4, 2026-09-16 23:2xZ] L40S (864 GB/s GDDR6 -> V* = 70.4k): the fourth hardware point (the A6000 box could not run the cu13 substrate).
    "NVIDIA_L40S": (70000, 90000),
}

SERVED_LOW_ROW_POLICY: Final[str] = "off"

# Every routed layer runs the count-adaptive binary-cohort body (D-574). Llama-3-8B
# routes layers 16-31; the per-layer capacities and the hand-chosen split at layer 22
# were constants fitted to one dataset and a paired A/B showed they bought nothing.
SERVED_ROUTED_LAYERS: Final[tuple[int, ...]] = tuple(range(16, 32))
SERVED_LAYER_POLICY: Final[str] = "binary_cohort"

SERVED_EXECUTION_MODE: Final[str] = "full_graph"
SERVED_COMPACT_ENABLED: Final[bool] = True
SERVED_COMPACT_PHASES: Final[frozenset[str]] = frozenset(("decode", "prefill"))
SERVED_DEVICE_ROUTE_TAPE: Final[bool] = True
SERVED_DEVICE_ROUTE_DIGEST: Final[bool] = True
SERVED_ROUTE_ACCOUNTING: Final[bool] = True
SERVED_SCHEDULER_CONVERGENCE: Final[bool] = True
SERVED_MASKED_DECODE_ATTENTION: Final[bool] = True
SERVED_CONDITIONAL_GRAPH: Final[bool] = True
SERVED_FUSED_PROJECT_INPUT: Final[bool] = True
# OFF. The gate arm script arm_env_vdec_fd.sh sets this, but the CAMPAIGN's manifest
# (21 variables) never did, so every measured cell ran without it. The design must be what
# was served, not what a neighbouring script happened to export -- taking the value from the
# gate arm instead of the manifest is the same mistake at one remove.
SERVED_DEFER_PROJECT_KV: Final[bool] = False
# Not set in the served manifest either: attestation layer counters, weighted scatter,
# prefill grouped MLP, eager semantic debug.
SERVED_LAYER_COUNTERS: Final[bool] = False
SERVED_WEIGHTED_SCATTER: Final[bool] = False
SERVED_PREFILL_GROUPED_MLP: Final[bool] = False

# Gate arithmetic is a property of the CHECKPOINT, not a tuning knob. ``released`` is the
# published FlexiDepth gate (w*MLP on RUN rows, (1-w)*PROJECT on the rest). An ste_hard
# checkpoint served under ``released`` gets scaling it was never trained with and fails
# silently -- the defect that voided the 09-03 quality gates. Qwen3 arms must override.
SERVED_GATE_MODE: Final[str] = "released"

# OFF because it is not route-identical: the fused kernel's parallel reduction differs in
# the last bit from PyTorch's, enough to flip a row whose sigmoid sits within epsilon of
# the 0.5 RUN/PROJECT threshold (measured: exactly 1 row of 735,888). Deterministic, but
# it breaks exact route equality against the FlexiDepth oracle.
SERVED_FUSED_ROUTER_NORM: Final[bool] = False

# ---------------------------------------------------------------------------
# LAUNCH ARGUMENTS: SGLang's DEFAULTS, except these, by name (D-184 Gate E; D-757)
# ---------------------------------------------------------------------------
# Every headline server must resolve to what `ServerArgs(model_path=...)` computes for the
# model on the device it runs on (`vskipper/src/vskipper/experiments/gates/verify_default_conformance.py`, at boot,
# both arms). A field may differ ONLY if it is listed here with the exact served value and
# the decision that exempted it. On 2026-09-13 two undeclared non-defaults (`dtype float16`
# on bf16 checkpoints; `mem_fraction_static 0.8` vs the computed 0.83) were found in every
# headline cell since v1.3, carried from a Turing dev host -- the cross-arm gate cannot see
# what both arms share, and the D-184 gate had been lost in the refactor.
SERVED_LAUNCH_EXEMPTIONS: Final[dict[str, dict[str, Any]]] = {
    # The unified triton substrate (owner, D-187): FlexiDepth's masked/routed attention is
    # served by the triton backend in both phases; the baseline runs the SAME substrate so
    # the only cross-arm difference is the treatment.
    "attention_backend": {"value": "triton", "decision": "D-187"},
    "prefill_attention_backend": {"value": "triton", "decision": "D-187"},
    "decode_attention_backend": {"value": "triton", "decision": "D-187"},
    # Consequence of enforcing the no-FlashInfer substrate for both arms
    # (SGLANG_IS_FLASHINFER_AVAILABLE=false in the campaign driver): upstream then defaults
    # its sampler to pytorch. Same on both arms; stated here so it is declared, not hidden.
    "sampling_backend": {"value": "pytorch", "decision": "D-187/D-757"},
    # The PAPER PROTOCOL (owner, D-757 add., 2026-09-13): fp16 and a static memory fraction
    # of 0.8, fixed rather than left to SGLang's version-dependent defaults so that every
    # campaign shares one numeric format and one K/V budget. Declared HERE, by name, and
    # applied by the driver only under the "paper" launch profile below; under
    # "sglang_default" these two are not exempt and Gate E expects the computed defaults.
    # [D-773, owner 2026-09-14] A second paper profile, `paper_bf16`, serves a checkpoint in
    # bf16 with the same 0.8 budget: Qwen3-4B is bf16-native and, measured on the same suite
    # and Triton substrate, fp16 costs its UPSTREAM 3x throughput (4.1 vs 12.3 req/s), which
    # would make the boundary test a test on a crippled substrate. Llama-3-8B stays fp16
    # (D-757 add.). A "profiles" rule maps each profile to its declared value; under any
    # profile it does not name, the field is held to the default.
    "dtype": {"profiles": {"paper": "float16", "paper_bf16": "bfloat16", "paper_flashinfer": "float16",
                           "paper_tp2": "float16", "paper_tp4": "float16",
                           "paper_pd_prefill": "float16", "paper_pd_prefill_b": "float16", "paper_pd_prefill_tp2": "float16", "paper_pd_decode_tp2": "float16",
                           "paper_pd_decode": "float16", "paper_pd_decode_g2": "float16", "paper_pd_decode_b": "float16"},
              "decision": "D-757 add. / D-773"},
    "mem_fraction_static": {"profiles": {"paper": 0.8, "paper_bf16": 0.8, "paper_flashinfer": 0.8,
                                         "paper_tp2": 0.8, "paper_tp4": 0.8,
                                         "paper_pd_prefill": 0.8, "paper_pd_prefill_b": 0.8, "paper_pd_prefill_tp2": 0.8, "paper_pd_decode_tp2": 0.8,
                                         "paper_pd_decode": 0.8, "paper_pd_decode_g2": 0.8, "paper_pd_decode_b": 0.8},
                            "decision": "D-757 add. / D-773"},
    # Rebuttal TP feasibility (owner, 2026-10-24): both arms run the same tensor-parallel size.
    "tp_size": {"profiles": {"paper_tp2": 2, "paper_tp4": 4, "paper_pd_prefill_tp2": 2, "paper_pd_decode_tp2": 2},
                "decision": "rebuttal TP feasibility, owner 2026-10-24"},
    # Rebuttal P/D feasibility (owner, 2026-10-24: 1 prefill instance at TP1 + 1 decode instance at TP2 on one
    # node, both arms in the same layout). The instance role, the KV transfer engine and its bootstrap port, the GPU
    # each instance starts at, and the prefill graphs on the prefill instance (the integrated server's default
    # backend, which SGLang's P/D hook would otherwise switch off there) are the same for both arms.
    "disaggregation_mode": {"profiles": {"paper_pd_prefill": "prefill", "paper_pd_prefill_b": "prefill", "paper_pd_prefill_tp2": "prefill",
                                         "paper_pd_decode_tp2": "decode",
                                         "paper_pd_decode": "decode", "paper_pd_decode_g2": "decode", "paper_pd_decode_b": "decode"},
                            "decision": "rebuttal P/D feasibility, owner 2026-10-24"},
    # mooncake over TCP on loopback (the spec's P/D environment sets SGLANG_HOST_IP=127.0.0.1): on this node's public interface
    # its TCP sessions failed after ~1 min of load, its intra-node NVLink mode stalled, and NIXL hung at startup (2026-10-24)
    "disaggregation_transfer_backend": {"profiles": {"paper_pd_prefill": "mooncake", "paper_pd_prefill_b": "mooncake", "paper_pd_prefill_tp2": "mooncake",
                                                     "paper_pd_decode_tp2": "mooncake",
                                                     "paper_pd_decode": "mooncake", "paper_pd_decode_g2": "mooncake", "paper_pd_decode_b": "mooncake"},
                                        "decision": "rebuttal P/D feasibility, owner 2026-10-24"},
    "disaggregation_bootstrap_port": {"profiles": {"paper_pd_prefill": 8998, "paper_pd_prefill_b": 8999, "paper_pd_prefill_tp2": 8998,
                                                   "paper_pd_decode_tp2": 8998,
                                                   "paper_pd_decode": 8998, "paper_pd_decode_g2": 8998, "paper_pd_decode_b": 8998},
                                      "decision": "rebuttal P/D feasibility, owner 2026-10-24"},
    # decode instance from GPU 2. Owner 2026-10-25: 2P1D, two TP1 prefill instances on GPUs 0 and 1 (paper_pd_prefill,
    # paper_pd_prefill_b) feeding one TP2 decode instance; the TP1 -> TP2 KV transfer uses SGLang's staging buffer (the spec's
    # P/D environment, both arms). paper_pd_prefill_tp2 is the earlier TP2 + TP2 layout (prefill on GPUs 0-1).
    # Owner 2026-10-26: P/D with every instance at TP1 on the four-GPU A100-80GB node -- 1P1D (prefill GPU 0, decode GPU 1:
    # paper_pd_decode) and 2P2D (prefill GPUs 0 and 1, decode GPUs 2 and 3: paper_pd_decode_g2, paper_pd_decode_b); TP1 -> TP1
    # transfers take mooncake's plain path, so the staging buffer is not used.
    "base_gpu_id": {"profiles": {"paper_pd_prefill": 0, "paper_pd_prefill_b": 1, "paper_pd_prefill_tp2": 0, "paper_pd_decode_tp2": 2,
                                 "paper_pd_decode": 1, "paper_pd_decode_g2": 2, "paper_pd_decode_b": 3},
                    "decision": "rebuttal P/D feasibility, owner 2026-10-24"},
    "cuda_graph_backend_prefill": {"profiles": {"paper_pd_prefill": "breakable", "paper_pd_prefill_b": "breakable", "paper_pd_prefill_tp2": "breakable"},
                                   "decision": "rebuttal P/D feasibility: prefill graphs on the prefill instance, both arms"},
    # Consequence of --tp-size 4 on the 40 GB card: SGLang's own memory-derived decode graph range is 160 rows there
    # (32 at TP1, the bare default this gate computes; 32 at TP2 as well). The served routed ladder (1,024) is attested in
    # vp_runtime and held equal across arms by the execution-difference gate; this is upstream's resolved value at TP4.
    "cuda_graph_config": {"profiles": {"paper_tp4": {"accept_if": {"decode.max_bs": 160, "decode.bs[max]": 160}},
                                       "paper_pd_prefill": {"accept_if": {"decode.backend": "disabled",
                                                                          "prefill.backend": "breakable"}},
                                       "paper_pd_prefill_b": {"accept_if": {"decode.backend": "disabled",
                                                                            "prefill.backend": "breakable"}},
                                       "paper_pd_prefill_tp2": {"accept_if": {"decode.backend": "disabled",
                                                                              "prefill.backend": "breakable"}}},
                          "decision": "rebuttal TP feasibility: SGLang's TP4 default decode range on the 40 GB card; P/D: a "
                                      "prefill instance never decodes (SGLang's P/D hook) and keeps breakable prefill graphs"},
    # What SGLang's own P/D hook and defaults resolve from the declared role (server_args / pd_disaggregation_hook), both arms:
    # a prefill instance switches CUDA graphs off (the prefill graphs are restored by cuda_graph_backend_prefill above) and
    # balances by bootstrap room; a decode instance serves from a chunk cache and reserves its in-transfer slots.
    "disable_cuda_graph": {"profiles": {"paper_pd_prefill": True, "paper_pd_prefill_b": True, "paper_pd_prefill_tp2": True},
                           "decision": "rebuttal P/D feasibility: SGLang's P/D hook on the prefill instance"},
    "load_balance_method": {"profiles": {"paper_pd_prefill": "follow_bootstrap_room", "paper_pd_prefill_b": "follow_bootstrap_room", "paper_pd_prefill_tp2": "follow_bootstrap_room"},
                            "decision": "rebuttal P/D feasibility: SGLang's default on a prefill instance"},
    "disable_radix_cache": {"profiles": {"paper_pd_decode_tp2": True, "paper_pd_decode": True, "paper_pd_decode_g2": True,
                                         "paper_pd_decode_b": True},
                            "decision": "rebuttal P/D feasibility: SGLang's P/D hook (decode instance: chunk cache)"},
    "disaggregation_decode_extra_slots": {"profiles": {"paper_pd_decode_tp2": 0, "paper_pd_decode": 0, "paper_pd_decode_g2": 0,
                                                       "paper_pd_decode_b": 0},
                                          "decision": "rebuttal P/D feasibility: SGLang's P/D hook default for this pool"},
}

# A LAUNCH PROFILE is the set of launch arguments the driver passes beyond the substrate
# (both arms, every boot). Two exist, both declared, neither an environment variable:
#   paper           -- the paper's protocol (D-757 add.): fp16, static memory fraction 0.8
#   sglang_default  -- nothing: SGLang's computed defaults (the cluster backup line, D-759)
# The spec names one (`launch_profile`); absent, the paper profile is used. Gate E verifies
# the served values against the defaults with the profile's exemptions applied, so a profile
# cannot smuggle a value: every non-default it produces must also be listed above.
SERVED_LAUNCH_PROFILES: Final[dict[str, dict[str, Any]]] = {
    "paper": {"dtype": "float16", "mem_fraction_static": 0.8},
    "paper_bf16": {"dtype": "bfloat16", "mem_fraction_static": 0.8},   # D-773: Qwen3-4B (native dtype)
    "paper_tp2": {"dtype": "float16", "mem_fraction_static": 0.8, "tp_size": 2},   # TP feasibility
    "paper_tp4": {"dtype": "float16", "mem_fraction_static": 0.8, "tp_size": 4},   # TP feasibility
    # P/D feasibility: the two instances of one arm (prefill at TP1 from GPU 0, decode at TP2 from GPU 1)
    "paper_pd_prefill": {"dtype": "float16", "mem_fraction_static": 0.8, "disaggregation_mode": "prefill",
                         "disaggregation_transfer_backend": "mooncake", "disaggregation_bootstrap_port": 8998,
                         "base_gpu_id": 0, "cuda_graph_backend_prefill": "breakable"},
    # owner 2026-10-25: 2P1D = two TP1 prefill instances (GPU 0 and GPU 1, own bootstrap ports) + one TP2 decode instance (GPUs 2-3)
    "paper_pd_prefill_b": {"dtype": "float16", "mem_fraction_static": 0.8, "disaggregation_mode": "prefill",
                           "disaggregation_transfer_backend": "mooncake", "disaggregation_bootstrap_port": 8999,
                           "base_gpu_id": 1, "cuda_graph_backend_prefill": "breakable"},
    "paper_pd_prefill_tp2": {"dtype": "float16", "mem_fraction_static": 0.8, "tp_size": 2, "disaggregation_mode": "prefill",
                             "disaggregation_transfer_backend": "mooncake", "disaggregation_bootstrap_port": 8998,
                             "base_gpu_id": 0, "cuda_graph_backend_prefill": "breakable"},
    "paper_pd_decode_tp2": {"dtype": "float16", "mem_fraction_static": 0.8, "tp_size": 2, "disaggregation_mode": "decode",
                            "disaggregation_transfer_backend": "mooncake", "disaggregation_bootstrap_port": 8998,
                            "base_gpu_id": 2},
    # owner 2026-10-26: TP1 decode instances on the A100-80GB node (1P1D: GPU 1; 2P2D: GPUs 2 and 3)
    "paper_pd_decode": {"dtype": "float16", "mem_fraction_static": 0.8, "disaggregation_mode": "decode",
                       "disaggregation_transfer_backend": "mooncake", "disaggregation_bootstrap_port": 8998, "base_gpu_id": 1},
    "paper_pd_decode_g2": {"dtype": "float16", "mem_fraction_static": 0.8, "disaggregation_mode": "decode",
                          "disaggregation_transfer_backend": "mooncake", "disaggregation_bootstrap_port": 8998, "base_gpu_id": 2},
    "paper_pd_decode_b": {"dtype": "float16", "mem_fraction_static": 0.8, "disaggregation_mode": "decode",
                         "disaggregation_transfer_backend": "mooncake", "disaggregation_bootstrap_port": 8998, "base_gpu_id": 3},
    #: [owner 2026-10-26, rebuttal: "if support flash attention is possible"] the paper protocol on SGLang's OWN attention
    #: substrate (FlashInfer, the A100 default, and its default sampler) instead of the unified Triton one -- see
    #: SERVED_ATTENTION_SUBSTRATE_BY_PROFILE. Both arms; the fork arm is vskipper_fi.
    "paper_flashinfer": {"dtype": "float16", "mem_fraction_static": 0.8},
    "sglang_default": {},
}

# [owner 2026-10-27: the pd_tuned 1P1D beside the paper-setting one] lane b's prefill instance = paper_pd_prefill with its own bootstrap
# port (8999): a second 1P1D group then runs under CUDA_VISIBLE_DEVICES 2,3 (device 0 = its prefill GPU, device 1 = its decode GPU via
# paper_pd_decode) and never touches the first group's GPUs. (Seeing GPU 0 as its device 0, an earlier mapping put CUDA contexts on the
# first group's prefill GPU and shrank that instance's K/V pool by 2.7k tokens.) Every exemption paper_pd_prefill carries applies.
SERVED_LAUNCH_PROFILES["paper_pd_prefill_lb"] = {**SERVED_LAUNCH_PROFILES["paper_pd_prefill"], "disaggregation_bootstrap_port": 8999}
for _field, _rule in SERVED_LAUNCH_EXEMPTIONS.items():
    _profiles = _rule.get("profiles")
    if isinstance(_profiles, dict) and "paper_pd_prefill" in _profiles:
        _profiles["paper_pd_prefill_lb"] = 8999 if _field == "disaggregation_bootstrap_port" else _profiles["paper_pd_prefill"]
del _field, _rule, _profiles
#: Tensor-parallel sizes the routed runtime serves (FlexiDepth Llama; projector bottleneck 896
#: and intermediate 14336 split evenly). The arm declares its size; the boot refuses a mismatch.
SERVED_TP_SIZES: Final[tuple[int, ...]] = (1, 2, 4)
# [owner 2026-11-06, reviewers qN7Z and TQAB] Control and variant profiles, each the paper profile plus one stock flag on BOTH arms.
SERVED_LAUNCH_EXEMPTIONS["cuda_graph_backend_decode"] = {
    "profiles": {"paper_nodecodegraph": "disabled"},
    "decision": "execution control: no decode graphs on both arms",
}
SERVED_LAUNCH_EXEMPTIONS["cuda_graph_config"]["profiles"]["paper_nodecodegraph"] = {"accept_if": {"decode.backend": "disabled"}}
SERVED_LAUNCH_EXEMPTIONS["cuda_graph_config"]["profiles"]["paper_chunk2048"] = {"accept_if": {"prefill.max_bs": 2048, "prefill.bs[max]": 2048}}
SERVED_LAUNCH_EXEMPTIONS["cuda_graph_config"]["profiles"]["paper_chunk1024"] = {"accept_if": {"prefill.max_bs": 1024, "prefill.bs[max]": 1024}}
SERVED_LAUNCH_EXEMPTIONS["chunked_prefill_size"] = {
    "profiles": {"paper_chunk2048": 2048, "paper_chunk1024": 1024},
    "decision": "chunked-prefill variant: the same chunk size on both arms",
}
for _name, _extra in (("paper_nodecodegraph", {"cuda_graph_backend_decode": "disabled"}),
                      ("paper_chunk2048", {"chunked_prefill_size": 2048}),
                      ("paper_chunk1024", {"chunked_prefill_size": 1024})):
    SERVED_LAUNCH_PROFILES[_name] = {**SERVED_LAUNCH_PROFILES["paper"], **_extra}
    for _field, _rule in SERVED_LAUNCH_EXEMPTIONS.items():
        _profiles = _rule.get("profiles")
        if isinstance(_profiles, dict) and "paper" in _profiles:
            _profiles[_name] = _profiles["paper"]
del _name, _extra, _field, _rule, _profiles
SERVED_LAUNCH_PROFILE_DEFAULT: Final[str] = "paper"

# The attention SUBSTRATE each launch profile runs on, both arms. "triton" is the unified substrate of every paper cell
# (D-187: the driver passes --attention-backend/--prefill-attention-backend/--decode-attention-backend=triton and
# SGLANG_IS_FLASHINFER_AVAILABLE=false, declared above as exemptions); "flashinfer" passes NO backend flag and leaves
# FlashInfer available, so both arms resolve SGLang's own defaults on the card (FlashInfer attention + sampler on the A100),
# which Gate E holds to the computed defaults. A profile not named here is "triton".
SERVED_ATTENTION_SUBSTRATE_BY_PROFILE: Final[dict[str, str]] = {"paper_flashinfer": "flashinfer"}

# ---------------------------------------------------------------------------
# THE CANONICAL ARMS
# ---------------------------------------------------------------------------
# An arm is WHAT IS BEING MEASURED; the design above is HOW it is served, and does not
# vary between arms. Selected by name -- never by exported variables. These replace
# test/vp/gates/arms/arm_env_*.sh, which are deleted.
ARMS: Final[dict[str, dict[str, Any]]] = {
    # OUR FORK WITH THE SKIPPER OFF. Read that again: this is NOT upstream SGLang, and this
    # comment used to claim it was ("Stock upstream SGLang ... the paper's baseline arm"),
    # which is how at least three sessions concluded the baseline was already upstream and
    # built campaigns on it. It is this tree -- same 193 commits, same hook sites in
    # models/llama.py, same vpipe package imported -- with the treatment not applied.
    #
    # THE PAPER'S BASELINE IS NOT THIS ARM. It is a separate freshly-cloned upstream tree at
    # commit 602c8615a1 carrying no vpipe/ package at all, staged and content-verified
    # against vskipper/configs/deploy/upstream_baseline.json and served by its own PYTHONPATH (owner order
    # D-587: "it has to be the based directly-freshly cloned sglang without our changes").
    #
    # This arm's legitimate role is the CONTROL that measures what our fork costs when it is
    # not treating anything -- D-597 found +0.06 %/+0.14 % output TPS against fresh upstream,
    # i.e. nothing. But that was measured at 150 commits past upstream and HEAD is now 193,
    # and an equivalence carried across a design change is what voided the v1.3 table. Serve
    # upstream for the baseline and this becomes a control rather than a load-bearing
    # assumption.
    #
    # TODO (owner 2026-09-10): rename this key. NOT to "upstream" -- that would make the old
    # lie permanent -- but to something that says what it is, e.g. "fork_noskip". Deferred
    # only because the name is written into vskipper/configs/deploy/active_arm, expect_stock.json and every
    # manifest already on the boxes.
    "stock": {"skipper": None, "phases": None, "regime_switch": False},
    # FlexiDepth, decode phase only.
    "vdec_fd": {"skipper": "flexidepth", "phases": "decode", "regime_switch": True},
    # FlexiDepth, prefill phase only.
    "vpre_binarycohort": {"skipper": "flexidepth", "phases": "prefill", "regime_switch": True},
    # FlexiDepth, both phases -- THE SERVED SYSTEM the paper reports (owner 2026-09-12: "use the
    # vskipper instead" of the historical lane-2 name integrated_it4). The old name stays an
    # ALIAS of the same dict below, so campaigns, specs, expectation files and manifests that
    # carry it (the A100 headline on tree 735d451c89) keep resolving to the identical design.
    "vskipper": {"skipper": "flexidepth", "phases": "both", "regime_switch": True},
    # [D-627] THE QUALITY POSTURE. Identical to integrated_it4 except that admission is OFF,
    # so every routed pass routes regardless of load.
    #
    # This restores what the deleted arm_env_*.sh scripts did: they exported NO
    # SGLANG_VP_REGIME_SWITCH, so the mechanism was off and the arm always routed. The D-611
    # refactor set regime_switch=True on every routed arm, and Codex flagged it (review 1, P1
    # F4) -- I dismissed it as matching the campaign manifest. It does match the CAMPAIGN. It
    # does not match the QUALITY lane, and the two need opposite postures:
    #
    #   performance : admission ON  -- measures the served design under real load
    #   quality     : admission OFF -- measures the SKIPPER; quality must not depend on load
    #
    # Measured consequence of getting this wrong: an lm-eval run at concurrency 16 sits far
    # below enter_rows=176, so the decode leg stays in prod_allrun. A quality run on
    # 2026-09-09 executed 16,702 decode passes with skip=0 -- it measured a system that never
    # skipped, and every quality number from it is void.
    "integrated_alwaysskip": {
        "skipper": "flexidepth",
        "phases": "both",
        "regime_switch": False,
    },
    # [D-849 add. 8, 2026-09-21] The legal-plan ablation's Full->FD arm: the prompt is
    # encoded by the dense body (skipper gated off for prompt tokens), every generated
    # token by FlexiDepth's native router -- the "stock prefill -> FD decode" plan the
    # phase-sticky controller can choose, served for EVERY request. No band.
    "integrated_denseprefix_fd": {
        "skipper": "flexidepth",
        "phases": "decode",
        "regime_switch": False,
    },
    # [D-849 add. 9] The FD->stock legal plan served for EVERY request: the prompt is
    # encoded by FlexiDepth's native routing, every generated token by the dense body
    # (skipper gated off for generated tokens) -- the mirror of the dense-prefix plan.
    "integrated_fdprefix_stock": {
        "skipper": "flexidepth",
        "phases": "prefill",
        "regime_switch": False,
    },
    # Arbitrary per-token routes with no semantics: the substrate-generality arm. Proves
    # the runtime assumes nothing about the policy that produced a route, and carries the
    # skip-rate x depth trade-off study.
    #
    # [D-619] BOTH phases, and named to say so. It was `vdec_randomskip`, decode-only, but
    # the study overlays FlexiDepth's operating point on this surface and the served system
    # (integrated_it4) routes both phases -- a decode-only mock would confound skip
    # magnitude with phase coverage. The mock itself is phase-agnostic: prepare_batch
    # `del phase` and derives routes from (request_ids, token_epochs), so this is
    # configuration, not new code. The old name would have been a label asserting something
    # the system does not do.
    "integrated_randomskip": {
        "skipper": "deterministic_mock",
        "phases": "both",
        "regime_switch": True,
        # These three are NOT free parameters: they are the values the deleted
        # arm_env_vdec_randomskip.sh exported, carried over verbatim. Choosing new ones
        # while "porting" the arm would be a silent design change (global rule R3) that
        # makes the generality arm incomparable to every earlier RandomSkip result.
        "mock_token_skip_rate": 0.5,
        "mock_skipped_depth_ratio": 0.5,
        "mock_seed": 1234,
    },
}
# [owner 2026-09-12] Deprecated name of the served system; identical design object.
ARMS["integrated_it4"] = ARMS["vskipper"]

# [owner 2026-11-06, reviewer qN7Z] The three-step execution control: same checkpoint, same router, decode only, always-route, so the
# steps differ only in how a routed decode layer executes. `integrated_denseprefix_fd` is the third step (packed cohorts in capture).
# Full-width: the MLP and the projector run on every row inside the captured graph, selected by the route mask.
ARMS["vskipper_ctrl_fullwidth"] = {
    "skipper": "flexidepth",
    "phases": "decode",
    "regime_switch": False,
    "low_row_policy": "native_dense",
}
# How the per-step cost scales with the number of routed layers (reviewer TQAB, Q1): the same decode-only arm on the last 8 and last 4 layers.
ARMS["vskipper_ctrl_routed8"] = {**ARMS["integrated_denseprefix_fd"], "routed_layers": tuple(range(24, 32))}
ARMS["vskipper_ctrl_routed4"] = {**ARMS["integrated_denseprefix_fd"], "routed_layers": tuple(range(28, 32))}
# [owner 2026-11-06, reviewer qN7Z] The same control on the prefill path, where a batch carries thousands of tokens through the MLP:
# always-route, packed cohorts or the full-width body. Both phases, because the full-width posture requires decode among them; the
# measurement sends one output token, so only prefill runs.
ARMS["vskipper_ctrl_prefill_cohort"] = {"skipper": "flexidepth", "phases": "both", "regime_switch": False}
ARMS["vskipper_ctrl_prefill_fullwidth"] = {**ARMS["vskipper_ctrl_prefill_cohort"], "low_row_policy": "native_dense"}
# [owner 2026-11-07] The served design with the full-width body in place of packed cohorts: the same question end to end, mode switch on.
# The body also feeds the route counts the prefill engagement escape reads, so the mode switch behaves as on the served arm.
ARMS["vskipper_fullwidth"] = {**ARMS["vskipper"], "low_row_policy": "native_dense", "fullwidth_engagement_counts": True}
# [owner 2026-11-14, repetition fix] The served design with a post-trained router/projector file (the GRPO candidate): identical to
# `vskipper` in every design field; only the weights key differs. The host file's `flexidepth_weights_cand_grpo` names the candidate
# file the job staged, and the campaign records its sha256 at stage-in. The decode band is the released checkpoint's (inherited).
ARMS["vskipper_cand_grpo"] = {**ARMS["vskipper"], "weights_key": "flexidepth_weights_cand_grpo"}
# Simple skipping: vpipe/eager.py, no capture, no cohorts, a full attention read on every row, row gather per layer. Launch profile
# paper_nodecodegraph, against upstream on the same profile.
ARMS["vskipper_ctrl_eager"] = {
    "skipper": "flexidepth",
    "phases": "decode",
    "regime_switch": False,
    "execution_mode": "direct_eager",
}

# [owner 2026-11-09, reviewer qN7Z] Packed cohorts WITHOUT decode CUDA graphs: the captured cohort control
# (`integrated_denseprefix_fd`: decode only, always-route) with the coverage stamp off, served only under profile
# `paper_nodecodegraph` against `upstream_nograph`. With no captured decode graph every decode pass is uncovered,
# and with the stamp off the full-graph cohort body runs on each of them uncaptured, the path uncovered passes
# took before the coverage stamp existed. It isolates the graphs' share with the cohorts held fixed, as the
# existing controls isolate the cohorts' share with the graphs held fixed.
ARMS["vskipper_ctrl_cohort_nograph"] = {**ARMS["integrated_denseprefix_fd"], "coverage_dense": False}

# [D-743] v1.4.5 mechanism ablation: the served design with compaction OFF (arm field; every
# other field identical to `vskipper`). Never served in a headline cell.
ARMS["vskipper_nocompact"] = {
    "skipper": "flexidepth", "phases": "both", "regime_switch": True, "compact": False,
}

# [D-744/D-745] v1.5 feasibility row: FlexiDepth-Qwen3-4B, the alignment-only `ste_hard`
# checkpoint (sealed penalty 1e-5; every other 4B/8B/14B checkpoint of the Aug 27-Sep 3 window
# is broken and is never served). Arm fields, all of them design, none of them environment:
#   gate_mode     hard_mask  -- the straight-through gate's forward: hard selection, NO w scaling
#   compact       False      -- compact routed-K/V repair does not support Qwen3's per-head q/k
#                              norms yet (graphs.py); dense repair is exact, just slower
#   routed_layers 18..35     -- what the checkpoint trained (36 layers, routing_layers in its
#                              config.json); the family's range, not Llama's 16..31
# The always-route twin is the quality posture (arm D of the 2x2) and the correctness
# configuration for cards without a K/V band entry (no regime switch, no band assertion).
#   weights_key   the host-file key naming this checkpoint's router/projector file (the Llama
#                 file stays under `flexidepth_weights`; one host file, one key per checkpoint)
#   design_skip_ratio 0.25 -- the alignment-only checkpoint's measured skip on the routed layers
#                 (training gate, D-3xx); the K/V-band rule's `s`
#   decode_kv_band  per device, from the same rule as the Llama band with this arm's inputs
#                 (L_r = 18, b = 4 KB, s = 0.25, tau scaled by routed-layer count since the tax is
#                 per routed layer, D-370): A100 V* = 315k -> exit 320k, enter 390k. Asserted at
#                 boot against the rule; a device without an entry refuses.
ARMS["vskipper_qwen3_4b"] = {
    "skipper": "flexidepth", "phases": "both", "regime_switch": True,
    "gate_mode": "hard_mask", "compact": False, "routed_layers": tuple(range(18, 36)),
    "weights_key": "flexidepth_weights_qwen3_4b", "design_skip_ratio": 0.25,
    "decode_kv_band": {"NVIDIA_A100": (320000, 390000)},
}
ARMS["vskipper_qwen3_4b_alwaysroute"] = {
    "skipper": "flexidepth", "phases": "both", "regime_switch": False,
    "gate_mode": "hard_mask", "compact": False, "routed_layers": tuple(range(18, 36)),
    "weights_key": "flexidepth_weights_qwen3_4b", "design_skip_ratio": 0.25,
}

# [D-778] The SHARED-BAND posture of the Qwen row: the learned arm served under the Llama band
# (A100 160k/200k) instead of its own rule band (A100 320k/390k declared, 570k/710k from the
# attested s = 0.138) -- the middle panel of the Qwen triptych (own band / shared band /
# always-route), mirroring Figure 3's shared-band map. A DECLARED deviation from the D-738 rule:
# `decode_kv_band_policy: "shared"` is the only reason this arm boots, and the attestation
# records it. Never a headline arm.
ARMS["vskipper_qwen3_4b_sharedband"] = {
    **ARMS["vskipper_qwen3_4b"],
    "decode_kv_band": dict(SERVED_DECODE_KV_BAND_BY_DEVICE),
    "decode_kv_band_policy": "shared",
}

# [D-849, 2026-09-21] GATE-ONLY arm for the per-request body-pinning correctness checks: the
# served arm with a tiny decode K/V band (exit 3k / enter 6k resident tokens), so a 32-request
# smoke can be driven through both pins deliberately (one request alone -> stock; a burst ->
# the mirror engages -> later admissions fd; a second burst while the first still decodes ->
# mixed steps). Its own declared deviation, `decode_kv_band_policy: "gate"` (the boot assertion
# accepts it like "shared" and logs it; the execution-difference gate never sees it); NEVER a paper arm.
# [D-849 add. 23] The MONOTONE lane: v1.7 unchanged (per-pass prefill criterion with the engagement
# escape, per-step roofline band, one prefix-cache namespace, no admission pins) plus one rule --
# a request's first routed decode step promotes it for the rest of its generation; never demoted.
# Hypothesis: the Table 10 runaways came from stock<->routed OSCILLATION, so this alone should put
# the natural runaway count at the always-route level while every headline cell stays v1.7.
ARMS["vskipper_monotone"] = {
    **ARMS["vskipper"],
    "admission_overrides": {"criterion": "monotone_decode"},
}

# [D-849 add. 24] NO-UPGRADE lane: the four fixed plans with the version-1 prefill criterion at admission (demotion, uncached
# count) but NO mid-generation promotion (decode_upgrade none). Evidence 02:3xZ: every dense->routed switch mid-generation is a
# loop source (monotone lane 120 vs 77; 639ec with 207 promotions 92 vs 82; dd41 with few 69/72 vs 77/84), so the decode body
# is decided once at the prefill->decode boundary and never changed.
ARMS["vskipper_noupgrade"] = {
    **ARMS["vskipper"],
    "admission_overrides": {"decode_upgrade": "none"},
}

# [D-849 add. 28] The A/B REFERENCE for the loop finding: the served design as it stood before add. 28
# (dense-prefilled requests follow the band at the boundary and are promoted at band HIGH; uncached-token
# prefill criterion) -- the plan mix that served dense->routed for 56-86 % of coqa/bbh requests. Never a paper arm.
ARMS["vskipper_denseprefix_routed"] = {
    **ARMS["vskipper"],
    "admission_overrides": {
        "decode_after_stock_prefill": "band",
        "decode_upgrade": "band_high",
        "prefill_tokens": "uncached",
    },
}

# [D-849 add. 29] Two v3 variants for the bbh/coqa knee (debug A100 05:5xZ: v3 bbh 1.25x = E2E +17.5 %, TTFT +111 %, because the
# engagement demotion prefilled 534/690 rounds dense -> stock->stock for 3,759 requests -> 91 % dense decode steps, and the 252
# routed requests made nearly every step a forced-RUN mixed step). (a) routed prefill for every >= 1,536-token round (the demotion
# only observes): the request that will decode routed is prefilled routed -- the legal way to a decode gain on bbh/coqa;
# (b) mixed steps partitioned into a dense replay + a routed replay instead of one routed replay with forced rows.
ARMS["vskipper_routedprefill"] = {
    **ARMS["vskipper"],
    "admission_overrides": {"prefill_demotion": "observe_only"},
}
ARMS["vskipper_partition"] = {
    **ARMS["vskipper"],
    "admission_overrides": {"mixed_step": "partition"},
}
ARMS["vskipper_routedprefill_partition"] = {
    **ARMS["vskipper"],
    "admission_overrides": {"prefill_demotion": "observe_only", "mixed_step": "partition"},
}

# [D-849 add. 35, owner 2026-09-22 12:1xZ "we can try this if it helps"] ONE SWITCH, EITHER DIRECTION: the served
# design plus a one-time fd->stock decode downgrade at band LOW for rows that never switched (a request changes its
# decode body at most once: promoted at HIGH or demoted at LOW, whichever the band asks first). Targets the CoQA
# low-load TPOT tail (rows born routed that outlive the HIGH band keep paying routed overhead below the crossover).
ARMS["vskipper_oneswitch"] = {
    **ARMS["vskipper"],
    "admission_overrides": {"decode_downgrade": "band_low"},
}

# [D-849 add. 36] DOWN-ONLY: no promotion at all (the one-promotion CoQA natural cell doubled the context-limit hits, 76 vs
# always-route 38, while the pure dense-prompt/routed-decode leg gave 11 vs 40: on CoQA the loop source is the mid-generation
# dense->routed promotion), plus the safe one-time fd->stock downgrade at band LOW. A request's decode body can only move
# toward the dense body, once. Candidate for the served design if the A100 cells hold.
ARMS["vskipper_downonly"] = {
    **ARMS["vskipper"],
    "admission_overrides": {"decode_upgrade": "none", "decode_downgrade": "band_low"},
}

# [D-849 add. 40] PROMOTE-THEN-DEMOTE: the served design (promotion at HIGH) plus one demotion at LOW that may follow the
# promotion; a demoted row is never re-promoted. Candidate for the CoQA low-load TPOT cost; measured on coqa 0.75x/knee first.
ARMS["vskipper_promote_demote"] = {
    **ARMS["vskipper"],
    "admission_overrides": {"decode_downgrade": "band_low_any"},
}

ARMS["vskipper_pingate_lowband"] = {
    **ARMS["vskipper"],
    "decode_kv_band": {
        "NVIDIA_A100": (3000, 6000),
        "NVIDIA_A100_40GB": (3000, 6000),
        "NVIDIA_H100_HBM3": (3000, 6000),
    },
    "decode_kv_band_policy": "gate",
}

# [rebuttal, owner 2026-10-27] P/D-tuned decode bands ("tuning based on the new pd split setup to make the decode rate also
# triggered"; method A): the paper's A100 band keeps its position relative to the load, moved to the P/D decode instance -- band =
# (160k, 200k) x (the decode instance's upstream resident-K/V estimate at 0.95 x Q* / the integrated instance's 412,165), on a 1k grid,
# exit = 0.8 x enter (pd_tuned_band.py, measured on the paper's main A100 host). A DECLARED deviation from the roofline rule
# (decode_kv_band_policy "pd_tuned", model_runner logs it, the execution-difference declaration names the exact band); served by
# every instance of the treatment (G1 holds one served design per group; a prefill instance never reads the band). The paper-setting
# P/D run keeps the rule's band (dense decode).
PD_TUNED_BANDS: Final[dict[str, tuple[int, int]]] = {"1p1d": (12000, 15000), "2p2d": (10000, 13000)}
for _layout, _band in PD_TUNED_BANDS.items():
    ARMS[f"vskipper_pdtuned_{_layout}"] = {
        **ARMS["vskipper"],
        "decode_kv_band": {"NVIDIA_A100": _band},
        "decode_kv_band_policy": "pd_tuned",
    }

# [D-830, 2026-09-16] The third model: FlexiDepth-Qwen3-8B, our alignment-only `ste_hard` checkpoint
# (coef 2.5e-5; 4-GPU training campaign 2026-09-16, model of record = the last checkpoint that passed
# the on-node quality gate, D-828). Same family port as Qwen3-4B (36 layers, routed 18..35, 8 K/V heads
# x 128 = 4 KB/token/layer). `design_skip_ratio` = the chat-template skip the gate attested on the
# served checkpoint (0.383 at step 5,000); `decode_kv_band` = the rule with this arm's inputs
# (L_r = 18, s = 0.383, tau scaled 18/16): A100 V* = 206k -> 210k/260k; H100 HBM3 -> 360k/450k;
# A100-40 -> 170k/210k. Asserted at boot; served GSM8K only (owner 18:5xZ), on the A100 node.
ARMS["vskipper_qwen3_8b"] = {
    "skipper": "flexidepth", "phases": "both", "regime_switch": True,
    "gate_mode": "hard_mask", "compact": False, "routed_layers": tuple(range(18, 36)),
    "weights_key": "flexidepth_weights_qwen3_8b", "design_skip_ratio": 0.383,
    "decode_kv_band": {"NVIDIA_A100": (210000, 260000), "NVIDIA_H100_HBM3": (360000, 450000), "NVIDIA_A100_40GB": (170000, 210000)},
}
ARMS["vskipper_qwen3_8b_alwaysroute"] = {
    "skipper": "flexidepth", "phases": "both", "regime_switch": False,
    "gate_mode": "hard_mask", "compact": False, "routed_layers": tuple(range(18, 36)),
    "weights_key": "flexidepth_weights_qwen3_8b", "design_skip_ratio": 0.383,
}

# [D-832 add., owner 2026-09-16 22:0xZ] The same model's checkpoint-7500 (chat skip 0.428; its 100-doc gate read gsm8k
# -11 pp, inside the gate's noise): served under the rule "highest skip whose quality holds", decided by block B's full
# lm-eval score at the knee. Band by the rule with s = 0.428: A100 180k/230k.
ARMS["vskipper_qwen3_8b_s7500"] = {
    **ARMS["vskipper_qwen3_8b"], "weights_key": "flexidepth_weights_qwen3_8b_s7500", "design_skip_ratio": 0.428,
    "decode_kv_band": {"NVIDIA_A100": (180000, 230000), "NVIDIA_H100_HBM3": (320000, 400000), "NVIDIA_A100_40GB": (150000, 190000)},
}
ARMS["vskipper_qwen3_8b_s7500_alwaysroute"] = {
    **ARMS["vskipper_qwen3_8b_alwaysroute"], "weights_key": "flexidepth_weights_qwen3_8b_s7500", "design_skip_ratio": 0.428,
}

# [D-840/D-843/D-847, 2026-09-17..19] Penalty-tuning arms warm-started from checkpoint-7500 (owner: skip > 0.5 with quality held):
# PENALTY 2e-4 (coef 5e-5) and PENALTY 4e-4 (coef 1e-4), both run to step 20,000 on CloudLab with a check every 1,250 steps.
# `design_skip_ratio` = the chat-template skip the on-node routing probe attested; bands by the rule (L_r = 18, tau scaled 18/16).
# RESULT (D-847, 2026-09-19): the paper serves `vskipper_qwen3_8b_c1e4s15000` (4e-4, step 15,000; served decode skip 0.42 at the
# GSM8K knee, E2E -7.3 +- 0.9 %). Appendix K also reports the 2e-4 arm at steps 10,000 and 18,750. Nothing else is served.
for _tag, _s, _bands in (
    ("c5e5s10000", 0.445, {"NVIDIA_A100": (180000, 220000), "NVIDIA_H100_HBM3": (310000, 380000), "NVIDIA_A100_40GB": (140000, 180000)}),
    ("c5e5s18750", 0.485, {"NVIDIA_A100": (160000, 200000), "NVIDIA_H100_HBM3": (280000, 350000), "NVIDIA_A100_40GB": (130000, 160000)}),
    ("c1e4s15000", 0.520, {"NVIDIA_A100": (150000, 190000), "NVIDIA_H100_HBM3": (260000, 330000), "NVIDIA_A100_40GB": (120000, 150000)}),
):
    ARMS[f"vskipper_qwen3_8b_{_tag}"] = {
        **ARMS["vskipper_qwen3_8b"], "weights_key": f"flexidepth_weights_qwen3_8b_{_tag}", "design_skip_ratio": _s,
        "decode_kv_band": dict(_bands),
    }
    ARMS[f"vskipper_qwen3_8b_{_tag}_alwaysroute"] = {
        **ARMS["vskipper_qwen3_8b_alwaysroute"], "weights_key": f"flexidepth_weights_qwen3_8b_{_tag}", "design_skip_ratio": _s,
    }
del _tag, _s, _bands

# THE SKIP-RATE x DEPTH SWEEP (plan item 3; owner scope 2026-09-10: gsm8k only, ALL 12 points,
# one rate, 3 reps, with the upstream anchor interleaved in the same session).
#
# Twelve named arms, because that is the only parameterisation the repo supports. There is no
# sweep loop, no CLI, and no value-taking arm field, and D-609 forbids expressing any of this
# as an environment variable -- the design lives in the tree. `resolve_arm` is a dict lookup
# and no consumer hard-codes the arm list, so naming them is sufficient.
#
# Built by comprehension rather than twelve hand-written dicts: the entries differ only in two
# floats, and twelve near-identical literals are exactly where a transposed digit silently
# mislabels a sweep point. It is still a module-level constant evaluated at import, which is
# what D-609 asks for. Each point also publishes its own rate/depth/seed through the
# attestation (`skipper.py:284-286`), so a mislabelled cell is detectable in `server_info`
# rather than taken on trust.
#
# The SEED is identical across all twelve: the curve must vary in rate and depth only.
#
# DEPTH is a ratio WITHIN the fixed routed set `SERVED_ROUTED_LAYERS` (range(16, 32)), taken
# from the tail -- 0.25/0.50/0.75 select layers 28-31 / 24-31 / 20-31. Do NOT express depth by
# changing SERVED_ROUTED_LAYERS: that is a design constant shared by every arm, and
# `verify_served_design` will correctly refuse the diff.
#
# `integrated_randomskip` above is LEFT ALONE. Its values are the deleted
# arm_env_vdec_randomskip.sh's, verbatim, which keeps it comparable to prior RandomSkip
# evidence. Note that `..._r50_d50` is configuration-identical to it, same seed included --
# so the two are a free internal consistency check, not a duplicate.
_SWEEP_SKIP_RATES = (0.10, 0.25, 0.50, 0.75)
_SWEEP_DEPTH_RATIOS = (0.25, 0.50, 0.75)


def _rule_band(arm: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """The arm's decode K/V band FROM THE RULE (Appendix "band from the roofline"), for the A100.

    [D-764, owner 2026-09-13] Each sweep arm reads its own removable work: a mock that skips a
    fraction `rate` of rows around a fraction `depth` of the routed set removes rate x depth of
    the routed (row, layer) work, and that is the `s` the rule takes (FlexiDepth's `s` = 0.5 is
    the same quantity: every routed layer's router skips half the rows). The v1.4 sweep served
    every mock under the FlexiDepth band (160k/200k, s = 0.5), so a 25 % mock was engaged
    where the rule predicts a loss and lost (+26 % E2E at 25 % x 25 %). Computed in the tree at
    import from the same function the boot assertion uses -- not tuned, and printed in the
    attestation of every cell; a device without a roofline entry refuses to boot.
    """
    from vskipper.runtime import roofline  # lazy: roofline reads this module's constants

    inputs = roofline.arm_kv_rule_inputs(arm)
    return {key: roofline.derived_kv_band(key, **inputs) for key in SERVED_DECODE_KV_BAND_BY_DEVICE}


def _design_skip_ratio_default() -> float:
    from vskipper.runtime import roofline  # lazy: roofline reads this module's constants

    return float(roofline.DESIGN_DECODE_SKIP_RATIO)


def _mock_arm(rate: float, depth: float, **family: Any) -> dict[str, Any]:
    arm: dict[str, Any] = {
        "skipper": "deterministic_mock",
        "phases": "both",
        "regime_switch": True,
        "mock_token_skip_rate": rate,
        "mock_skipped_depth_ratio": depth,
        "mock_seed": 1234,
        "design_skip_ratio": rate * depth,
        **family,
    }
    arm["decode_kv_band"] = _rule_band(arm)
    return arm


# [owner 2026-11-06, reviewer DAyT] D³ (Fan et al., Findings of ACL 2025), a training-free skipper developed independently of
# FlexiDepth, at its published BBH setting (alpha 0.99, start 0.6), no tuning. Decode only: the prompt runs full depth. s is the share
# of the 16 routed decisions it drops: 11.93 layers per generated token over upstream's recorded BBH lengths at the knee. Layers
# 16-18 stay RUN. The FlexiDepth checkpoint's base weights are the base model, its router and projector are unused here.
def _d3_arm(**family: Any) -> dict[str, Any]:
    arm: dict[str, Any] = {
        "skipper": "position_decay",
        "phases": "decode",
        "regime_switch": True,
        "d3_alpha": 0.99,
        "d3_start_fraction": 0.6,
        "d3_num_layers": 32,
        "design_skip_ratio": 11.93 / 16,
        **family,
    }
    arm["decode_kv_band"] = _rule_band(arm)
    return arm


ARMS["vskipper_d3_bbh"] = _d3_arm()
ARMS["vskipper_d3_bbh_alwaysroute"] = {**_d3_arm(), "regime_switch": False}
del ARMS["vskipper_d3_bbh_alwaysroute"]["decode_kv_band"]

# [owner 2026-11-07, reviewer DAyT] D³ in vSkipper as its paper runs it: Llama-2-7B, no CoT, BBH alpha 0.9, start 0.6, GSM8K alpha 0.9999, start 0.2. Their Table 6
# layer counts fix these values, with every row counted to its maximum length (19.17 layers at 200 tokens, 29.90 at 1,024). s is that share of the routed
# decisions: 12.835 of 13 on BBH, 2.096 of 4 on GSM8K. The routed layers are exactly those the rule can drop. Llama-2 has 32 K/V heads and no FlexiDepth checkpoint.
_D3_LLAMA2 = {"d3_empty_modules": True, "kv_bytes_per_token_layer": 2 * 32 * 128 * 2, "served_model_revision": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"}


def _d3_pair(name: str, **family: Any) -> None:
    ARMS[name] = _d3_arm(**family)
    ARMS[f"{name}_alwaysroute"] = {**ARMS[name], "regime_switch": False}
    del ARMS[f"{name}_alwaysroute"]["decode_kv_band"]


_d3_pair("vskipper_d3_l2_bbh", d3_alpha=0.9, d3_start_fraction=0.6, routed_layers=tuple(range(19, 32)), design_skip_ratio=12.835 / 13,
         kv_band_round_tokens=5000, **_D3_LLAMA2)
_d3_pair("vskipper_d3_l2_gsm8k", d3_alpha=0.9999, d3_start_fraction=0.2, routed_layers=tuple(range(6, 10)), design_skip_ratio=2.096 / 4, **_D3_LLAMA2)

# [owner 2026-11-07, reviewer DAyT] D³ retuned on Llama-3-8B on the same client: the most skip within the released FlexiDepth checkpoint's own quality loss. BBH alpha 0.9,
# start 0.2 drops 25.21 of 26 routed layers at 200 tokens, GSM8K alpha 0.9999, start 0.8 drops 2.096 of 4 at 1,024. The routed layers are again those the rule can drop.
_d3_pair("vskipper_d3_l3_bbh", d3_alpha=0.9, d3_start_fraction=0.2, routed_layers=tuple(range(6, 32)), design_skip_ratio=25.21 / 26, d3_empty_modules=True)
_d3_pair("vskipper_d3_l3_gsm8k", d3_alpha=0.9999, d3_start_fraction=0.8, routed_layers=tuple(range(25, 29)), design_skip_ratio=2.096 / 4, d3_empty_modules=True)

ARMS.update({
    f"integrated_randomskip_r{int(rate * 100)}_d{int(depth * 100)}": _mock_arm(rate, depth)
    for rate in _SWEEP_SKIP_RATES
    for depth in _SWEEP_DEPTH_RATIOS
})

# [D-776] The UNGATED twins of the nine mock arms: regime switch OFF on both legs, so the routed body
# serves every pass at every occupancy (no band, like vskipper_qwen3_4b_alwaysroute). One map beside the
# shared-band map (step 6) and the own-band map (step 10): what engagement control is worth, arm by arm.
def _mock_arm_ungated(rate: float, depth: float) -> dict[str, Any]:
    arm = _mock_arm(rate, depth)
    arm["regime_switch"] = False
    del arm["decode_kv_band"]
    return arm


# [plan v4 item 7, 2026-09-16] The SHARED-BAND twins of the nine mock arms (Fig. 3b, the v1.4/v1.5 "fixed band" sweep):
# every mock served under the FlexiDepth band (A100 160k/200k) instead of its own rule band -- a DECLARED deviation
# (`decode_kv_band_policy: "shared"`, D-778 posture), so the map shows what serving a mock under the wrong band costs.
def _mock_arm_sharedband(rate: float, depth: float) -> dict[str, Any]:
    arm = _mock_arm(rate, depth)
    arm["decode_kv_band"] = dict(SERVED_DECODE_KV_BAND_BY_DEVICE)
    arm["decode_kv_band_policy"] = "shared"
    return arm


ARMS.update({
    f"integrated_randomskip_r{int(rate * 100)}_d{int(depth * 100)}_sharedband": _mock_arm_sharedband(rate, depth)
    for rate in _SWEEP_SKIP_RATES
    for depth in _SWEEP_DEPTH_RATIOS
})

ARMS.update({
    f"integrated_randomskip_r{int(rate * 100)}_d{int(depth * 100)}_alwaysroute": _mock_arm_ungated(rate, depth)
    for rate in _SWEEP_SKIP_RATES
    for depth in _SWEEP_DEPTH_RATIOS
})

# [D-764] The Qwen3-4B applicability-boundary points: the SAME mock on the Qwen family (its
# routed range, hard-mask bodies, its projector weights, its own rule-derived band). The
# learned Qwen skipper removes ~0.25 x 1.0 of the routed work; these two remove 0.375 and
# 0.5625 -- past the crossover -- so the row can separate "the family does not work" from "the
# trained skipper removes too little work".
ARMS.update({
    f"integrated_randomskip_qwen3_4b_r{int(rate * 100)}_d{int(depth * 100)}": _mock_arm(
        rate, depth,
        gate_mode="hard_mask", compact=False, routed_layers=tuple(range(18, 36)),
        weights_key="flexidepth_weights_qwen3_4b",
    )
    for rate, depth in ((0.50, 0.75), (0.75, 0.75))
})


# [rebuttal TP feasibility, owner 2026-10-24] FlexiDepth Llama at tensor-parallel size 2 and 4:
# the served design and always-route with `tp` declared. The served arm's thresholds follow the
# rule with the per-GPU K/V bytes (b / tp), asserted at boot like every other arm.
for _tp in (2, 4):
    for _base in ("vskipper", "integrated_alwaysskip"):
        _tp_arm = {**ARMS[_base], "tp": _tp}
        if ARMS[_base]["regime_switch"]:
            _tp_arm["decode_kv_band"] = _rule_band(_tp_arm)
        ARMS[f"{_base}_tp{_tp}"] = _tp_arm


def resolve_arm(name: str) -> dict[str, Any]:
    """Return the frozen definition for ``name``, or fail closed."""

    try:
        return dict(ARMS[name])
    except KeyError:
        raise ValueError(
            f"unknown arm {name!r}; canonical arms are {sorted(ARMS)} "
            "(defined in vpipe/design.py, D-609 -- arms are not configured by environment)"
        ) from None


# ---------------------------------------------------------------------------
# HOST PATHS
# ---------------------------------------------------------------------------
# Filesystem locations differ per machine and are NOT design. They come from a committed
# per-host file so they are versioned and auditable, never from shell exports. Absent or
# unreadable = fail closed at boot, which is the opposite of an env var's silent default.
_HOST_CONFIG_ENV = "SGLANG_VP_HOST_CONFIG"  # a PATH to the file, never a design value
_REQUIRED_HOST_KEYS = ("flexidepth_weights",)


def host_config(path: str | None = None) -> dict[str, str]:
    """Load the per-host deployment paths, failing closed."""

    raw = path or os.environ.get(_HOST_CONFIG_ENV, "")
    if not raw:
        raise ValueError(
            f"{_HOST_CONFIG_ENV} must point at a committed host-config file "
            "(deploy/hosts/<host>.json). Host paths are not supplied as environment "
            "values (D-609)."
        )
    p = Path(raw)
    if not p.is_file():
        raise ValueError(f"{_HOST_CONFIG_ENV} points at {raw!r}, which is not a file")
    cfg = json.loads(p.read_text())
    missing = [k for k in _REQUIRED_HOST_KEYS if not str(cfg.get(k, "")).strip()]
    if missing:
        raise ValueError(f"host config {raw} is missing required keys: {missing}")
    for key in list(cfg):
        if key not in ("conditional_graph_helper", "moe_config_dir") and not key.startswith("flexidepth_weights"):
            continue
        value = str(cfg.get(key, "") or "").strip()
        if value and not Path(value).exists():
            raise ValueError(f"host config {raw}: {key} = {value!r} does not exist")
    return {k: str(v) for k, v in cfg.items()}


def design_attestation() -> dict[str, Any]:
    """The design, for the manifest, so read-back can verify what was served.

    This is what the launch gate compares against: the boot writes this into
    ``observed_runtime`` and the campaign refuses to start if it does not match.
    """

    # [v1.5] Three fields may be overridden PER ARM (design.py ARMS, never the environment):
    # gate_mode (a property of the checkpoint), compact (a mechanism switch) and routed_layers
    # (the family's trained range). The attestation reports the ARM's effective value, so the
    # served-design gate compares what this arm declares against what this process serves.
    return {
        "source": "vpipe/design.py (constants, D-609)",
        "regime_switch": SERVED_REGIME_SWITCH,
        "low_row_policy": arm_low_row_policy(),
        "routed_layers": list(arm_routed_layers()),
        # [plan v3] the rule input the execution-difference gate needs to recompute this arm's
        # band for the device it detects (FlexiDepth arms: the design ratio; mocks: rate x depth)
        "design_skip_ratio": float(active_arm().get("design_skip_ratio", _design_skip_ratio_default())),
        "layer_policy": SERVED_LAYER_POLICY,
        "execution_mode": arm_execution_mode(),
        "compact_enabled": arm_compact_enabled(),
        "compact_phases": sorted(SERVED_COMPACT_PHASES),
        # [v1.5] The decode K/V band this arm DECLARES per device (the served value for the
        # device it runs on is in `served_design.regime_switch`, asserted against the roofline
        # rule at boot). `regime_switch` above carries the Llama-3-8B/A100 constants for every
        # arm; this field is what a Qwen or H100 boot actually resolves from.
        "decode_kv_band_by_device": {
            key: list(band)
            for key, band in sorted(
                (active_arm().get("decode_kv_band") or SERVED_DECODE_KV_BAND_BY_DEVICE).items()
            )
        },
        # [D-778] "rule" (the band above is the roofline rule's for this arm, asserted at boot) or
        # "shared" (a declared deviation: the global band served under another arm's inputs).
        "decode_kv_band_policy": active_arm().get("decode_kv_band_policy", "rule"),
        "gate_mode": arm_gate_mode(),
        "fused_router_norm": SERVED_FUSED_ROUTER_NORM,
    }


# ---------------------------------------------------------------------------
# RESOLVED HOST PATHS
# ---------------------------------------------------------------------------
# The served path asks for a path; it never asks the environment. The one variable
# that survives anywhere is SGLANG_VP_HOST_CONFIG, and it carries a POINTER to a
# committed file -- not a value. If it is absent, or the file is missing, or a path
# inside it does not exist, boot fails loudly. That is the opposite of an env var,
# whose absence produces a silent default.
_HOST_CACHE: dict[str, dict[str, str]] = {}


def _host() -> dict[str, str]:
    key = os.environ.get(_HOST_CONFIG_ENV, "")
    if key not in _HOST_CACHE:
        _HOST_CACHE[key] = host_config(key or None)
    return _HOST_CACHE[key]


def flexidepth_weights_path() -> str:
    """Absolute path to the ACTIVE ARM's router/projector weights.

    The host file names one file per checkpoint: `flexidepth_weights` (Llama-3-8B, required) and
    any `flexidepth_weights_<model>` an arm selects through its `weights_key` field (v1.5). An arm
    whose key is absent from the host file fails closed here.
    """

    key = str(active_arm().get("weights_key", "flexidepth_weights"))
    host = _host()
    if not str(host.get(key, "")).strip():
        raise ValueError(
            f"host config has no {key!r} (the active arm {active_arm_name()!r} selects it); "
            "add the checkpoint's weights file to deploy/hosts/<host>.json"
        )
    return host[key]


def conditional_graph_helper_path() -> str:
    return _host().get("conditional_graph_helper", "")


def moe_config_dir() -> str:
    return _host().get("moe_config_dir", "")


def served_model_revision() -> str:
    return active_arm().get("served_model_revision") or _host().get("served_model_revision", "")


# ---------------------------------------------------------------------------
# THE ACTIVE ARM
# ---------------------------------------------------------------------------
# Selected by a DURABLE FILE, not a shell export. The campaign script writes one arm
# name into `vskipper/configs/deploy/active_arm`; the served path reads it; the manifest records what
# resolved. A file survives the process, can be inspected after the fact, and diffs --
# an `export` leaves no trace and cannot be checked once the shell is gone.
#
# An unknown name raises. There is no "default that quietly applies", which is the
# property that let a wrong configuration run for eighteen hours.
_ACTIVE_ARM_FILE = "vskipper/configs/deploy/active_arm"
# [owner 2026-10-26, rebuttal FlashInfer arm, option 1: "FlashInfer, prefill routing only"] The served design on SGLang's
# FlashInfer substrate (launch profile paper_flashinfer). FlashInfer plans its decode work on the host once per batch, so it
# cannot skip a device-chosen subset of rows per layer inside a captured graph: the row-gated decode attention that prices the
# decode switch (roofline: a skipped row saves that layer's attention read of its context) exists only in the Triton kernel.
# This arm therefore keeps the served PREFILL routing and the per-request admission pinning unchanged and decodes every request
# on the dense body: an FD prefill decodes stock (``decode_after_fd_prefill: "stock"``, the fd->stock legal plan), a stock
# prefill decodes stock, nothing is promoted. Decode is not a phase this arm routes (``phases: "prefill"``), so the decode
# step runs SGLang's own full decode graph over the dense body -- the computation upstream's decode graph runs (the FD
# conditional decode backend exists only for an arm that routes decode, and its stock low band asserts it); the coverage
# ladder still captures to 4x max_bs = 1024 (skipper deployed). This admission block never reads the decode band
# (RegimeSwitchAdmissionConfig.band_free), so it pins without a decode leg. Masked decode attention (the Triton row mask) is
# off: no routed decode step ever runs. Never a headline arm.
ARMS["vskipper_fi"] = {
    **ARMS["vskipper"],
    "phases": "prefill",
    "masked_decode_attention": False,
    "admission_overrides": {
        "decode_after_fd_prefill": "stock",
        "decode_after_stock_prefill": "stock",
        "decode_upgrade": "none",
    },
}


_ARM_CACHE: dict[str, str] = {}


def _repo_root() -> Path:
    # vpipe/design.py -> vpipe -> srt -> sglang -> python -> <tree root>
    return Path(__file__).resolve().parents[4]


def active_arm_name() -> str:
    """Name of the arm this process is serving, from the durable file."""

    if "name" not in _ARM_CACHE:
        f = _repo_root() / _ACTIVE_ARM_FILE
        if not f.is_file():
            # [Codex F12] No silent default. Substituting a default arm here would quietly
            # serve the integrated skipper for a deployment that forgot to state its arm --
            # the "default that quietly applies" property that let a wrong configuration
            # run for eighteen hours. An unstated arm is an incomplete deployment.
            raise ValueError(
                f"{_ACTIVE_ARM_FILE} is missing from the deployed tree ({f}). The arm is "
                f"not defaulted: write one of {sorted(ARMS)} into that file (D-609/D-611)."
            )
        name = f.read_text().strip()
        resolve_arm(name)  # fail closed on an unknown name
        _ARM_CACHE["name"] = name
    return _ARM_CACHE["name"]


def active_arm() -> dict[str, Any]:
    return resolve_arm(active_arm_name())


def skipper_deployed() -> bool:
    """True when the active arm serves a skipper at all (stock serves none)."""

    return active_arm().get("skipper") is not None


def arm_phases() -> frozenset[str]:
    """The request phases the active arm routes. Empty when it serves no skipper."""

    phases = active_arm().get("phases")
    if not phases:
        return frozenset()
    if phases == "both":
        return frozenset(("decode", "prefill"))
    return frozenset((str(phases),))


def arm_gate_mode() -> str:
    """The routed-MLP gate arithmetic the ACTIVE ARM's checkpoint was trained with.

    A property of the checkpoint, carried as an arm field (``gate_mode``); the served
    design's ``SERVED_GATE_MODE`` (released) applies to every arm that does not declare one.
    """

    return str(active_arm().get("gate_mode", SERVED_GATE_MODE))


def arm_compact_enabled() -> bool:
    """Route-aware compaction for the active arm (arm field ``compact``; default = the design)."""

    return bool(active_arm().get("compact", SERVED_COMPACT_ENABLED))


def arm_masked_decode_attention() -> bool:
    """Triton row-masked decode attention for the active arm (arm field ``masked_decode_attention``; default = the design)."""

    return bool(active_arm().get("masked_decode_attention", SERVED_MASKED_DECODE_ATTENTION))


def arm_routed_layers() -> tuple[int, ...]:
    """The routed-layer range the active arm's checkpoint trained (default = the design's)."""

    return tuple(int(i) for i in active_arm().get("routed_layers", SERVED_ROUTED_LAYERS))


def arm_coverage_dense() -> bool:
    """Whether the coverage stamp guards this arm's decode passes (arm field ``coverage_dense``, default True).

    Only the cohorts-without-graphs control turns it off, so that its uncaptured decode passes run the routed
    cohort body instead of the dense fall-through. Every other arm keeps the stamp."""

    return bool(active_arm().get("coverage_dense", True))


def arm_execution_mode() -> str:
    """How the active arm executes a routed layer (arm field ``execution_mode``; default = the design's)."""

    return str(active_arm().get("execution_mode", SERVED_EXECUTION_MODE))


def arm_low_row_policy() -> str:
    """The routed-MLP body of the active arm (arm field ``low_row_policy``; default = the design's)."""

    return str(active_arm().get("low_row_policy", SERVED_LOW_ROW_POLICY))


def mechanism(value: bool, *, decode_only: bool = False) -> bool:
    """Apply a design constant only where the arm can actually run it.

    [D-609] The constants above describe THE SERVED SKIPPER SYSTEM. They are not
    universal truths about the process: an arm serving no skipper (``stock``) must have
    every one of them off, and a decode-phase mechanism needs an arm with a decode phase.

    The old arm_env_*.sh scripts encoded this implicitly by simply not exporting a
    variable in the arms where it did not apply. Deleting the scripts made that
    implicit knowledge disappear, and the smoke gate found it immediately: ``stock``
    demanded full_graph weights it does not have, and ``vpre_binarycohort`` asserted a
    conditional decode graph with no decode phase. Both were unservable. This function
    is where that knowledge now lives, once, checkably -- instead of in six scripts.
    """

    if not skipper_deployed():
        return False
    if arm_execution_mode() != SERVED_EXECUTION_MODE:
        return False   # the eager control arm runs none of the full-graph mechanisms
    if decode_only and "decode" not in arm_phases():
        return False
    return value
