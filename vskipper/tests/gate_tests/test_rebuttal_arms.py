"""The rebuttal arms (owner 2026-11-06): the execution control, D³ and the variant launch profiles.

Each arm is one declared difference from the served design, and the served arms must not move."""
import pytest

torch = pytest.importorskip("torch")

from vskipper.runtime import design
from vskipper.runtime.skipper import PositionDecayFullGraphAdapter


def d3(alpha=0.99, start=0.6, layers=32):
    return PositionDecayFullGraphAdapter(alpha=alpha, start_fraction=start, num_layers=layers)


def state(adapter, generated, prompt_len=1000, valid=None):
    rows = len(generated)
    return adapter.prepare_batch(
        hidden_states=torch.zeros(rows, 4),
        request_ids=None,
        token_epochs=torch.tensor([prompt_len + i - 1 for i in generated], dtype=torch.int64),
        valid_rows=torch.ones(rows, dtype=torch.bool) if valid is None else valid,
        route_layer_order=tuple(range(16, 32)),
        prompt_lens=torch.full((rows,), prompt_len, dtype=torch.int64),
    )


def run_mask(adapter, batch, layer):
    return adapter.route(torch.zeros(1), router=None, layer_id=layer, batch_state=batch).explicit_run_mask.view(-1).tolist()


def test_d3_drops_the_published_interior_range_and_it_grows_upward():
    adapter = d3()
    assert adapter.start_layer == 19
    # i=1 keeps 31 of 32: one layer, layer 19. i=10 keeps 28: layers 19-22. i=50 keeps 19: the range reaches the last layer.
    batch = state(adapter, [1, 10, 50, 200])
    assert run_mask(adapter, batch, 18) == [True, True, True, True]
    assert run_mask(adapter, batch, 19) == [False, False, False, False]
    assert run_mask(adapter, batch, 20) == [True, False, False, False]
    assert run_mask(adapter, batch, 22) == [True, False, False, False]
    assert run_mask(adapter, batch, 23) == [True, True, False, False]
    assert run_mask(adapter, batch, 31) == [True, True, False, False]


def test_d3_kept_layers_are_nested_so_no_dropped_layer_is_read_again():
    adapter = d3()
    dropped = []
    for i in range(1, 120):
        batch = state(adapter, [i])
        dropped.append({layer for layer in range(16, 32) if not run_mask(adapter, batch, layer)[0]})
    assert all(a <= b for a, b in zip(dropped, dropped[1:]))


def test_d3_padding_rows_run_and_every_branch_weight_is_one():
    adapter = d3()
    batch = state(adapter, [50, 50], valid=torch.tensor([True, False]))
    assert run_mask(adapter, batch, 25) == [False, True]
    action = adapter.route(torch.zeros(1), router=None, layer_id=25, batch_state=batch)
    assert action.branch_weights.view(-1).tolist() == [1.0, 1.0]


def test_d3_refuses_a_start_outside_the_routed_layers():
    with pytest.raises(RuntimeError, match="outside the routed"):
        state(d3(start=0.2), [1])


def test_the_rule_keeps_d3_from_needing_a_tuned_threshold():
    arm = design.ARMS["vskipper_d3_bbh"]
    exit_kv, enter_kv = arm["decode_kv_band"]["NVIDIA_A100"]
    assert 100_000 <= exit_kv <= 110_000 and enter_kv > exit_kv
    assert "decode_kv_band" not in design.ARMS["vskipper_d3_bbh_alwaysroute"]


def test_the_control_arms_differ_from_the_decode_only_arm_by_one_field():
    base = design.ARMS["integrated_denseprefix_fd"]
    assert {k: v for k, v in design.ARMS["vskipper_ctrl_fullwidth"].items() if k != "low_row_policy"} == base
    assert {k: v for k, v in design.ARMS["vskipper_ctrl_eager"].items() if k != "execution_mode"} == base


def test_the_served_arm_declares_neither_control_field():
    for field in ("execution_mode", "low_row_policy"):
        assert field not in design.ARMS["vskipper"]


def test_variant_profiles_are_the_paper_profile_plus_one_flag():
    paper = design.SERVED_LAUNCH_PROFILES["paper"]
    assert design.SERVED_LAUNCH_PROFILES["paper_nodecodegraph"] == {**paper, "cuda_graph_backend_decode": "disabled"}
    assert design.SERVED_LAUNCH_PROFILES["paper_chunk2048"] == {**paper, "chunked_prefill_size": 2048}
    assert design.SERVED_LAUNCH_EXEMPTIONS["dtype"]["profiles"]["paper_chunk2048"] == "float16"


def test_the_scaling_arms_route_a_tail_of_the_decode_only_arm():
    assert design.ARMS["vskipper_ctrl_routed8"]["routed_layers"] == tuple(range(24, 32))
    assert design.ARMS["vskipper_ctrl_routed4"]["routed_layers"] == tuple(range(28, 32))
    assert design.ARMS["vskipper_ctrl_routed4"]["phases"] == "decode"


def test_the_prefill_control_differs_only_in_the_mlp_body():
    cohort, full = design.ARMS["vskipper_ctrl_prefill_cohort"], design.ARMS["vskipper_ctrl_prefill_fullwidth"]
    assert cohort == {"skipper": "flexidepth", "phases": "both", "regime_switch": False}
    assert {k: v for k, v in full.items() if k != "low_row_policy"} == cohort
    assert full["low_row_policy"] == "native_dense"


def test_served_fullwidth_differs_from_the_served_arm_only_in_the_body():
    assert design.ARMS["vskipper_fullwidth"] == {**design.ARMS["vskipper"], "low_row_policy": "native_dense", "fullwidth_engagement_counts": True}


def test_d3_llama2_arms_declare_the_papers_setting_and_their_own_kv():
    bbh, gsm = design.ARMS["vskipper_d3_l2_bbh"], design.ARMS["vskipper_d3_l2_gsm8k"]
    assert (bbh["d3_alpha"], bbh["d3_start_fraction"]) == (0.9, 0.6) and (gsm["d3_alpha"], gsm["d3_start_fraction"]) == (0.9999, 0.2)
    for arm, start in ((bbh, 19), (gsm, 6)):
        assert arm["routed_layers"][0] == start and arm["d3_empty_modules"] and arm["kv_bytes_per_token_layer"] == 16384
    assert "decode_kv_band" not in design.ARMS["vskipper_d3_l2_bbh_alwaysroute"]
    assert "d3_empty_modules" not in design.ARMS["vskipper"] and "served_model_revision" not in design.ARMS["vskipper"]


def test_d3_llama3_arms_declare_the_retuned_setting_on_llama3_kv():
    bbh, gsm = design.ARMS["vskipper_d3_l3_bbh"], design.ARMS["vskipper_d3_l3_gsm8k"]
    assert (bbh["d3_alpha"], bbh["d3_start_fraction"]) == (0.9, 0.2) and (gsm["d3_alpha"], gsm["d3_start_fraction"]) == (0.9999, 0.8)
    assert bbh["routed_layers"] == tuple(range(6, 32)) and gsm["routed_layers"] == tuple(range(25, 29))
    for arm in (bbh, gsm):
        assert arm["d3_empty_modules"] and "kv_bytes_per_token_layer" not in arm and "served_model_revision" not in arm
    assert bbh["decode_kv_band"]["NVIDIA_A100"] == (80000, 100000)     # the A100 entry's 1,935 GB/s, as every band on this device
    assert gsm["decode_kv_band"]["NVIDIA_A100"] == (150000, 190000)
    assert "decode_kv_band" not in design.ARMS["vskipper_d3_l3_bbh_alwaysroute"]


def test_d3_llama2_bands_leave_room_between_enter_and_exit():
    assert design.ARMS["vskipper_d3_l2_bbh"]["decode_kv_band"]["NVIDIA_A100"] == (20000, 25000)
    assert design.ARMS["vskipper_d3_l2_gsm8k"]["decode_kv_band"]["NVIDIA_A100"] == (40000, 50000)

