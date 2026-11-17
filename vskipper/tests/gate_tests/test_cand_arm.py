"""The repetition-fix candidate arm is the served design with another router/projector file, nothing else."""
from vskipper.runtime import design


def test_cand_grpo_arm_is_the_served_design_with_another_weights_file():
    cand, served = dict(design.ARMS["vskipper_cand_grpo"]), dict(design.ARMS["vskipper"])
    assert cand.pop("weights_key") == "flexidepth_weights_cand_grpo"
    assert "weights_key" not in served
    assert cand == served
