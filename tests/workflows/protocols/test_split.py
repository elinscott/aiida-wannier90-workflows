# pylint: disable=redefined-outer-name
"""Tests for the ``Wannier90SplitWorkChain.get_builder_from_protocol`` method."""
import pytest

from aiida.plugins import WorkflowFactory

from aiida_quantumespresso.common.types import ElectronicType, SpinType

from aiida_wannier90_workflows.common.types import (
    WannierDisentanglementType,
    WannierFrozenType,
    WannierProjectionType,
)
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain

Wannier90SplitWorkChain = WorkflowFactory("wannier90_workflows.split")


@pytest.fixture
def generate_split_builder_inputs(generate_builder_inputs, fixture_code):
    """Return inputs for ``Wannier90SplitWorkChain.get_builder_from_protocol``."""

    def _generate_split_builder_inputs(structure_id="Si"):
        from aiida.tools import get_explicit_kpoints_path

        inputs = generate_builder_inputs(structure_id)
        inputs["codes"]["split"] = fixture_code("wannier90_workflows.split")
        # Neither of these has a usable default here, so both have to be given.
        inputs["projection_type"] = WannierProjectionType.ATOMIC_PROJECTORS_QE
        inputs["bands_kpoints"] = get_explicit_kpoints_path(inputs["structure"])[
            "explicit_kpoints"
        ]
        return inputs

    return _generate_split_builder_inputs


@pytest.fixture
def capture_val_kwargs(monkeypatch):
    """Record the keyword arguments the valence ``Wannier90BaseWorkChain`` builder receives."""
    captured = {}
    original = Wannier90BaseWorkChain.get_builder_from_protocol.__func__

    def _spy(cls, code, **kwargs):
        captured.update(kwargs)
        return original(cls, code, **kwargs)

    monkeypatch.setattr(
        Wannier90BaseWorkChain,
        "get_builder_from_protocol",
        classmethod(_spy),
    )
    return captured


@pytest.mark.parametrize(
    "overrides",
    (
        {"electronic_type": ElectronicType.METAL},
        {"electronic_type": ElectronicType.INSULATOR},
        {
            "disentanglement_type": WannierDisentanglementType.NONE,
            "frozen_type": WannierFrozenType.NONE,
        },
        {
            "frozen_type": WannierFrozenType.NONE,
            "disentanglement_type": WannierDisentanglementType.NONE,
        },
    ),
)
def test_forced_keywords_accepted(generate_split_builder_inputs, overrides):
    """Test the keywords forced on the valence builder no longer collide with ``kwargs``."""
    builder = Wannier90SplitWorkChain.get_builder_from_protocol(
        **generate_split_builder_inputs(),
        **overrides,
    )
    assert "valcond" in builder
    assert "val" in builder


def test_forced_keywords_reach_valence_builder(
    generate_split_builder_inputs, capture_val_kwargs
):
    """Test the valence builder is built insulating, unpolarized and without disentanglement."""
    builder = Wannier90SplitWorkChain.get_builder_from_protocol(
        **generate_split_builder_inputs(),
        electronic_type=ElectronicType.METAL,
        disentanglement_type=WannierDisentanglementType.SMV,
        frozen_type=WannierFrozenType.PROJECTABILITY,
    )

    assert capture_val_kwargs["electronic_type"] == ElectronicType.INSULATOR
    assert capture_val_kwargs["spin_type"] == SpinType.NONE
    assert capture_val_kwargs["disentanglement_type"] == WannierDisentanglementType.NONE
    assert capture_val_kwargs["frozen_type"] == WannierFrozenType.NONE

    # `disentanglement_type=NONE` is the one forced value that leaves a trace in the
    # parameters the builder is populated with.
    for namespace in (builder.val, builder.cond):
        parameters = namespace["wannier90"]["parameters"].get_dict()
        assert parameters["dis_num_iter"] == 0
        assert "dis_proj_min" not in parameters
        assert "dis_froz_max" not in parameters


def test_spin_type(generate_split_builder_inputs):
    """Test ``spin_type`` is accepted when unpolarized and refused otherwise."""
    builder = Wannier90SplitWorkChain.get_builder_from_protocol(
        **generate_split_builder_inputs(),
        spin_type=SpinType.NONE,
    )
    assert "valcond" in builder

    for spin_type in (SpinType.COLLINEAR, SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT):
        with pytest.raises(ValueError, match="always unpolarized"):
            Wannier90SplitWorkChain.get_builder_from_protocol(
                **generate_split_builder_inputs(),
                spin_type=spin_type,
            )
