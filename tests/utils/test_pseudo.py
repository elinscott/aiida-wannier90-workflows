"""Unit tests for the :py:mod:`~aiida_wannier90_workflows.utils.pseudo` module."""

import pytest


class FakePseudo:
    """Duck-typed stand-in for a ``PseudoPotentialData``."""

    def __init__(self, element, z_valence, md5="0" * 32):
        self.element = element
        self.z_valence = z_valence
        self.md5 = md5
        self.filename = f"{element}.upf"

    def get_content(self):
        """Return invalid UPF content, so the wave-function cross-check is skipped."""
        return "not a upf file"


@pytest.mark.parametrize(
    ("element", "z_valence", "expected"),
    (
        ("Si", 4, ["3S", "3P"]),
        ("O", 6, ["2S", "2P"]),
        # Semicore-in-valence pseudisation: shells collected outermost-inward
        # until the valence electrons are accounted for.
        ("Ti", 12, ["3S", "3P", "4S", "3D"]),
        ("Cu", 19, ["3S", "3P", "4S", "3D"]),
    ),
)
def test_infer_pseudo_orbitals(element, z_valence, expected):
    """Aufbau inference reproduces the curated-table labels."""
    from aiida_wannier90_workflows.utils.pseudo import _infer_pseudo_orbitals

    entry = _infer_pseudo_orbitals(FakePseudo(element, z_valence))
    assert entry is not None
    assert entry["pswfcs"] == expected
    assert entry["semicores"] == []


def test_infer_pseudo_orbitals_without_z_valence():
    """No ``z_valence`` means no inference."""
    from aiida_wannier90_workflows.utils.pseudo import _infer_pseudo_orbitals

    assert _infer_pseudo_orbitals(FakePseudo("Si", None)) is None


def test_get_pseudo_orbitals_overrides_win():
    """An explicit override bypasses tables and inference."""
    from aiida_wannier90_workflows.utils.pseudo import PseudoOrbitals, get_pseudo_orbitals

    override = PseudoOrbitals(pswfcs=["3S", "3P", "4S", "3D"], semicores=["3S", "3P"])
    result = get_pseudo_orbitals({"Ti": FakePseudo("Ti", 12)}, overrides={"Ti": override})
    assert result["Ti"]["semicores"] == ["3S", "3P"]


def test_get_pseudo_orbitals_inference_warns():
    """A pseudo missing from the tables resolves by inference, with a warning."""
    from aiida_wannier90_workflows.utils.pseudo import get_pseudo_orbitals

    with pytest.warns(UserWarning, match="inferred from z_valence"):
        result = get_pseudo_orbitals({"Si": FakePseudo("Si", 4)})
    assert result["Si"]["pswfcs"] == ["3S", "3P"]
    assert result["Si"]["semicores"] == []


def test_get_pseudo_orbitals_unresolvable_raises():
    """When inference is impossible the error explains the override escape hatch."""
    from aiida_wannier90_workflows.utils.pseudo import get_pseudo_orbitals

    with pytest.raises(ValueError, match="overrides"):
        get_pseudo_orbitals({"Xx": FakePseudo("Xx", None)})


@pytest.mark.parametrize(
    ("orbital", "expect_frozen"),
    (
        # No key at all: a pseudo-atomic orbital, frozen.
        ({"label": "S", "l": 0}, True),
        # Yuhao-protocol alpha bookkeeping: "UPF" marks the original
        # pseudo-atomic orbital, a number marks a generated one.
        ({"label": "S", "l": 0, "alpha": "UPF"}, True),
        ({"label": "S", "l": 0, "alpha": 1.5}, False),
        # An explicit ``frozen`` takes precedence over ``alpha``.
        ({"label": "S", "l": 0, "frozen": False}, False),
        ({"label": "S", "l": 0, "frozen": False, "alpha": "UPF"}, False),
        ({"label": "S", "l": 0, "frozen": True, "alpha": 1.5}, True),
    ),
)
def test_get_frozen_list_ext(generate_structure, orbital, expect_frozen):
    """Test the per-orbital frozen selection of ``get_frozen_list_ext``."""
    from aiida_wannier90_workflows.utils.pseudo import get_frozen_list_ext

    structure = generate_structure("Si")

    frozen_list = get_frozen_list_ext(
        structure=structure,
        external_projectors={"Si": [orbital]},
        spin_non_collinear=False,
    )

    # Bulk silicon has two sites; an s orbital contributes one projector each.
    assert frozen_list == ([1, 2] if expect_frozen else [])


def test_get_frozen_list_ext_mixed(generate_structure):
    """Test frozen indexing across a mixed orbital table."""
    from aiida_wannier90_workflows.utils.pseudo import get_frozen_list_ext

    structure = generate_structure("Si")

    frozen_list = get_frozen_list_ext(
        structure=structure,
        external_projectors={
            "Si": [
                {"label": "S", "l": 0, "alpha": "UPF"},
                {"label": "P", "l": 1, "frozen": False},
            ]
        },
        spin_non_collinear=False,
    )

    # Per site: s (1 projector, frozen) then p (3 projectors, not frozen).
    assert frozen_list == [1, 5]
