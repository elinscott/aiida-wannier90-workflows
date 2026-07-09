"""Module with common data types."""

import enum


class WannierProjectionType(enum.Enum):
    """Enumeration to indicate the Wannier initial projection type."""

    # analytic functions which are solutions of hydrogenic Schrödinger equation,
    # as implemented in Wannier90.
    ANALYTIC = "analytic"

    # Wannier90 random projection
    RANDOM = "random"

    # Selected columns of density matrix method, automatically generated from density matrix
    SCDM = "scdm"

    # Atomic pseudo wavefunctions contained in the QE pseudopotentials
    ATOMIC_PROJECTORS_QE = "atomic_projectors_qe"

    # Atomic orbitals from external source
    ATOMIC_PROJECTORS_EXTERNAL = "atomic_projectors_external"


class WannierDisentanglementType(enum.Enum):
    """Enumeration to indicate the Wannier disentanglement type."""

    # no disentanglement
    NONE = "none"

    # Souza-Marzari-Vanderbuilt disentanglement, as implemented in Wannier90
    SMV = "smv"


class WannierFrozenType(enum.Enum):
    """Enumeration to indicate the Wannier frozen type."""

    # no frozen states
    NONE = "none"

    # a fixed dis_froz_max, default is fermi_energy + 2 eV
    ENERGY_FIXED = "energy_fixed"

    # automatically set the energy dis_froz_max based on bands projectability,
    # TODO: update: default is having an energy that includes all states that have projectability >= threshold,  # pylint: disable=fixme
    # default projectability threshold is 0.9
    ENERGY_AUTO = "energy_auto"
    # TODO add doc, for each entry explain number of free parameters and detail explaination  # pylint: disable=fixme

    # disentaglement per kpoint based on projectability, default thresholds are min/max = 0.01/0.95
    PROJECTABILITY = "projectability"

    # fixed window + projectability per kpoint, default is fermi_energy + 2 eV and min/max = 0.01/0.95
    FIXED_PLUS_PROJECTABILITY = "fixed_plus_projectability"


class OptimizeStrategy(enum.Enum):
    """Enumeration to indicate the optimization strategy for dis_proj_min/max."""

    # Exhaustive grid search over all (dis_proj_min, dis_proj_max) combinations
    GRID = "grid"

    # Bayesian optimization using Gaussian process surrogate model
    BAYESIAN = "bayesian"


class OptimizeMetric(enum.Enum):
    """Enumeration to indicate the metric used for optimization."""

    # Fermi-Dirac-weighted bands distance with configurable mu and sigma
    FERMI_DIRAC = "fermi_dirac"

    # Legacy alias: Fermi-Dirac at Ef+2eV with sigma=0.1
    FERMI_DIRAC_EF2 = "fermi_dirac_ef2"

    # Unweighted RMS bands distance across all bands equally (sigma -> inf)
    UNWEIGHTED_RMS = "unweighted_rms"


class OptimizeMuReference(enum.Enum):
    """Enumeration to indicate the reference point for the Fermi-Dirac mu parameter."""

    # mu = fermi_energy + mu_shift
    FERMI_ENERGY = "fermi_energy"

    # mu = CBM + mu_shift, where CBM is extracted from the reference bands
    CBM = "cbm"

    # mu = VBM + mu_shift, where VBM is extracted from the reference bands
    VBM = "vbm"


class WannierFileFormat(enum.Enum):
    """Enumeration to indicate the format of amn/mmn/eig/spn/unk files."""

    # Fortran formatted text file
    FORTRAN_FORMATTED = "fortran_formatted"

    # Fortran unformatted binary file
    FORTRAN_UNFORMATTED = "fortran_unformatted"

    # HDF5 file
    HDF5 = "hdf5"
