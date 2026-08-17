"""Reader and writer for Wannier90 ``.amn`` files.

The ``.amn`` file produced by ``pw2wannier90`` (and consumed by
``wannier90``) stores the projection matrix

    A_{ib, iw, ik} = <psi_{ib,k} | g_{iw}>

as a flat list of records. The canonical format is::

    <header line>
    <num_bands>  <num_kpoints>  <num_wann>
    ib iw ik  Re(A_{ib,iw,ik})  Im(A_{ib,iw,ik})
    ...

with the data loop ordered outer-to-inner as ``ik``, ``iw``, ``ib``, all
indices 1-based on disk, 0-based in memory.
"""

from __future__ import annotations

import pathlib
from typing import Union

import numpy as np

PathLike = Union[str, pathlib.Path]


def read_amn(path: PathLike) -> tuple[str, np.ndarray]:
    """Read an ``.amn`` file.

    Parameters
    ----------
    path
        Path to the ``.amn`` file.

    Returns
    -------
    header
        The first line of the file (metadata comment written by
        ``pw2wannier90``), with the trailing newline stripped.
    A
        Complex projection matrix with shape
        ``(num_kpoints, num_wann, num_bands)``.

    """
    path = pathlib.Path(path)
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        dims = f.readline().split()
        num_bands = int(dims[0])
        num_kpoints = int(dims[1])
        num_wann = int(dims[2])
        A = np.zeros((num_kpoints, num_wann, num_bands), dtype=np.complex128)
        for _ in range(num_bands * num_wann * num_kpoints):
            parts = f.readline().split()
            ib = int(parts[0]) - 1
            iw = int(parts[1]) - 1
            ik = int(parts[2]) - 1
            A[ik, iw, ib] = complex(float(parts[3]), float(parts[4]))
    return header, A


def write_amn(path: PathLike, A: np.ndarray, header: str) -> None:
    """Write an ``.amn`` file in the canonical ``pw2wannier90`` format.

    Parameters
    ----------
    path
        Destination file path.
    A
        Complex projection matrix of shape
        ``(num_kpoints, num_wann, num_bands)``.
    header
        Header line (first line of the file, arbitrary comment).

    """
    path = pathlib.Path(path)
    if A.ndim != 3:
        raise ValueError(
            f"A must have shape (num_kpoints, num_wann, num_bands); got {A.shape}"
        )
    num_kpoints, num_wann, num_bands = A.shape
    with path.open("w", encoding="utf-8") as f:
        f.write(header.rstrip("\n") + "\n")
        f.write(f"{num_bands:12d}{num_kpoints:12d}{num_wann:12d}\n")
        for ik in range(num_kpoints):
            for iw in range(num_wann):
                for ib in range(num_bands):
                    val = A[ik, iw, ib]
                    f.write(
                        f"{ib + 1:5d}{iw + 1:5d}{ik + 1:5d}"
                        f"{val.real:18.12f}{val.imag:18.12f}\n"
                    )
