import io
import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal, Optional, Union, overload

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import (
    IPythonConsole,  # type: ignore
    MolDrawOptions,
    rdMolDraw2D,
)
from rdkit.Chem.rdDepictor import Compute2DCoords

from .config import DofDrawSettings, dofconfig

try:
    from IPython.display import SVG
except ImportError:
    svg_support = False
else:
    svg_support = True


@lru_cache(maxsize=4096)
def _get_atom_dof_color_cached(
    base_color: tuple[float, float, float],
    proximity: float,
    min_alpha: float,
    fog_color: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Calculate the RGBA color of an atom with depth-of-field effect."""
    base_rgb = np.array(base_color)
    fog_rgb = np.array(fog_color)
    dark_color_rgba = np.array([*base_rgb, 1.0])
    light_rgb = base_rgb * 0.2 + fog_rgb * 0.8
    light_color_rgba = np.array([*light_rgb, min_alpha])
    final_color = light_color_rgba + proximity * (dark_color_rgba - light_color_rgba)
    return tuple(final_color.tolist())


def _apply_rdkit_global_options(target_dopts: MolDrawOptions):
    """Reflect global options from IPythonConsole.drawOptions."""
    if IPythonConsole is None or not hasattr(IPythonConsole, "drawOptions"):
        return
    source_dopts: MolDrawOptions = IPythonConsole.drawOptions
    if source_dopts is None:
        return
    for attr in dir(source_dopts):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(source_dopts, attr)
            if callable(val):
                continue
            if hasattr(target_dopts, attr):
                setattr(target_dopts, attr, val)
        except Exception:
            pass


def _prepare_mol_data(
    mol: Union[Chem.Mol, Chem.RWMol],
    settings: DofDrawSettings,
    keep_key_atom_colors: bool = True,
) -> tuple[
    Chem.Mol,
    dict[int, tuple[float, float, float, float]],
    dict[int, tuple[float, float, float, float]],
]:
    """
    Internal Helper: Process a single molecule for DOF drawing.
    Returns: (Prepared Molecule, Atom Colors Dict, Bond Colors Dict)
    """
    if not mol:
        raise ValueError("Invalid molecule")

    mol_copy = Chem.Mol(mol)

    if mol_copy.GetNumConformers() == 0:
        Compute2DCoords(mol_copy)

    conf = mol_copy.GetConformer()
    pos = conf.GetPositions()
    z_coords = pos[:, 2]

    if z_coords.size > 1 and z_coords.max() != z_coords.min():
        norm_z = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())
        proximity = norm_z
    else:
        proximity = np.full(z_coords.shape, 1.0)

    highlight_atom_colors: dict[int, tuple[float, float, float, float]] = {}
    carbon_base_color = settings.get_atom_color(6)

    for i in range(mol_copy.GetNumAtoms()):
        atom = mol_copy.GetAtomWithIdx(i)
        atomic_num = atom.GetAtomicNum()
        base_color = settings.get_atom_color(atomic_num)
        target_color = base_color if keep_key_atom_colors else carbon_base_color

        highlight_atom_colors[i] = _get_atom_dof_color_cached(
            base_color=target_color,
            proximity=proximity[i],
            min_alpha=settings.min_alpha,
            fog_color=settings.fog_color,
        )

    highlight_bond_colors: dict[int, tuple[float, float, float, float]] = {}
    for i in range(mol_copy.GetNumBonds()):
        bond = mol_copy.GetBondWithIdx(i)
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        c1 = _get_atom_dof_color_cached(
            carbon_base_color,
            proximity[atom1_idx],
            settings.min_alpha,
            settings.fog_color,
        )
        c2 = _get_atom_dof_color_cached(
            carbon_base_color,
            proximity[atom2_idx],
            settings.min_alpha,
            settings.fog_color,
        )
        bond_color_arr = (np.array(c1) + np.array(c2)) / 2
        highlight_bond_colors[i] = tuple(bond_color_arr.tolist())

    return mol_copy, highlight_atom_colors, highlight_bond_colors


# =============================================================================
# Single Molecule Drawer
# =============================================================================


@overload
def MolToDofImage(
    mol: Union[Chem.Mol, Chem.RWMol],
    size: Optional[tuple[int, int]] = None,
    legend: str = "",
    use_svg: Literal[True] = True,
    return_image: Literal[True] = True,
    *,
    settings: Optional[DofDrawSettings] = None,
    **kwargs: Any,
) -> "SVG": ...
@overload
def MolToDofImage(
    mol: Union[Chem.Mol, Chem.RWMol],
    size: Optional[tuple[int, int]] = None,
    legend: str = "",
    use_svg: Literal[False] = False,
    return_image: Literal[True] = True,
    *,
    settings: Optional[DofDrawSettings] = None,
    **kwargs: Any,
) -> Image.Image: ...
@overload
def MolToDofImage(
    mol: Union[Chem.Mol, Chem.RWMol],
    size: Optional[tuple[int, int]] = None,
    legend: str = "",
    use_svg: Literal[True] = True,
    return_image: Literal[False] = False,
    *,
    settings: Optional[DofDrawSettings] = None,
    **kwargs: Any,
) -> str: ...
@overload
def MolToDofImage(
    mol: Union[Chem.Mol, Chem.RWMol],
    size: Optional[tuple[int, int]] = None,
    legend: str = "",
    use_svg: Literal[False] = False,
    return_image: Literal[False] = False,
    *,
    settings: Optional[DofDrawSettings] = None,
    **kwargs: Any,
) -> bytes: ...


def MolToDofImage(  # noqa: N802
    mol: Union[Chem.Mol, Chem.RWMol],
    size: Optional[tuple[int, int]] = None,
    legend: str = "",
    use_svg: bool = True,
    return_image: bool = True,
    *,
    settings: Optional[DofDrawSettings] = None,
    **kwargs: Any,
) -> Union["SVG", str, Image.Image, bytes]:
    """Draw a single molecule with DOF effect."""
    if settings is None:
        settings = dofconfig

    draw_size = size if size else settings.default_size

    ready_mol, atom_colors, bond_colors = _prepare_mol_data(mol, settings)

    if use_svg:
        drawer = rdMolDraw2D.MolDraw2DSVG(draw_size[0], draw_size[1])
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(draw_size[0], draw_size[1])

    dopts = drawer.drawOptions()
    _apply_rdkit_global_options(dopts)
    dopts.continuousHighlight = False
    dopts.circleAtoms = False
    for k, v in kwargs.items():
        if hasattr(dopts, k):
            setattr(dopts, k, v)

    drawer.DrawMolecule(
        ready_mol,
        legend=legend,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBonds=list(bond_colors.keys()),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()

    if use_svg:
        svg_text: str = drawer.GetDrawingText()
        if return_image:
            if not svg_support:
                raise ImportError("IPython required for SVG.")
            return SVG(svg_text)
        return svg_text
    else:
        png_data: bytes = drawer.GetDrawingText()  # type: ignore
        if return_image:
            return Image.open(io.BytesIO(png_data))
        return png_data


# =============================================================================
# Grid Drawer
# =============================================================================


@overload
def MolGridToDofImage(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol, None]],
    molsPerRow: int = 3,  # noqa: N803
    subImgSize: tuple[int, int] = (300, 300),  # noqa: N803
    legends: Optional[Sequence[Union[str, None]]] = None,
    use_svg: Literal[True] = True,
    return_image: Literal[True] = True,
    *,
    settings: Optional[DofDrawSettings] = None,
    **kwargs: Any,
) -> "SVG": ...
@overload
def MolGridToDofImage(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol, None]],
    molsPerRow: int = 3,  # noqa: N803
    subImgSize: tuple[int, int] = (300, 300),  # noqa: N803
    legends: Optional[Sequence[Union[str, None]]] = None,
    use_svg: Literal[False] = False,
    return_image: Literal[True] = True,
    *,
    settings: Optional[DofDrawSettings] = None,
    **kwargs: Any,
) -> Image.Image: ...
@overload
def MolGridToDofImage(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol, None]],
    molsPerRow: int = 3,  # noqa: N803
    subImgSize: tuple[int, int] = (300, 300),  # noqa: N803
    legends: Optional[Sequence[Union[str, None]]] = None,
    use_svg: Literal[True] = True,
    return_image: Literal[False] = False,
    *,
    settings: Optional[DofDrawSettings] = None,
    **kwargs: Any,
) -> str: ...
@overload
def MolGridToDofImage(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol, None]],
    molsPerRow: int = 3,  # noqa: N803
    subImgSize: tuple[int, int] = (300, 300),  # noqa: N803
    legends: Optional[Sequence[Union[str, None]]] = None,
    use_svg: Literal[False] = False,
    return_image: Literal[False] = False,
    *,
    settings: Optional[DofDrawSettings] = None,
    **kwargs: Any,
) -> bytes: ...
def MolGridToDofImage(  # noqa: N802
    mols: Sequence[Union[Chem.Mol, Chem.RWMol, None]],
    molsPerRow: int = 3,  # noqa: N803
    subImgSize: tuple[int, int] = (300, 300),  # noqa: N803
    legends: Optional[Sequence[Union[str, None]]] = None,
    use_svg: bool = True,
    return_image: bool = True,
    *,
    settings: Optional[DofDrawSettings] = None,
    **kwargs: Any,
) -> Union["SVG", str, Image.Image, bytes]:
    """
    Draw a grid of molecules with DOF effect.
    Compatible with RDKit's Chem.Draw.MolsToGridImage arguments.
    """
    if settings is None:
        settings = dofconfig

    # Handle empty input list to avoid RDKit errors
    if not mols:
        n_rows = 1
        full_width = subImgSize[0] * molsPerRow
        full_height = subImgSize[1] * n_rows
        if use_svg:
            drawer = rdMolDraw2D.MolDraw2DSVG(
                full_width, full_height, subImgSize[0], subImgSize[1]
            )
            drawer.FinishDrawing()
            svg_text = drawer.GetDrawingText()
            return SVG(svg_text) if return_image else svg_text
        else:
            # For non-SVG, return a blank PIL image or its byte representation
            blank_image = Image.new("RGB", (full_width, full_height), (255, 255, 255))
            if return_image:
                return blank_image
            else:
                byte_arr = io.BytesIO()
                blank_image.save(byte_arr, format="PNG")
                return byte_arr.getvalue()

    valid_mols = []
    valid_legends = []

    all_atom_colors = []
    all_bond_colors = []
    all_highlight_atoms = []
    all_highlight_bonds = []

    for i, m in enumerate(mols):
        if m is None:
            m = Chem.Mol()

        try:
            ready_mol, atom_cols, bond_cols = _prepare_mol_data(m, settings)
        except Exception:
            ready_mol = m
            atom_cols, bond_cols = {}, {}

        valid_mols.append(ready_mol)

        if legends and i < len(legends) and legends[i]:
            valid_legends.append(str(legends[i]))
        else:
            valid_legends.append("")

        all_atom_colors.append(atom_cols)
        all_bond_colors.append(bond_cols)
        all_highlight_atoms.append(list(atom_cols.keys()))
        all_highlight_bonds.append(list(bond_cols.keys()))

    n_mols = len(valid_mols)
    n_rows = math.ceil(n_mols / molsPerRow)
    full_width = subImgSize[0] * molsPerRow
    full_height = subImgSize[1] * n_rows

    if use_svg:
        drawer = rdMolDraw2D.MolDraw2DSVG(
            full_width, full_height, subImgSize[0], subImgSize[1]
        )
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(
            full_width, full_height, subImgSize[0], subImgSize[1]
        )

    dopts = drawer.drawOptions()
    _apply_rdkit_global_options(dopts)
    dopts.continuousHighlight = False
    dopts.circleAtoms = False

    for k, v in kwargs.items():
        if hasattr(dopts, k):
            setattr(dopts, k, v)

    drawer.DrawMolecules(
        valid_mols,
        legends=valid_legends,
        highlightAtoms=all_highlight_atoms,
        highlightAtomColors=all_atom_colors,
        highlightBonds=all_highlight_bonds,
        highlightBondColors=all_bond_colors,
    )
    drawer.FinishDrawing()
    if use_svg:
        svg_text: str = drawer.GetDrawingText()
        if return_image:
            if not svg_support:
                raise ImportError("IPython required for SVG.")
            return SVG(svg_text)
        return svg_text
    else:
        png_data: bytes = drawer.GetDrawingText()  # type: ignore
        if return_image:
            return Image.open(io.BytesIO(png_data))
        return png_data
