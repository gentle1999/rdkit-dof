import io
import math
import os
import warnings
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw as ChemDraw  # pyright: ignore[reportAttributeAccessIssue]
from rdkit.Chem.Draw import (  # pyright: ignore[reportMissingImports]
    MolDrawOptions,
    rdMolDraw2D,
)
from rdkit.Chem.Draw.rdMolDraw2D import (  # pyright: ignore[reportMissingImports]
    MolDraw2D,
)
from rdkit.Chem.rdDepictor import Compute2DCoords

from .config import DofDrawSettings, dofconfig

RGBColor = Tuple[float, float, float]
RGBAColor = Tuple[float, float, float, float]
Size = Tuple[int, int]
if TYPE_CHECKING:
    PathLikeStr = Union[str, os.PathLike[Any]]
else:
    PathLikeStr = Union[str, os.PathLike]
ColorMap = Dict[int, RGBAColor]
IPythonConsole: Any = None
SVG_NS = "http://www.w3.org/2000/svg"

ET.register_namespace("", SVG_NS)

_SVG: Any = None
try:
    from IPython.core.display import SVG  # pyright: ignore[reportMissingImports]
except ImportError:
    svg_support = False
else:
    IPythonConsole = getattr(ChemDraw, "IPythonConsole", None)
    _SVG = SVG
    svg_support = True


def _make_svg_image(svg_text: str) -> Any:
    if not svg_support or _SVG is None:
        raise ImportError("IPython required for SVG.")
    return cast(Any, _SVG)(svg_text)


def _contains_non_ascii_text(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def _warn_if_unicode_legend(legends: Sequence[str]) -> None:
    if not any(legend and _contains_non_ascii_text(legend) for legend in legends):
        return
    warnings.warn(
        "Unicode/non-ASCII legend text is not supported by rdkit-dof because "
        "RDKit drawing backends do not reliably render it. Use ASCII legend "
        "text for portable output.",
        UserWarning,
        stacklevel=3,
    )


@lru_cache(maxsize=4096)
def _get_atom_dof_color_cached(
    base_color: RGBColor,
    proximity: float,
    min_alpha: float,
    fog_color: RGBColor,
) -> RGBAColor:
    """Calculate the RGBA color of an atom with depth-of-field effect."""
    base_rgb = np.array(base_color)
    fog_rgb = np.array(fog_color)
    dark_color_rgba = np.array([*base_rgb, 1.0])
    light_rgb = base_rgb * 0.2 + fog_rgb * 0.8
    light_color_rgba = np.array([*light_rgb, min_alpha])
    final_color = light_color_rgba + proximity * (dark_color_rgba - light_color_rgba)
    return cast(RGBAColor, tuple(final_color.tolist()))


def _get_saturated_highlight_color(color: RGBAColor) -> RGBAColor:
    """Use highlight RGB directly and keep it fully opaque."""
    return (color[0], color[1], color[2], 1.0)


def _normalize_frame_durations(
    duration: Union[int, Sequence[int]],
    frame_count: int,
) -> Union[int, List[int]]:
    """Validate frame durations in milliseconds."""
    if isinstance(duration, int):
        if duration <= 0:
            raise ValueError("duration must be a positive integer")
        return duration

    durations = [int(frame_duration) for frame_duration in duration]
    if len(durations) != frame_count:
        raise ValueError("duration sequence must have the same length as mols")
    if any(frame_duration <= 0 for frame_duration in durations):
        raise ValueError("all duration values must be positive integers")
    return durations


def _duration_list(
    duration: Union[int, Sequence[int]],
    frame_count: int,
) -> List[int]:
    normalized = _normalize_frame_durations(duration, frame_count)
    if isinstance(normalized, int):
        return [normalized for _ in range(frame_count)]
    return normalized


def _format_seconds(milliseconds: int) -> str:
    return f"{milliseconds / 1000:g}s"


def _format_key_time(value: float) -> str:
    if value <= 0:
        return "0"
    if value >= 1:
        return "1"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _make_opacity_animation(
    frame_index: int,
    durations: Sequence[int],
    loop: int,
) -> ET.Element:
    total_duration = sum(durations)
    frame_start = sum(durations[:frame_index]) / total_duration
    frame_end = sum(durations[: frame_index + 1]) / total_duration

    if frame_index == 0:
        key_times = [0.0, frame_end, 1.0]
        values = ["1", "0", "0"]
    elif frame_index == len(durations) - 1:
        key_times = [0.0, frame_start, 1.0]
        values = ["0", "1", "1"]
    else:
        key_times = [0.0, frame_start, frame_end, 1.0]
        values = ["0", "1", "0", "0"]

    return ET.Element(
        f"{{{SVG_NS}}}animate",
        {
            "attributeName": "opacity",
            "values": ";".join(values),
            "keyTimes": ";".join(_format_key_time(value) for value in key_times),
            "dur": _format_seconds(total_duration),
            "repeatCount": "indefinite" if loop == 0 else str(loop),
            "calcMode": "discrete",
        },
    )


def _element_local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _merge_svg_defs(root: ET.Element, defs: ET.Element) -> None:
    root_defs = root.find(f"{{{SVG_NS}}}defs")
    if root_defs is None:
        root_defs = ET.Element(f"{{{SVG_NS}}}defs")
        root.insert(0, root_defs)

    existing = {
        ET.tostring(child, encoding="unicode", method="xml")
        for child in list(root_defs)
    }
    for child in list(defs):
        signature = ET.tostring(child, encoding="unicode", method="xml")
        if signature in existing:
            continue
        root_defs.append(child)
        existing.add(signature)


def _make_animated_svg_text(
    frame_svg_texts: Sequence[str],
    size: Size,
    durations: Sequence[int],
    loop: int,
) -> str:
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "baseProfile": "full",
            "width": f"{size[0]}px",
            "height": f"{size[1]}px",
            "viewBox": f"0 0 {size[0]} {size[1]}",
        },
    )

    for i, svg_text in enumerate(frame_svg_texts):
        frame_root = ET.fromstring(svg_text)
        group = ET.SubElement(
            root,
            f"{{{SVG_NS}}}g",
            {"opacity": "1" if i == 0 else "0"},
        )
        if len(frame_svg_texts) > 1:
            group.append(_make_opacity_animation(i, durations, loop))
        for child in list(frame_root):
            if _element_local_name(child) == "defs":
                _merge_svg_defs(root, child)
                continue
            group.append(child)

    return ET.tostring(root, encoding="unicode", method="xml")


def _apply_rdkit_global_options(target_dopts: MolDrawOptions) -> None:
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
) -> Tuple[
    Chem.Mol,
    ColorMap,
    ColorMap,
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

    highlight_atom_colors: ColorMap = {}
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

    highlight_bond_colors: ColorMap = {}
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
        highlight_bond_colors[i] = cast(RGBAColor, tuple(bond_color_arr.tolist()))

    return mol_copy, highlight_atom_colors, highlight_bond_colors


# =============================================================================
# Single Molecule Drawer
# =============================================================================


@overload
def MolToDofImage(
    mol: Union[Chem.Mol, Chem.RWMol],
    size: Optional[Size] = None,
    legend: str = "",
    *,
    use_svg: Literal[True] = True,
    return_image: Literal[True] = True,
    return_drawer: Literal[False] = False,
    settings: Optional[DofDrawSettings] = None,
    highlightAtoms: Optional[Sequence[int]] = None,  # noqa: N803
    highlightBonds: Optional[Sequence[int]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> "SVG": ...
@overload
def MolToDofImage(
    mol: Union[Chem.Mol, Chem.RWMol],
    size: Optional[Size] = None,
    legend: str = "",
    *,
    use_svg: Literal[False],
    return_image: Literal[True] = True,
    return_drawer: Literal[False] = False,
    settings: Optional[DofDrawSettings] = None,
    highlightAtoms: Optional[Sequence[int]] = None,  # noqa: N803
    highlightBonds: Optional[Sequence[int]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> Image.Image: ...
@overload
def MolToDofImage(
    mol: Union[Chem.Mol, Chem.RWMol],
    size: Optional[Size] = None,
    legend: str = "",
    *,
    use_svg: Literal[True] = True,
    return_image: Literal[False],
    return_drawer: Literal[False] = False,
    settings: Optional[DofDrawSettings] = None,
    highlightAtoms: Optional[Sequence[int]] = None,  # noqa: N803
    highlightBonds: Optional[Sequence[int]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> str: ...
@overload
def MolToDofImage(
    mol: Union[Chem.Mol, Chem.RWMol],
    size: Optional[Size] = None,
    legend: str = "",
    *,
    use_svg: Literal[False],
    return_image: Literal[False],
    return_drawer: Literal[False] = False,
    settings: Optional[DofDrawSettings] = None,
    highlightAtoms: Optional[Sequence[int]] = None,  # noqa: N803
    highlightBonds: Optional[Sequence[int]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> bytes: ...
@overload
def MolToDofImage(
    mol: Union[Chem.Mol, Chem.RWMol],
    size: Optional[Size] = None,
    legend: str = "",
    *,
    use_svg: bool = True,
    return_image: bool = True,
    return_drawer: Literal[True],
    settings: Optional[DofDrawSettings] = None,
    highlightAtoms: Optional[Sequence[int]] = None,  # noqa: N803
    highlightBonds: Optional[Sequence[int]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> MolDraw2D: ...
def MolToDofImage(  # noqa: N802
    mol: Union[Chem.Mol, Chem.RWMol],
    size: Optional[Size] = None,
    legend: str = "",
    use_svg: bool = True,
    return_image: bool = True,
    return_drawer: bool = False,
    *,
    settings: Optional[DofDrawSettings] = None,
    highlightAtoms: Optional[Sequence[int]] = None,  # noqa: N803
    highlightBonds: Optional[Sequence[int]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    filename: Optional[str] = None,
    **kwargs: Any,
) -> Union["SVG", str, Image.Image, bytes, MolDraw2D]:
    """Draw a single molecule with DOF effect."""
    warn_on_unicode_legend = bool(kwargs.pop("_warn_on_unicode_legend", True))

    if settings is None:
        settings = dofconfig
    if warn_on_unicode_legend and legend:
        _warn_if_unicode_legend([legend])

    draw_size = size if size else settings.default_size

    ready_mol, atom_colors, bond_colors = _prepare_mol_data(mol, settings)
    saturated_highlight_color = _get_saturated_highlight_color(highlightColor)
    if highlightAtoms:
        for atom_idx in highlightAtoms:
            atom_colors[atom_idx] = saturated_highlight_color
    if highlightBonds:
        for bond_idx in highlightBonds:
            bond_colors[bond_idx] = saturated_highlight_color
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
    if return_drawer:
        return drawer
    drawer.FinishDrawing()

    if use_svg:
        svg_text: str = drawer.GetDrawingText()
        if filename:
            with open(filename, "w") as f:
                f.write(svg_text)
        if return_image:
            return _make_svg_image(svg_text)
        return svg_text
    else:
        png_data: bytes = drawer.GetDrawingText()  # type: ignore[assignment, unused-ignore]
        if filename:
            with open(filename, "wb") as f:
                f.write(png_data)
        if return_image:
            return Image.open(io.BytesIO(png_data))
        return png_data


# =============================================================================
# Animated GIF Drawer
# =============================================================================


@overload
def MolsToDofGif(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol]],
    size: Optional[Size] = None,
    legends: Optional[Sequence[Union[str, None]]] = None,
    duration: Union[int, Sequence[int]] = 200,
    loop: int = 0,
    *,
    return_image: Literal[True] = True,
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> Image.Image: ...
@overload
def MolsToDofGif(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol]],
    size: Optional[Size] = None,
    legends: Optional[Sequence[Union[str, None]]] = None,
    duration: Union[int, Sequence[int]] = 200,
    loop: int = 0,
    *,
    return_image: Literal[False],
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> bytes: ...
def MolsToDofGif(  # noqa: N802
    mols: Sequence[Union[Chem.Mol, Chem.RWMol]],
    size: Optional[Size] = None,
    legends: Optional[Sequence[Union[str, None]]] = None,
    duration: Union[int, Sequence[int]] = 200,
    loop: int = 0,
    *,
    return_image: bool = True,
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    filename: Optional[str] = None,
    **kwargs: Any,
) -> Union[Image.Image, bytes]:
    """
    Draw a sequence of molecules as an animated GIF with DOF effect.

    Each molecule is rendered as one PNG frame through MolToDofImage, then Pillow
    combines the frames into a GIF.
    """
    if settings is None:
        settings = dofconfig
    if not mols:
        raise ValueError("mols must contain at least one molecule")
    if legends is not None and len(legends) != len(mols):
        raise ValueError("legends must have the same length as mols")
    if highlightAtomLists is not None and len(highlightAtomLists) != len(mols):
        raise ValueError("highlightAtomLists must have the same length as mols")
    if highlightBondLists is not None and len(highlightBondLists) != len(mols):
        raise ValueError("highlightBondLists must have the same length as mols")

    gif_duration = _normalize_frame_durations(duration, len(mols))
    draw_size = size if size else settings.default_size
    if legends is not None:
        _warn_if_unicode_legend([str(legend) for legend in legends if legend])
    frames: List[Image.Image] = []

    for i, mol in enumerate(mols):
        legend = str(legends[i]) if legends is not None and legends[i] else ""
        highlight_atoms = (
            highlightAtomLists[i] if highlightAtomLists is not None else None
        )
        highlight_bonds = (
            highlightBondLists[i] if highlightBondLists is not None else None
        )
        frame = cast(
            Image.Image,
            MolToDofImage(
                mol,
                size=draw_size,
                legend=legend,
                use_svg=False,
                return_image=True,
                settings=settings,
                highlightAtoms=highlight_atoms,
                highlightBonds=highlight_bonds,
                highlightColor=highlightColor,
                _warn_on_unicode_legend=False,
                **kwargs,
            ),
        ).convert("RGB")
        frames.append(frame)

    byte_arr = io.BytesIO()
    frames[0].save(
        byte_arr,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=gif_duration,
        loop=loop,
    )
    gif_data = byte_arr.getvalue()

    if filename:
        with open(filename, "wb") as f:
            f.write(gif_data)

    if return_image:
        return Image.open(io.BytesIO(gif_data))
    return gif_data


@overload
def MolsToDofSvgAnimation(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol]],
    size: Optional[Size] = None,
    legends: Optional[Sequence[Union[str, None]]] = None,
    duration: Union[int, Sequence[int]] = 200,
    loop: int = 0,
    *,
    return_image: Literal[True] = True,
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> "SVG": ...
@overload
def MolsToDofSvgAnimation(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol]],
    size: Optional[Size] = None,
    legends: Optional[Sequence[Union[str, None]]] = None,
    duration: Union[int, Sequence[int]] = 200,
    loop: int = 0,
    *,
    return_image: Literal[False],
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> str: ...
def MolsToDofSvgAnimation(  # noqa: N802
    mols: Sequence[Union[Chem.Mol, Chem.RWMol]],
    size: Optional[Size] = None,
    legends: Optional[Sequence[Union[str, None]]] = None,
    duration: Union[int, Sequence[int]] = 200,
    loop: int = 0,
    *,
    return_image: bool = True,
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    filename: Optional[str] = None,
    **kwargs: Any,
) -> Union["SVG", str]:
    """
    Draw a sequence of molecules as an animated SVG with DOF effect.

    The output uses SMIL opacity animation and stays as vector SVG.
    """
    if settings is None:
        settings = dofconfig
    if not mols:
        raise ValueError("mols must contain at least one molecule")
    if legends is not None and len(legends) != len(mols):
        raise ValueError("legends must have the same length as mols")
    if highlightAtomLists is not None and len(highlightAtomLists) != len(mols):
        raise ValueError("highlightAtomLists must have the same length as mols")
    if highlightBondLists is not None and len(highlightBondLists) != len(mols):
        raise ValueError("highlightBondLists must have the same length as mols")

    svg_durations = _duration_list(duration, len(mols))
    draw_size = size if size else settings.default_size
    if legends is not None:
        _warn_if_unicode_legend([str(legend) for legend in legends if legend])
    frame_svg_texts: List[str] = []

    for i, mol in enumerate(mols):
        legend = str(legends[i]) if legends is not None and legends[i] else ""
        highlight_atoms = (
            highlightAtomLists[i] if highlightAtomLists is not None else None
        )
        highlight_bonds = (
            highlightBondLists[i] if highlightBondLists is not None else None
        )
        frame_svg_texts.append(
            cast(
                str,
                MolToDofImage(
                    mol,
                    size=draw_size,
                    legend=legend,
                    use_svg=True,
                    return_image=False,
                    settings=settings,
                    highlightAtoms=highlight_atoms,
                    highlightBonds=highlight_bonds,
                    highlightColor=highlightColor,
                    _warn_on_unicode_legend=False,
                    **kwargs,
                ),
            )
        )

    svg_text = _make_animated_svg_text(frame_svg_texts, draw_size, svg_durations, loop)
    if filename:
        with open(filename, "w") as f:
            f.write(svg_text)
    if return_image:
        return _make_svg_image(svg_text)
    return svg_text


# =============================================================================
# Grid Drawer
# =============================================================================


@overload
def MolsToGridDofImage(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol, None]],
    molsPerRow: int = 3,  # noqa: N803
    subImgSize: Size = (300, 300),  # noqa: N803
    legends: Optional[Sequence[Union[str, None]]] = None,
    *,
    use_svg: Literal[True] = True,
    return_image: Literal[True] = True,
    return_drawer: Literal[False] = False,
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> "SVG": ...
@overload
def MolsToGridDofImage(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol, None]],
    molsPerRow: int = 3,  # noqa: N803
    subImgSize: Size = (300, 300),  # noqa: N803
    legends: Optional[Sequence[Union[str, None]]] = None,
    *,
    use_svg: Literal[False],
    return_image: Literal[True] = True,
    return_drawer: Literal[False] = False,
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> Image.Image: ...
@overload
def MolsToGridDofImage(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol, None]],
    molsPerRow: int = 3,  # noqa: N803
    subImgSize: Size = (300, 300),  # noqa: N803
    legends: Optional[Sequence[Union[str, None]]] = None,
    *,
    use_svg: Literal[True] = True,
    return_image: Literal[False],
    return_drawer: Literal[False] = False,
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> str: ...
@overload
def MolsToGridDofImage(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol, None]],
    molsPerRow: int = 3,  # noqa: N803
    subImgSize: Size = (300, 300),  # noqa: N803
    legends: Optional[Sequence[Union[str, None]]] = None,
    *,
    use_svg: Literal[False],
    return_image: Literal[False],
    return_drawer: Literal[False] = False,
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> bytes: ...
@overload
def MolsToGridDofImage(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol, None]],
    molsPerRow: int = 3,  # noqa: N803
    subImgSize: Size = (300, 300),  # noqa: N803
    legends: Optional[Sequence[Union[str, None]]] = None,
    *,
    use_svg: bool = True,
    return_image: bool = True,
    return_drawer: Literal[True],
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    **kwargs: Any,
) -> MolDraw2D: ...
def MolsToGridDofImage(  # noqa: N802
    mols: Sequence[Union[Chem.Mol, Chem.RWMol, None]],
    molsPerRow: int = 3,  # noqa: N803
    subImgSize: Size = (300, 300),  # noqa: N803
    legends: Optional[Sequence[Union[str, None]]] = None,
    use_svg: bool = True,
    return_image: bool = True,
    return_drawer: bool = False,
    *,
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,  # noqa: N803
    highlightColor: RGBAColor = (1.0, 0.0, 0.0, 1.0),  # noqa: N803
    filename: Optional[str] = None,
    **kwargs: Any,
) -> Union["SVG", str, Image.Image, bytes]:
    """
    Draw a grid of molecules with DOF effect.
    Compatible with RDKit's Chem.Draw.MolsToGridImage arguments.
    """
    if settings is None:
        settings = dofconfig
    if legends is not None:
        _warn_if_unicode_legend([str(legend) for legend in legends if legend])

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
            empty_svg_text = drawer.GetDrawingText()
            if filename:
                with open(filename, "w") as f:
                    f.write(empty_svg_text)
            return _make_svg_image(empty_svg_text) if return_image else empty_svg_text
        else:
            # For non-SVG, return a blank PIL image or its byte representation
            blank_image = Image.new("RGB", (full_width, full_height), (255, 255, 255))
            if filename:
                blank_image.save(filename)
            if return_image:
                return blank_image
            else:
                byte_arr = io.BytesIO()
                blank_image.save(byte_arr, format="PNG")
                return byte_arr.getvalue()

    if highlightAtomLists:
        assert len(highlightAtomLists) == len(mols), (
            "highlightAtomLists must have the same length as mols"
        )
    if highlightBondLists:
        assert len(highlightBondLists) == len(mols), (
            "highlightBondLists must have the same length as mols"
        )
    valid_mols: List[Union[Chem.Mol, Chem.RWMol]] = []
    valid_legends: List[str] = []

    all_atom_colors: List[ColorMap] = []
    all_bond_colors: List[ColorMap] = []
    all_highlight_atoms: List[List[int]] = []
    all_highlight_bonds: List[List[int]] = []
    if highlightAtomLists is None:
        highlightAtomLists = [[] for _ in mols]  # noqa: N806
    if highlightBondLists is None:
        highlightBondLists = [[] for _ in mols]  # noqa: N806
    saturated_highlight_color = _get_saturated_highlight_color(highlightColor)
    for i, (m, atom_list, bond_list) in enumerate(
        zip(mols, highlightAtomLists, highlightBondLists)
    ):
        if m is None:
            m = Chem.Mol()

        try:
            ready_mol, atom_colors, bond_colors = _prepare_mol_data(m, settings)
        except Exception:
            ready_mol = m
            atom_colors, bond_colors = {}, {}
        for atom_idx in atom_list:
            atom_colors[atom_idx] = saturated_highlight_color
        for bond_idx in bond_list:
            bond_colors[bond_idx] = saturated_highlight_color
        valid_mols.append(ready_mol)

        if legends and i < len(legends) and legends[i]:
            legend = str(legends[i])
            valid_legends.append(legend)
        else:
            valid_legends.append("")

        all_atom_colors.append(atom_colors)
        all_bond_colors.append(bond_colors)
        all_highlight_atoms.append(list(atom_colors.keys()))
        all_highlight_bonds.append(list(bond_colors.keys()))

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
    if return_drawer:
        return drawer
    drawer.FinishDrawing()
    if use_svg:
        grid_svg_text: str = drawer.GetDrawingText()
        if filename:
            with open(filename, "w") as f:
                f.write(grid_svg_text)
        if return_image:
            return _make_svg_image(grid_svg_text)
        return grid_svg_text
    else:
        png_data: bytes = drawer.GetDrawingText()  # type: ignore[assignment, unused-ignore]
        if filename:
            with open(filename, "wb") as f:
                f.write(png_data)
        if return_image:
            return Image.open(io.BytesIO(png_data))
        return png_data
