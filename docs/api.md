# API Reference

[Back to README](../README.md) | [简体中文](zh/api.md)

## `MolToDofImage`

```python
MolToDofImage(
    mol: Union[Chem.Mol, Chem.RWMol],
    size: Optional[Tuple[int, int]] = None,
    legend: str = "",
    use_svg: bool = True,
    return_image: bool = True,
    return_drawer: bool = False,
    *,
    settings: Optional[DofDrawSettings] = None,
    highlightAtoms: Optional[Sequence[int]] = None,
    highlightBonds: Optional[Sequence[int]] = None,
    highlightColor: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    filename: Optional[str] = None,
    **kwargs: Any,
) -> Union["SVG", str, Image.Image, bytes, MolDraw2D]
```

Draws one molecule with the DOF effect.

| Parameter | Description |
| --- | --- |
| `mol` | RDKit molecule. 3D conformers drive depth; missing conformers are converted to 2D coordinates. |
| `size` | Output size as `(width, height)`. Defaults to `settings.default_size`. |
| `legend` | Text rendered below the molecule. |
| `use_svg` | `True` for SVG, `False` for PNG. |
| `return_image` | `True` returns an IPython SVG or Pillow image. `False` returns raw SVG text or PNG bytes. |
| `return_drawer` | Returns the RDKit `MolDraw2D` drawer before `FinishDrawing()`. |
| `settings` | Optional local `DofDrawSettings`. Defaults to global `dofconfig`. |
| `highlightAtoms` | Atom indices to highlight. |
| `highlightBonds` | Bond indices to highlight. |
| `highlightColor` | RGBA highlight color, with channels in the `0.0` to `1.0` range. The RGB channels are rendered at full opacity and are not depth-faded. |
| `filename` | Saves the generated SVG or PNG directly to a path. |
| `**kwargs` | Any matching RDKit `MolDrawOptions` attributes. |

## `MolsToGridDofImage`

```python
MolsToGridDofImage(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol, None]],
    molsPerRow: int = 3,
    subImgSize: Tuple[int, int] = (300, 300),
    legends: Optional[Sequence[Union[str, None]]] = None,
    use_svg: bool = True,
    return_image: bool = True,
    return_drawer: bool = False,
    *,
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,
    highlightColor: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    filename: Optional[str] = None,
    **kwargs: Any,
) -> Union["SVG", str, Image.Image, bytes, MolDraw2D]
```

Draws a grid of molecules with the DOF effect.

| Parameter | Description |
| --- | --- |
| `mols` | Molecules to draw. `None` entries become empty cells; an empty list returns a blank image. |
| `molsPerRow` | Number of grid cells per row. |
| `subImgSize` | Per-cell size as `(width, height)`. |
| `legends` | Optional per-cell legends. |
| `highlightAtomLists` | Per-molecule atom highlight lists. Must match `len(mols)` when provided. |
| `highlightBondLists` | Per-molecule bond highlight lists. Must match `len(mols)` when provided. |
| Other parameters | Same behavior as `MolToDofImage`. |

## `DofDrawSettings`

```python
DofDrawSettings(
    preset_style="default",
    fog_color=(0.95, 0.95, 0.95),
    min_alpha=0.4,
    default_size=(800, 800),
    enable_ipython=True,
    atom_colors={},
    *,
    env_file=".env",
)
```

`DofDrawSettings` can be used directly for local configuration or through the global `dofconfig` object.

Extra keyword arguments are ignored for compatibility with the previous settings implementation.

## Public Exports

```python
from rdkit_dof import (
    DofDrawSettings,
    MolToDofImage,
    MolsToGridDofImage,
    dofconfig,
)
```
