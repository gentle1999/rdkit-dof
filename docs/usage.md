# Usage Guide

[Back to README](../README.md) | [简体中文](zh/usage.md)

For runnable examples with saved outputs, open the [English Quickstart Notebook](../examples/rdkit_dof_quickstart.en.ipynb).

## Single Molecule Rendering

Use `MolToDofImage` for one molecule. A 3D conformer gives the best DOF effect; if no conformer exists, `rdkit-dof` computes 2D coordinates automatically and renders a flat-depth image.

```python
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit_dof import MolToDofImage

mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
EmbedMolecule(mol, randomSeed=42)
MMFFOptimizeMolecule(mol)

MolToDofImage(mol, legend="Ethanol", filename="ethanol.svg")
```

## Legend Text

Unicode/non-ASCII legend text is not supported. `rdkit-dof` emits a warning and
passes the text through to RDKit unchanged. Use ASCII legends for portable
output.

## Output Modes

SVG is the default. Set `use_svg=False` for PNG output.

```python
svg_text = MolToDofImage(mol, return_image=False)
png_bytes = MolToDofImage(mol, use_svg=False, return_image=False)
png_image = MolToDofImage(mol, use_svg=False, return_image=True)
```

Pass `filename` to save directly. The file extension should match `use_svg`.

## Highlighting

Highlight atom and bond indices with an RGBA color. Highlighted atoms and bonds use the given RGB color at full opacity instead of depth fading.

```python
MolToDofImage(
    mol,
    highlightAtoms=[0, 1],
    highlightBonds=[0],
    highlightColor=(1.0, 0.0, 0.0, 1.0),
)
```

## Grid Rendering

Use `MolsToGridDofImage` for multiple molecules. `None` entries are rendered as empty cells, and an empty molecule list returns a blank image.

```python
from rdkit_dof import MolsToGridDofImage

MolsToGridDofImage(
    [mol, mol, None],
    molsPerRow=3,
    legends=["A", "B", ""],
    filename="grid.svg",
)
```

Per-molecule highlighting uses lists aligned with `mols`.

```python
MolsToGridDofImage(
    mols,
    highlightAtomLists=[[0], [], [1, 2]],
    highlightBondLists=[[0], [], []],
)
```

## Animation Rendering

Use `MolsToDofGif` when a sequence of RDKit molecules should become an animated
GIF. Each molecule is rendered as one frame.

```python
from rdkit_dof import MolsToDofGif

gif_image = MolsToDofGif(
    mols,
    size=(500, 400),
    legends=["Frame 1", "Frame 2", "Frame 3"],
    duration=250,
    filename="molecules.gif",
)
```

Pass a duration sequence when individual frames need different timings.

```python
gif_bytes = MolsToDofGif(
    mols,
    duration=[150, 250, 400],
    return_image=False,
)
```

Use `MolsToDofSvgAnimation` for vector SVG animation output.

```python
from rdkit_dof import MolsToDofSvgAnimation

svg_text = MolsToDofSvgAnimation(
    mols,
    size=(500, 400),
    duration=250,
    return_image=False,
    filename="molecules.svg",
)
```

## Local Settings

Use a local `DofDrawSettings` object when one drawing call should use a different style without changing global state.

```python
from rdkit_dof import DofDrawSettings, MolToDofImage

settings = DofDrawSettings(preset_style="dark", min_alpha=0.25)
MolToDofImage(mol, settings=settings)
```

## Custom RDKit Drawing

Set `return_drawer=True` to get the underlying RDKit drawer before `FinishDrawing()`.

```python
drawer = MolToDofImage(mol, return_drawer=True)
# Add custom RDKit drawing operations here.
drawer.FinishDrawing()
svg_text = drawer.GetDrawingText()
```

## Jupyter/IPython Integration

When enabled, `rdkit-dof` registers a formatter so RDKit `Mol` and `RWMol` objects display with the DOF style in notebooks.

```python
from rdkit_dof import dofconfig

dofconfig.enable_ipython_integration(True)
dofconfig.enable_ipython_integration(False)
```

Options configured through `IPythonConsole.drawOptions` are copied into each drawer before rendering. Explicit keyword arguments passed to `MolToDofImage` or `MolsToGridDofImage` are applied afterward.

## Regenerating Showcase Images

```bash
python scripts/_generate_comparison_images.py
```
