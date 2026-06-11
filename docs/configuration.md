# Configuration

[Back to README](../README.md) | [简体中文](zh/configuration.md)

`rdkit-dof` uses standard-library dataclasses for configuration. You can configure drawing globally through `dofconfig`, locally with `DofDrawSettings`, or from `.env` / environment variables.

## Global Configuration

```python
from rdkit_dof import dofconfig

dofconfig.use_style("nature")
dofconfig.fog_color = (0.1, 0.1, 0.1)
dofconfig.min_alpha = 0.3
dofconfig.default_size = (500, 500)
dofconfig.atom_colors[8] = (1.0, 0.2, 0.2)
```

`use_style()` switches to a preset and resets custom atom colors.

## Local Configuration

```python
from rdkit_dof import DofDrawSettings, MolToDofImage

settings = DofDrawSettings(
    preset_style="dark",
    min_alpha=0.25,
    default_size=(700, 500),
)

MolToDofImage(mol, settings=settings)
```

Local settings are useful when you want one image to differ from the global defaults.

## Preset Styles

| Style | Notes |
| --- | --- |
| `default` | RDKit-like base colors with light fog. |
| `nature` | Softer publication-style colors. |
| `jacs` | High-contrast print-style colors. |
| `dark` | Bright atom colors and dark fog. |

When `preset_style="dark"` and `fog_color` is still the default light fog, the fog color automatically changes to `(0.1, 0.1, 0.1)`.

## Environment Variables

Configuration keys use the `RDKIT_DOF_` prefix and upper snake case.

```env
RDKIT_DOF_PRESET_STYLE=dark
RDKIT_DOF_FOG_COLOR=[0.1, 0.1, 0.1]
RDKIT_DOF_MIN_ALPHA=0.2
RDKIT_DOF_DEFAULT_SIZE=[600, 500]
RDKIT_DOF_ENABLE_IPYTHON=false
RDKIT_DOF_ATOM_COLORS={"8": [1.0, 0.2, 0.2]}
```

Values may be JSON-style lists/dicts or Python tuple/list literals for simple `.env` use cases. Environment variables override values loaded from `.env`. Explicit constructor arguments override both.

## Properties

| Property | Type | Default | Description |
| --- | --- | --- | --- |
| `preset_style` | `default`, `nature`, `jacs`, `dark` | `default` | Base style preset. |
| `fog_color` | RGB tuple | `(0.95, 0.95, 0.95)` | Fog/background color. |
| `min_alpha` | float | `0.4` | Alpha used for the farthest atoms and bonds. |
| `default_size` | `(int, int)` | `(800, 800)` | Default size for `MolToDofImage`. |
| `enable_ipython` | bool | `True` | Enables notebook formatter registration when possible. |
| `atom_colors` | mapping | Based on style | Atomic number to RGB color. |

## Jupyter/IPython

```python
from rdkit_dof import dofconfig

dofconfig.enable_ipython_integration(True)
dofconfig.enable_ipython_integration(False)
```

If enabled, RDKit `Mol` and `RWMol` objects render through `MolToDofImage` in notebooks. RDKit options from `IPythonConsole.drawOptions` are copied before explicit drawing keyword arguments are applied.
