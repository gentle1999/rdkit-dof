# 使用指南

[返回 README](../../README.zh.md) | [English](../usage.md)

如果需要可直接运行并带有输出结果的示例，请打开 [中文 Quickstart Notebook](../../examples/rdkit_dof_quickstart.zh.ipynb)。

## 单分子绘图

使用 `MolToDofImage` 绘制单个分子。3D 构象可以提供最佳景深效果；如果分子没有构象，`rdkit-dof` 会自动计算 2D 坐标，并绘制深度变化较平的图像。

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

## Legend 文本

不支持 Unicode/非 ASCII legend；`rdkit-dof` 会发出 warning 并原样交给
RDKit。为保证可移植输出，请使用 ASCII legend。

## 输出模式

默认输出 SVG。设置 `use_svg=False` 可输出 PNG。

```python
svg_text = MolToDofImage(mol, return_image=False)
png_bytes = MolToDofImage(mol, use_svg=False, return_image=False)
png_image = MolToDofImage(mol, use_svg=False, return_image=True)
```

传入 `filename` 可直接保存文件。建议文件扩展名与 `use_svg` 保持一致。

## 高亮

可以通过原子和键索引进行高亮，并指定 RGBA 颜色。被高亮的原子和键会以指定 RGB 颜色完全不透明显示，不参与景深淡化。

```python
MolToDofImage(
    mol,
    highlightAtoms=[0, 1],
    highlightBonds=[0],
    highlightColor=(1.0, 0.0, 0.0, 1.0),
)
```

## 网格绘图

使用 `MolsToGridDofImage` 绘制多个分子。`None` 会渲染为空白单元格；传入空列表会返回空白图像。

```python
from rdkit_dof import MolsToGridDofImage

MolsToGridDofImage(
    [mol, mol, None],
    molsPerRow=3,
    legends=["A", "B", ""],
    filename="grid.svg",
)
```

逐分子高亮列表需要与 `mols` 对齐。

```python
MolsToGridDofImage(
    mols,
    highlightAtomLists=[[0], [], [1, 2]],
    highlightBondLists=[[0], [], []],
)
```

## 动图绘制

当一组 RDKit 分子需要生成动图时，使用 `MolsToDofGif`。每个分子会被渲染为一帧。

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

如果不同帧需要不同停留时间，可以传入逐帧 `duration` 序列。

```python
gif_bytes = MolsToDofGif(
    mols,
    duration=[150, 250, 400],
    return_image=False,
)
```

如果需要矢量 SVG 动画输出，可以使用 `MolsToDofSvgAnimation`。

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

## 局部配置

如果只希望某一次绘图使用不同样式，而不修改全局状态，可以传入局部 `DofDrawSettings`。

```python
from rdkit_dof import DofDrawSettings, MolToDofImage

settings = DofDrawSettings(preset_style="dark", min_alpha=0.25)
MolToDofImage(mol, settings=settings)
```

## 自定义 RDKit 绘制

设置 `return_drawer=True` 可以在 `FinishDrawing()` 前获得底层 RDKit drawer。

```python
drawer = MolToDofImage(mol, return_drawer=True)
# 在这里添加自定义 RDKit 绘制操作。
drawer.FinishDrawing()
svg_text = drawer.GetDrawingText()
```

## Jupyter/IPython 集成

启用后，`rdkit-dof` 会注册 formatter，让 notebook 中的 RDKit `Mol` 和 `RWMol` 对象以 DOF 风格显示。

```python
from rdkit_dof import dofconfig

dofconfig.enable_ipython_integration(True)
dofconfig.enable_ipython_integration(False)
```

`IPythonConsole.drawOptions` 中的 RDKit 选项会在绘制前复制到当前 drawer。传给 `MolToDofImage` 或 `MolsToGridDofImage` 的关键字参数会在其后覆盖对应选项。

## 重新生成展示图

```bash
python scripts/_generate_comparison_images.py
```
