# API 参考

[返回 README](../../README.zh.md) | [English](../api.md)

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

绘制单个分子的 DOF 图像。

| 参数 | 说明 |
| --- | --- |
| `mol` | RDKit 分子。3D 构象用于计算深度；没有构象时会自动计算 2D 坐标。 |
| `size` | 输出尺寸 `(宽, 高)`。默认使用 `settings.default_size`。 |
| `legend` | 分子图下方的文字。 |
| `use_svg` | `True` 输出 SVG，`False` 输出 PNG。 |
| `return_image` | `True` 返回 IPython SVG 或 Pillow 图像；`False` 返回 SVG 文本或 PNG bytes。 |
| `return_drawer` | 在 `FinishDrawing()` 前返回 RDKit `MolDraw2D` drawer。 |
| `settings` | 可选局部 `DofDrawSettings`。默认使用全局 `dofconfig`。 |
| `highlightAtoms` | 需要高亮的原子索引。 |
| `highlightBonds` | 需要高亮的键索引。 |
| `highlightColor` | RGBA 高亮颜色，通道范围为 `0.0` 到 `1.0`。RGB 通道会以完全不透明方式渲染，不参与景深淡化。 |
| `filename` | 直接保存生成的 SVG 或 PNG。 |
| `**kwargs` | 任何匹配的 RDKit `MolDrawOptions` 属性。 |

不支持 Unicode/非 ASCII legend；`rdkit-dof` 会发出 warning 并原样交给
RDKit。为保证可移植输出，请使用 ASCII legend。

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

绘制分子网格的 DOF 图像。

| 参数 | 说明 |
| --- | --- |
| `mols` | 要绘制的分子。`None` 会变成空白单元格；空列表会返回空白图像。 |
| `molsPerRow` | 每行单元格数量。 |
| `subImgSize` | 每个单元格尺寸 `(宽, 高)`。 |
| `legends` | 可选的单元格文字。 |
| `highlightAtomLists` | 逐分子的原子高亮列表；提供时长度必须等于 `len(mols)`。 |
| `highlightBondLists` | 逐分子的键高亮列表；提供时长度必须等于 `len(mols)`。 |
| 其他参数 | 行为与 `MolToDofImage` 相同。 |

## `MolsToDofGif`

```python
MolsToDofGif(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol]],
    size: Optional[Tuple[int, int]] = None,
    legends: Optional[Sequence[Union[str, None]]] = None,
    duration: Union[int, Sequence[int]] = 200,
    loop: int = 0,
    *,
    return_image: bool = True,
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,
    highlightColor: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    filename: Optional[str] = None,
    **kwargs: Any,
) -> Union[Image.Image, bytes]
```

将一组分子绘制为带 DOF 效果的 GIF 动图。每个分子渲染为一帧，并使用相同尺寸。

| 参数 | 说明 |
| --- | --- |
| `mols` | 作为帧绘制的分子列表，至少需要包含一个分子。 |
| `size` | 每帧尺寸 `(宽, 高)`。默认使用 `settings.default_size`。 |
| `legends` | 可选的逐帧文字；提供时长度必须等于 `len(mols)`。 |
| `duration` | 每帧持续时间，单位为毫秒。可传入统一整数，也可传入长度等于 `len(mols)` 的逐帧序列。 |
| `loop` | 传给 Pillow 的 GIF 循环次数。`0` 表示无限循环。 |
| `return_image` | `True` 返回 Pillow GIF 图像；`False` 返回 GIF bytes。 |
| `highlightAtomLists` | 逐帧原子高亮列表；提供时长度必须等于 `len(mols)`。 |
| `highlightBondLists` | 逐帧键高亮列表；提供时长度必须等于 `len(mols)`。 |
| 其他参数 | 行为与 `MolToDofImage` 相同。 |

## `MolsToDofSvgAnimation`

```python
MolsToDofSvgAnimation(
    mols: Sequence[Union[Chem.Mol, Chem.RWMol]],
    size: Optional[Tuple[int, int]] = None,
    legends: Optional[Sequence[Union[str, None]]] = None,
    duration: Union[int, Sequence[int]] = 200,
    loop: int = 0,
    *,
    return_image: bool = True,
    settings: Optional[DofDrawSettings] = None,
    highlightAtomLists: Optional[Sequence[Sequence[int]]] = None,
    highlightBondLists: Optional[Sequence[Sequence[int]]] = None,
    highlightColor: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    filename: Optional[str] = None,
    **kwargs: Any,
) -> Union["SVG", str]
```

将一组分子绘制为矢量 SVG 动画，内部使用 SMIL 的透明度动画切换帧。每个分子渲染为一帧，并使用相同尺寸。

| 参数 | 说明 |
| --- | --- |
| `mols` | 作为帧绘制的分子列表，至少需要包含一个分子。 |
| `size` | 每帧尺寸 `(宽, 高)`。默认使用 `settings.default_size`。 |
| `legends` | 可选的逐帧文字；提供时长度必须等于 `len(mols)`。 |
| `duration` | 每帧持续时间，单位为毫秒。可传入统一整数，也可传入长度等于 `len(mols)` 的逐帧序列。 |
| `loop` | SVG 动画重复次数。`0` 表示无限重复。 |
| `return_image` | `True` 返回 IPython SVG 对象；`False` 返回 SVG 文本。 |
| `highlightAtomLists` | 逐帧原子高亮列表；提供时长度必须等于 `len(mols)`。 |
| `highlightBondLists` | 逐帧键高亮列表；提供时长度必须等于 `len(mols)`。 |
| 其他参数 | 行为与 `MolToDofImage` 相同。 |

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

`DofDrawSettings` 可用于局部配置，也可通过全局 `dofconfig` 使用。

额外关键字参数会被忽略，以保持与旧版 settings 实现的兼容性。

## 公开导出

```python
from rdkit_dof import (
    DofDrawSettings,
    MolToDofImage,
    MolsToDofGif,
    MolsToDofSvgAnimation,
    MolsToGridDofImage,
    dofconfig,
)
```
