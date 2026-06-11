# 配置说明

[返回 README](../../README.zh.md) | [English](../configuration.md)

`rdkit-dof` 使用标准库 dataclass 进行配置。可以通过全局 `dofconfig`、局部 `DofDrawSettings`，或 `.env` / 环境变量进行设置。

## 全局配置

```python
from rdkit_dof import dofconfig

dofconfig.use_style("nature")
dofconfig.fog_color = (0.1, 0.1, 0.1)
dofconfig.min_alpha = 0.3
dofconfig.default_size = (500, 500)
dofconfig.atom_colors[8] = (1.0, 0.2, 0.2)
```

`use_style()` 会切换预设样式，并重置自定义原子颜色。

## 局部配置

```python
from rdkit_dof import DofDrawSettings, MolToDofImage

settings = DofDrawSettings(
    preset_style="dark",
    min_alpha=0.25,
    default_size=(700, 500),
)

MolToDofImage(mol, settings=settings)
```

当只希望某一张图不同于全局默认值时，局部配置更合适。

## 预设样式

| 样式 | 说明 |
| --- | --- |
| `default` | 接近 RDKit 的基础颜色和浅色雾化。 |
| `nature` | 更柔和的论文图风格颜色。 |
| `jacs` | 高对比度的印刷风格颜色。 |
| `dark` | 明亮原子颜色和深色雾化。 |

当 `preset_style="dark"` 且 `fog_color` 仍为默认浅色雾化时，雾化颜色会自动改为 `(0.1, 0.1, 0.1)`。

## 环境变量

配置键使用 `RDKIT_DOF_` 前缀和大写蛇形命名。

```env
RDKIT_DOF_PRESET_STYLE=dark
RDKIT_DOF_FOG_COLOR=[0.1, 0.1, 0.1]
RDKIT_DOF_MIN_ALPHA=0.2
RDKIT_DOF_DEFAULT_SIZE=[600, 500]
RDKIT_DOF_ENABLE_IPYTHON=false
RDKIT_DOF_ATOM_COLORS={"8": [1.0, 0.2, 0.2]}
```

对于简单 `.env` 用例，配置值可以写成 JSON 风格的列表/字典，也可以写成 Python tuple/list 字面量。环境变量会覆盖 `.env` 中的值；显式构造参数会覆盖二者。

## 配置属性

| 属性 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `preset_style` | `default`, `nature`, `jacs`, `dark` | `default` | 基础样式预设。 |
| `fog_color` | RGB tuple | `(0.95, 0.95, 0.95)` | 雾化/背景颜色。 |
| `min_alpha` | float | `0.4` | 最远原子和键使用的 alpha。 |
| `default_size` | `(int, int)` | `(800, 800)` | `MolToDofImage` 的默认尺寸。 |
| `enable_ipython` | bool | `True` | 尽可能启用 notebook formatter 注册。 |
| `atom_colors` | mapping | 基于样式 | 原子序数到 RGB 颜色。 |

## Jupyter/IPython

```python
from rdkit_dof import dofconfig

dofconfig.enable_ipython_integration(True)
dofconfig.enable_ipython_integration(False)
```

启用后，RDKit `Mol` 和 `RWMol` 对象会在 notebook 中通过 `MolToDofImage` 渲染。来自 `IPythonConsole.drawOptions` 的 RDKit 选项会先被复制，然后再应用显式传入的绘图关键字参数。
