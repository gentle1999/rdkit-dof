"""
Author: TMJ
Date: 2025-12-01 12:38:03
LastEditors: TMJ
LastEditTime: 2025-12-02 11:12:43
Description: 请填写简介
"""

import ast
import html
import json
import os
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union, cast

from rdkit.Chem import Mol, RWMol

from .palettes import (
    DARK_NEON_STYLE,
    DEFAULT_STYLE,
    JACS_STYLE,
    NATURE_STYLE,
    AtomColorMap,
)

StyleName = Literal["default", "nature", "jacs", "dark"]
RGBColor = Tuple[float, float, float]
Size = Tuple[int, int]
EnvFile = Optional[Union[str, Path]]

_ENV_PREFIX = "RDKIT_DOF_"
_DEFAULT_FOG_COLOR: RGBColor = (0.95, 0.95, 0.95)
_UNSET = object()
_STYLE_MAP: Dict[StyleName, AtomColorMap] = {
    "default": DEFAULT_STYLE,
    "nature": NATURE_STYLE,
    "jacs": JACS_STYLE,
    "dark": DARK_NEON_STYLE,
}


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv(env_file: EnvFile) -> Dict[str, str]:
    if env_file is None:
        return {}

    path = Path(env_file)
    if not path.is_file():
        return {}

    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        values[key.strip()] = _strip_env_value(value)
    return values


def _load_env_values(env_file: EnvFile) -> Dict[str, str]:
    values = _load_dotenv(env_file)
    values.update(
        {key: value for key, value in os.environ.items() if key.startswith(_ENV_PREFIX)}
    )
    return values


def _select_value(
    field_name: str,
    explicit_value: object,
    env_values: Mapping[str, str],
    default_value: object,
) -> object:
    if explicit_value is not _UNSET:
        return explicit_value

    env_key = f"{_ENV_PREFIX}{field_name.upper()}"
    if env_key in env_values:
        return _parse_raw_value(env_values[env_key])

    return default_value


def _parse_raw_value(raw_value: str) -> object:
    value = raw_value.strip()
    if value == "":
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _coerce_style(value: object) -> StyleName:
    if value in _STYLE_MAP:
        return cast(StyleName, value)
    raise ValueError(
        "preset_style must be one of 'default', 'nature', 'jacs', or 'dark'"
    )


def _coerce_rgb(value: object, field_name: str) -> RGBColor:
    if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of three numbers")
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    return (float(value[0]), float(value[1]), float(value[2]))


def _coerce_size(value: object) -> Size:
    if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes)):
        raise ValueError("default_size must be a sequence of two integers")
    if len(value) != 2:
        raise ValueError("default_size must contain exactly two values")
    return (int(value[0]), int(value[1]))


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError("enable_ipython must be a boolean value")


def _coerce_float(value: object, field_name: str) -> float:
    if isinstance(value, (str, int, float)):
        return float(value)
    raise ValueError(f"{field_name} must be a number")


def _coerce_atom_colors(value: object) -> AtomColorMap:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise ValueError("atom_colors must be a mapping of atomic number to RGB color")

    colors: AtomColorMap = {}
    for atomic_num, color in value.items():
        colors[int(atomic_num)] = _coerce_rgb(color, "atom_colors")
    return colors


@dataclass(init=False)
class DofDrawSettings:
    """
    Drawing configuration with optional RDKIT_DOF_ environment overrides.

    Logic:
    1. load preset style
    2. if dark mode, and fog color is default, set fog color to (0.1, 0.1, 0.1)
    3. override specific atoms with user-provided atom_colors
    """

    preset_style: StyleName
    fog_color: RGBColor
    min_alpha: float
    default_size: Size
    enable_ipython: bool
    atom_colors: AtomColorMap
    _ipython_formatter_backups: Dict[Tuple[int, Any], object] = field(
        default_factory=dict, init=False, repr=False
    )

    def __init__(
        self,
        preset_style: object = _UNSET,
        fog_color: object = _UNSET,
        min_alpha: object = _UNSET,
        default_size: object = _UNSET,
        enable_ipython: object = _UNSET,
        atom_colors: object = _UNSET,
        *,
        env_file: EnvFile = ".env",
        **_: object,
    ) -> None:
        env_values = _load_env_values(env_file)

        self.preset_style = _coerce_style(
            _select_value("preset_style", preset_style, env_values, "default")
        )
        self.fog_color = _coerce_rgb(
            _select_value("fog_color", fog_color, env_values, _DEFAULT_FOG_COLOR),
            "fog_color",
        )
        self.min_alpha = _coerce_float(
            _select_value("min_alpha", min_alpha, env_values, 0.4),
            "min_alpha",
        )
        self.default_size = _coerce_size(
            _select_value("default_size", default_size, env_values, (800, 800))
        )
        self.enable_ipython = _coerce_bool(
            _select_value("enable_ipython", enable_ipython, env_values, True)
        )
        self.atom_colors = _coerce_atom_colors(
            _select_value("atom_colors", atom_colors, env_values, {})
        )
        self._ipython_formatter_backups = {}
        self._apply_style_logic(self.preset_style)

    def _apply_style_logic(self, style: StyleName) -> None:
        base_colors = _STYLE_MAP[style].copy()

        if self.preset_style == "dark" and self.fog_color == _DEFAULT_FOG_COLOR:
            self.fog_color = (0.1, 0.1, 0.1)

        if self.atom_colors:
            base_colors.update(self.atom_colors)

        self.atom_colors = base_colors

    def get_atom_color(self, atomic_num: int) -> RGBColor:
        return self.atom_colors.get(
            atomic_num, self.atom_colors.get(6, (0.2, 0.2, 0.2))
        )

    def use_style(self, style: StyleName) -> None:
        """
        Switch to a different preset style, resetting any custom atom colors.
        """
        self.preset_style = _coerce_style(style)
        self.atom_colors = {}
        self._apply_style_logic(self.preset_style)

    def enable_ipython_integration(self, enable: bool = True) -> None:
        """
        Toggle whether to use the DOF effect drawer as the default renderer
        for RDKit Mol objects in Jupyter/IPython.

        Args:
            enable (bool): True to enable DOF rendering, False to restore RDKit default.
        """
        try:
            from IPython.core.getipython import (  # pyright: ignore[reportMissingImports]
                get_ipython,
            )
        except ImportError:
            return
        get_ipython_func = cast(Any, get_ipython)
        ip = get_ipython_func()
        if ip is None:
            return
        formatters = {
            "image/svg+xml": ip.display_formatter.formatters["image/svg+xml"],
            "text/html": ip.display_formatter.formatters["text/html"],
        }

        if enable:
            from .core import MolToDofImage

            def _dof_drawer_hook(mol: Union[Mol, RWMol], legend: str = "") -> str:
                return MolToDofImage(
                    mol,
                    legend=legend,
                    use_svg=True,
                    return_image=False,
                    settings=self,
                )

            def _dof_drawer_html_hook(
                mol: Union[Mol, RWMol],
            ) -> Optional[str]:
                ipython_console = import_module("rdkit.Chem.Draw.IPythonConsole")

                props = mol.GetPropsAsDict()
                if not ipython_console.ipython_showProperties or not props:
                    return None

                legend = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                svg_text = _dof_drawer_hook(mol, legend=legend)
                svg_start = svg_text.find("<svg")
                if svg_start >= 0:
                    svg_text = svg_text[svg_start:]
                rows = [
                    '<tr><td colspan="2" style="text-align: center;">'
                    f"{svg_text}</td></tr>"
                ]
                max_properties = int(ipython_console.ipython_maxProperties)
                for index, (prop_name, prop_value) in enumerate(props.items()):
                    if max_properties >= 0 and index >= max_properties:
                        rows.append(
                            '<tr><td colspan="2" style="text-align: center">'
                            "Property list truncated.<br />Increase "
                            "IPythonConsole.ipython_maxProperties (or set it to -1) "
                            "to see more properties.</td></tr>"
                        )
                        break
                    rows.append(
                        '<tr><th style="text-align: right">'
                        f"{html.escape(str(prop_name))}</th>"
                        '<td style="text-align: left">'
                        f"{html.escape(str(prop_value))}</td></tr>"
                    )
                return f"<table>{''.join(rows)}</table>"

            hooks = {
                "image/svg+xml": _dof_drawer_hook,
                "text/html": _dof_drawer_html_hook,
            }
            for mime_type, formatter in formatters.items():
                for mol_type in (Mol, RWMol):
                    backup_key = (id(formatter), mol_type)
                    if backup_key not in self._ipython_formatter_backups:
                        self._ipython_formatter_backups[backup_key] = (
                            formatter.type_printers.get(mol_type, _UNSET)
                        )
                    formatter.for_type(mol_type, hooks[mime_type])
        else:
            for formatter in formatters.values():
                for mol_type in (Mol, RWMol):
                    backup_key = (id(formatter), mol_type)
                    if backup_key not in self._ipython_formatter_backups:
                        continue
                    previous_printer = self._ipython_formatter_backups.pop(backup_key)
                    if previous_printer is None or previous_printer is _UNSET:
                        formatter.type_printers.pop(mol_type, None)
                    else:
                        formatter.type_printers[mol_type] = previous_printer


dofconfig = DofDrawSettings()
if dofconfig.enable_ipython:
    dofconfig.enable_ipython_integration(True)
