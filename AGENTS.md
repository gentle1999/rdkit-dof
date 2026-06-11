# Repository Guidelines

## Project Structure & Module Organization

This is a Python package for RDKit molecule rendering with depth-of-field effects.
Core library code lives in `src/rdkit_dof/`: `core.py` contains drawing logic,
`config.py` defines dataclass settings, `palettes.py` stores theme colors, and
`__init__.py` exposes the public API. Tests are in `tests/` and are split by
single-molecule and grid drawing behavior. Generated showcase images live in
`assets/`; update them with `scripts/_generate_comparison_images.py` when visual
output changes. Packaging and tool configuration are in `pyproject.toml`.

## Build, Test, and Development Commands

- `uv sync --all-extras --dev`: install runtime and development dependencies.
- `uv run pytest`: run the full test suite.
- `uv run pytest tests/test_single_drawer.py`: run a focused test module.
- `uv run ruff check .`: lint imports, naming, pyflakes, bugbear, and upgrades.
- `uv run ruff format .`: format Python files using the configured Ruff style.
- `uv run mypy src`: run strict type checking for package code.
- `uv run pyright`: run Pyright over `src/` and `tests/`.
- `uv build`: build source and wheel distributions.
- `uv run python scripts/_generate_comparison_images.py`: regenerate README assets.

## Coding Style & Naming Conventions

Target Python 3.8+. Use 4-space indentation, double quotes, and an 88-character
line width. Ruff handles formatting and import ordering; first-party imports are
under `rdkit_dof`. Prefer typed functions and keep mypy strictness in mind.
Public drawing functions intentionally follow RDKit-style names such as
`MolToDofImage`, `MolsToGridDofImage`, `highlightAtoms`, and `highlightBonds`;
do not rename these API-compatible parameters just to satisfy snake_case rules.

## Testing Guidelines

Use pytest. Name test files `test_*.py`, fixtures with descriptive snake_case
names, and tests as `test_<behavior>`. Cover both SVG and PNG paths where
applicable, including raw data, image objects, file saving, empty inputs,
highlighting, and RDKit conformer edge cases. Prefer `tmp_path` for filesystem
tests and mocks only where direct RDKit drawing would make assertions brittle.

## Commit & Pull Request Guidelines

Recent history uses short, imperative messages with lightweight prefixes such as
`feat:`, `Refactor`, `update`, and `mod`. Prefer consistent conventional prefixes
when possible, for example `fix: handle empty conformer grids` or
`test: cover SVG file saving`. Pull requests should describe the behavior change,
list validation commands run, link related issues, and include updated images
when rendering output or README examples change.

## Configuration & Release Notes

Configuration can be supplied through `dofconfig` or `RDKIT_DOF_` environment
variables. Avoid committing local `.env` files, build outputs, or cache
directories. CI tests Python 3.8 through 3.14 on Linux, macOS, and Windows, and
publishes to PyPI only from GitHub releases.
