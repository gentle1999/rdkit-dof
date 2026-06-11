import pytest

from rdkit_dof.config import DofDrawSettings

ENV_KEYS = [
    "RDKIT_DOF_PRESET_STYLE",
    "RDKIT_DOF_FOG_COLOR",
    "RDKIT_DOF_MIN_ALPHA",
    "RDKIT_DOF_DEFAULT_SIZE",
    "RDKIT_DOF_ENABLE_IPYTHON",
    "RDKIT_DOF_ATOM_COLORS",
]


@pytest.fixture(autouse=True)
def clean_rdkit_dof_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_default_settings_load_default_style():
    settings = DofDrawSettings(env_file=None)

    assert settings.preset_style == "default"
    assert settings.fog_color == (0.95, 0.95, 0.95)
    assert settings.min_alpha == 0.4
    assert settings.default_size == (800, 800)
    assert settings.enable_ipython is True
    assert settings.get_atom_color(6) == (0.2, 0.2, 0.2)


def test_dark_style_updates_default_fog_color():
    settings = DofDrawSettings(preset_style="dark", env_file=None)

    assert settings.preset_style == "dark"
    assert settings.fog_color == (0.1, 0.1, 0.1)


def test_atom_colors_override_preset_colors():
    settings = DofDrawSettings(
        atom_colors={8: (1.0, 0.2, 0.2)},
        env_file=None,
    )

    assert settings.get_atom_color(8) == (1.0, 0.2, 0.2)
    assert settings.get_atom_color(6) == (0.2, 0.2, 0.2)


def test_environment_variables_are_loaded(monkeypatch):
    monkeypatch.setenv("RDKIT_DOF_PRESET_STYLE", "nature")
    monkeypatch.setenv("RDKIT_DOF_FOG_COLOR", "[0.1, 0.2, 0.3]")
    monkeypatch.setenv("RDKIT_DOF_MIN_ALPHA", "0.25")
    monkeypatch.setenv("RDKIT_DOF_DEFAULT_SIZE", "[500, 400]")
    monkeypatch.setenv("RDKIT_DOF_ENABLE_IPYTHON", "false")
    monkeypatch.setenv("RDKIT_DOF_ATOM_COLORS", '{"8": [0.9, 0.1, 0.1]}')

    settings = DofDrawSettings(env_file=None)

    assert settings.preset_style == "nature"
    assert settings.fog_color == (0.1, 0.2, 0.3)
    assert settings.min_alpha == 0.25
    assert settings.default_size == (500, 400)
    assert settings.enable_ipython is False
    assert settings.get_atom_color(8) == (0.9, 0.1, 0.1)


def test_dotenv_file_is_loaded(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "RDKIT_DOF_PRESET_STYLE=jacs",
                "RDKIT_DOF_MIN_ALPHA=0.3",
                "RDKIT_DOF_DEFAULT_SIZE=(600, 500)",
            ]
        ),
        encoding="utf-8",
    )

    settings = DofDrawSettings(env_file=env_file)

    assert settings.preset_style == "jacs"
    assert settings.min_alpha == 0.3
    assert settings.default_size == (600, 500)


def test_explicit_arguments_override_environment(monkeypatch):
    monkeypatch.setenv("RDKIT_DOF_MIN_ALPHA", "0.2")
    monkeypatch.setenv("RDKIT_DOF_ENABLE_IPYTHON", "false")

    settings = DofDrawSettings(
        min_alpha=0.8,
        enable_ipython=True,
        env_file=None,
    )

    assert settings.min_alpha == 0.8
    assert settings.enable_ipython is True


def test_extra_arguments_are_ignored():
    settings = DofDrawSettings(unknown_option=True, env_file=None)

    assert settings.preset_style == "default"


def test_invalid_style_raises_value_error():
    with pytest.raises(ValueError, match="preset_style"):
        DofDrawSettings(preset_style="invalid", env_file=None)


def test_invalid_color_raises_value_error():
    with pytest.raises(ValueError, match="fog_color"):
        DofDrawSettings(fog_color=(0.1, 0.2), env_file=None)
