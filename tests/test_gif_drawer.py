import pytest
from PIL import Image, ImageSequence
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule

from rdkit_dof import MolsToDofGif, MolsToDofSvgAnimation


@pytest.fixture
def molecules_3d():
    mols = []
    for smiles in ["CCO", "CCN", "COC"]:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        mol = Chem.AddHs(mol)
        EmbedMolecule(mol, randomSeed=42)
        mols.append(mol)
    return mols


def test_mols_to_dof_gif_returns_image(molecules_3d):
    gif = MolsToDofGif(molecules_3d, size=(240, 200), return_image=True)

    assert isinstance(gif, Image.Image)
    assert gif.format == "GIF"
    assert getattr(gif, "is_animated", False)
    assert getattr(gif, "n_frames", 1) == len(molecules_3d)


def test_mols_to_dof_gif_returns_bytes(molecules_3d):
    gif_data = MolsToDofGif(molecules_3d, size=(240, 200), return_image=False)

    assert isinstance(gif_data, bytes)
    assert gif_data.startswith(b"GIF8")


def test_mols_to_dof_gif_saves_file(molecules_3d, tmp_path):
    output_file = tmp_path / "molecules.gif"

    MolsToDofGif(molecules_3d, size=(240, 200), filename=str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0
    with Image.open(output_file) as gif:
        assert gif.format == "GIF"
        assert getattr(gif, "n_frames", 1) == len(molecules_3d)


def test_mols_to_dof_gif_accepts_frame_options(molecules_3d):
    gif = MolsToDofGif(
        molecules_3d,
        size=(240, 200),
        legends=["Ethanol", "Ethylamine", "Dimethyl ether"],
        duration=[100, 150, 200],
        highlightAtomLists=[[0], [1], [2]],
        highlightBondLists=[[0], [0], [0]],
        highlightColor=(0.0, 1.0, 0.0, 0.5),
        return_image=True,
    )

    durations = [frame.info["duration"] for frame in ImageSequence.Iterator(gif)]
    assert durations == [100, 150, 200]


def test_mols_to_dof_gif_rejects_empty_input():
    with pytest.raises(ValueError, match="mols must contain at least one molecule"):
        MolsToDofGif([])


def test_mols_to_dof_gif_rejects_mismatched_legends(molecules_3d):
    with pytest.raises(ValueError, match="legends must have the same length as mols"):
        MolsToDofGif(molecules_3d, legends=["one"])


def test_mols_to_dof_gif_warns_once_for_unicode_legends(molecules_3d):
    with pytest.warns(UserWarning, match="Unicode/non-ASCII legend text") as record:
        MolsToDofGif(
            molecules_3d,
            legends=["乙醇", "乙胺", "二甲醚"],
            return_image=False,
        )

    assert len(record) == 1


def test_mols_to_dof_gif_rejects_mismatched_duration(molecules_3d):
    with pytest.raises(ValueError, match="duration sequence"):
        MolsToDofGif(molecules_3d, duration=[100, 200])


def test_mols_to_dof_svg_animation_returns_text(molecules_3d):
    svg_text = MolsToDofSvgAnimation(
        molecules_3d,
        size=(240, 200),
        duration=[100, 150, 200],
        return_image=False,
    )

    assert isinstance(svg_text, str)
    assert "<svg" in svg_text
    assert "<animate" in svg_text
    assert svg_text.count("<g opacity=") == len(molecules_3d)
    assert 'dur="0.45s"' in svg_text
    assert 'repeatCount="indefinite"' in svg_text


def test_mols_to_dof_svg_animation_saves_file(molecules_3d, tmp_path):
    output_file = tmp_path / "molecules.svg"

    MolsToDofSvgAnimation(
        molecules_3d,
        size=(240, 200),
        filename=str(output_file),
        return_image=False,
    )

    assert output_file.exists()
    content = output_file.read_text()
    assert "<svg" in content
    assert "<animate" in content


def test_mols_to_dof_svg_animation_single_frame_has_no_animate(molecules_3d):
    svg_text = MolsToDofSvgAnimation(
        [molecules_3d[0]],
        size=(240, 200),
        return_image=False,
    )

    assert "<svg" in svg_text
    assert "<animate" not in svg_text
    assert svg_text.count("<g opacity=") == 1


def test_mols_to_dof_svg_animation_accepts_frame_options(molecules_3d):
    svg_text = MolsToDofSvgAnimation(
        molecules_3d,
        size=(240, 200),
        legends=["Ethanol", "Ethylamine", "Dimethyl ether"],
        duration=250,
        loop=2,
        highlightAtomLists=[[0], [1], [2]],
        highlightBondLists=[[0], [0], [0]],
        highlightColor=(0.0, 1.0, 0.0, 0.5),
        return_image=False,
    )

    assert 'repeatCount="2"' in svg_text
    assert 'dur="0.75s"' in svg_text


def test_mols_to_dof_svg_animation_rejects_empty_input():
    with pytest.raises(ValueError, match="mols must contain at least one molecule"):
        MolsToDofSvgAnimation([])


def test_mols_to_dof_svg_animation_warns_once_for_unicode_legends(molecules_3d):
    with pytest.warns(UserWarning, match="Unicode/non-ASCII legend text") as record:
        MolsToDofSvgAnimation(
            molecules_3d,
            legends=["乙醇", "乙胺", "二甲醚"],
            return_image=False,
        )

    assert len(record) == 1


def test_mols_to_dof_svg_animation_rejects_mismatched_duration(molecules_3d):
    with pytest.raises(ValueError, match="duration sequence"):
        MolsToDofSvgAnimation(molecules_3d, duration=[100, 200])
