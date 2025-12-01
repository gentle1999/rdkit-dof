"""
Author: TMJ
Date: 2025-12-01 14:54:08
LastEditors: TMJ
LastEditTime: 2025-12-01 14:56:16
Description: Test the core functionality of MolToDofImage.
"""

import pytest
from PIL.Image import Image
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

from rdkit_dof.config import dofconfig
from rdkit_dof.core import MolToDofImage


@pytest.fixture(scope="module")
def sample_mol():
    """
    Provides a sample molecule (methane) with 3D coordinates.
    """
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    EmbedMolecule(mol, randomSeed=42)
    MMFFOptimizeMolecule(mol)
    return mol


def test_mol_to_dof_image_returns_image(sample_mol):
    """
    Tests that MolToDofImage runs without errors and returns a PIL Image.
    """
    # WHEN
    img = MolToDofImage(sample_mol, use_svg=False)

    # THEN
    assert img is not None
    assert isinstance(img, Image)


def test_use_style_switches_configuration(sample_mol):
    """
    Tests that use_style correctly updates the global config object.
    """
    # GIVEN
    # Reset to default state first
    dofconfig.use_style("default")
    default_fog = dofconfig.fog_color
    default_carbon_color = dofconfig.get_atom_color(6)

    # WHEN
    dofconfig.use_style("dark")

    # THEN
    assert dofconfig.preset_style == "dark"
    assert dofconfig.fog_color != default_fog
    assert dofconfig.fog_color == (0.1, 0.1, 0.1)
    assert dofconfig.get_atom_color(6) != default_carbon_color

    # Clean up by resetting to default for other tests
    dofconfig.use_style("default")
