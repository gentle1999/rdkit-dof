"""
Author: TMJ
Date: 2025-12-01 15:22:40
LastEditors: TMJ
LastEditTime: 2025-12-01 16:17:11
Description: Generates comparison images for the README file.
- Default RDKit vs. rdkit-dof for a single molecule.
- Default RDKit vs. rdkit-dof for a grid of molecules.
"""

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

from rdkit_dof import MolGridToDofImage, MolToDofImage, dofconfig


def generate_single_mol_comparison():
    """Generates comparison for a single complex molecule."""
    print("Generating single molecule comparison...")
    smiles = "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)C)C"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    EmbedMolecule(mol, randomSeed=42)
    MMFFOptimizeMolecule(mol)

    img_size = (600, 400)
    legend = "Paclitaxel"

    # 1. Default RDKit drawing
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(img_size[0], img_size[1])
    drawer.DrawMolecule(mol, legend=legend)
    drawer.FinishDrawing()
    with open("assets/comparison_single_default.svg", "w") as f:
        f.write(drawer.GetDrawingText())
    print("  - Saved assets/comparison_single_default.svg")

    # 2. rdkit-dof drawing
    dofconfig.use_style("default")
    print(dofconfig.model_dump())
    dof_img = MolToDofImage(
        mol, size=img_size, legend=legend, use_svg=True, return_image=False
    )
    with open("assets/comparison_single_dof.svg", "w") as f:
        f.write(dof_img)
    print("  - Saved assets/comparison_single_dof.svg")


def generate_grid_comparison():
    """Generates comparison for a grid of molecules."""
    print("Generating grid comparison...")
    smiles_list = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)NC1=CC=C(O)C=C1",  # Paracetamol
        "CCO",  # Ethanol
        "OCC1OC(CCC2=CNC3=CC=CC=C32)C(O)C(O)C1O",  # Serotonin
        "C(C(C(C(C(=O)CO)O)O)O)O",  # Glucose (open-chain)
        "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",  # Glucose (closed-chain)
    ]
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    legends = [
        "Aspirin",
        "Ibuprofen",
        "Caffeine",
        "Paracetamol",
        "Ethanol",
        "Serotonin",
        "Glucose (open-chain)",
        "Glucose (closed-chain)",
    ]

    # Generate 3D conformer for each
    mols_with_conformer = []
    for mol in mols:
        mol = Chem.AddHs(mol)
        EmbedMolecule(mol, randomSeed=42)
        MMFFOptimizeMolecule(mol)
        mols_with_conformer.append(mol)

    img_size = (300, 300)
    mols_per_row = 2  # Adjusted for better layout

    # 1. Default RDKit grid
    grid_img = Draw.MolsToGridImage(
        mols_with_conformer,
        molsPerRow=mols_per_row,
        subImgSize=img_size,
        legends=legends,
        useSVG=True,
    )
    with open("assets/comparison_grid_default.svg", "w") as f:
        f.write(grid_img.data)
    print("  - Saved assets/comparison_grid_default.svg")

    # 2. rdkit-dof grid
    dofconfig.use_style("default")
    print(dofconfig.model_dump())
    dof_grid_img = MolGridToDofImage(
        mols_with_conformer,
        molsPerRow=mols_per_row,
        subImgSize=img_size,
        legends=legends,
        use_svg=True,
        return_image=False,
    )
    with open("assets/comparison_grid_dof.svg", "w") as f:
        f.write(dof_grid_img)
    print("  - Saved assets/comparison_grid_dof.svg")


if __name__ == "__main__":
    generate_single_mol_comparison()
    print("-" * 20)
    generate_grid_comparison()
    print("\nAll comparison images generated successfully.")
