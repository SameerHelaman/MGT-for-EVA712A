"""RDKit atom and bond features shared by ligand graph models."""

from __future__ import annotations

# Provide fixed categorical chemistry enumerations.
from rdkit import Chem


# Encode every atomic number from hydrogen through oganesson plus one unknown bin.
ATOMIC_NUMBERS = list(range(1, 119))
# Encode common heavy-atom degrees plus an overflow bin.
ATOM_DEGREES = [0, 1, 2, 3, 4, 5]
# Encode common formal charges while preserving an overflow bin.
FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3]
# Encode RDKit hybridization states used by ordinary drug-like molecules.
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
# Encode attached-hydrogen counts plus an overflow bin.
TOTAL_HYDROGENS = [0, 1, 2, 3, 4]
# Encode specified tetrahedral chirality and an unspecified state.
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
# Encode the principal chemical bond orders plus an overflow bin.
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
# Encode common double-bond stereochemical states plus an overflow bin.
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOANY,
]


def one_hot_with_unknown(value, choices: list) -> list[float]:
    """Encode one categorical value with a final unknown/overflow position."""
    # Allocate one position for every known choice and one for unknown values.
    encoded = [0.0] * (len(choices) + 1)
    # Use the known category position when present.
    if value in choices:
        encoded[choices.index(value)] = 1.0
    # Use the final overflow position for uncommon or unsupported values.
    else:
        encoded[-1] = 1.0
    # Return the fixed-length floating-point vector.
    return encoded


def atom_features(atom: Chem.Atom) -> list[float]:
    """Create a fixed molecular-context feature vector for one RDKit atom."""
    # Encode elemental identity.
    features = one_hot_with_unknown(atom.GetAtomicNum(), ATOMIC_NUMBERS)
    # Encode the number of directly bonded neighbours.
    features += one_hot_with_unknown(atom.GetDegree(), ATOM_DEGREES)
    # Preserve the atom's standardized formal charge.
    features += one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGES)
    # Encode local orbital hybridization.
    features += one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATIONS)
    # Mark aromatic atoms explicitly.
    features += [float(atom.GetIsAromatic())]
    # Mark atoms belonging to at least one ring.
    features += [float(atom.IsInRing())]
    # Encode implicit and explicit attached hydrogen count.
    features += one_hot_with_unknown(atom.GetTotalNumHs(), TOTAL_HYDROGENS)
    # Preserve specified tetrahedral chirality.
    features += one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAGS)
    # Return the complete atom vector.
    return features


def bond_features(bond: Chem.Bond) -> list[float]:
    """Create a fixed chemical feature vector for one RDKit bond."""
    # Encode single, double, triple, aromatic, or uncommon bond type.
    features = one_hot_with_unknown(bond.GetBondType(), BOND_TYPES)
    # Mark conjugated bonds.
    features += [float(bond.GetIsConjugated())]
    # Mark bonds belonging to rings.
    features += [float(bond.IsInRing())]
    # Preserve double-bond stereochemistry where specified.
    features += one_hot_with_unknown(bond.GetStereo(), BOND_STEREO)
    # Return the complete bond vector.
    return features


# Calculate feature dimensions from the functions' categorical definitions.
ATOM_FEATURE_DIM = (
    len(ATOMIC_NUMBERS)
    + 1
    + len(ATOM_DEGREES)
    + 1
    + len(FORMAL_CHARGES)
    + 1
    + len(HYBRIDIZATIONS)
    + 1
    + 2
    + len(TOTAL_HYDROGENS)
    + 1
    + len(CHIRAL_TAGS)
    + 1
)
# Calculate the fixed chemical-bond feature dimension.
BOND_FEATURE_DIM = (
    len(BOND_TYPES)
    + 1
    + 2
    + len(BOND_STEREO)
    + 1
)
