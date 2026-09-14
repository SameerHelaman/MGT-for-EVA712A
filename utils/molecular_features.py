# =============================================================================
# MODULE: utils/molecular_features.py
# PURPOSE: Defines fixed RDKit atom and bond feature vocabularies and encoders.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Returns Python objects/tensors to its caller; this module does not directly define a persistent output artifact.
# CALCULATIONS: Categorical atom/bond properties are one-hot encoded with an explicit unknown bucket.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""RDKit atom and bond features shared by ligand graph models."""

from __future__ import annotations  # Enable postponed evaluation of type annotations.

# Provide fixed categorical chemistry enumerations.
from rdkit import Chem  # Import selected classes or functions from the named dependency.


# Encode every atomic number from hydrogen through oganesson plus one unknown bin.
ATOMIC_NUMBERS = list(range(1, 119))  # Bind this name to an intermediate value, configuration setting, or result.
# Encode common heavy-atom degrees plus an overflow bin.
ATOM_DEGREES = [0, 1, 2, 3, 4, 5]  # Bind this name to an intermediate value, configuration setting, or result.
# Encode common formal charges while preserving an overflow bin.
FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3]  # Bind this name to an intermediate value, configuration setting, or result.
# Encode RDKit hybridization states used by ordinary drug-like molecules.
HYBRIDIZATIONS = [  # Bind this name to an intermediate value, configuration setting, or result.
    Chem.rdchem.HybridizationType.SP,  # Perform this step of the surrounding calculation or control-flow block.
    Chem.rdchem.HybridizationType.SP2,  # Perform this step of the surrounding calculation or control-flow block.
    Chem.rdchem.HybridizationType.SP3,  # Perform this step of the surrounding calculation or control-flow block.
    Chem.rdchem.HybridizationType.SP3D,  # Perform this step of the surrounding calculation or control-flow block.
    Chem.rdchem.HybridizationType.SP3D2,  # Perform this step of the surrounding calculation or control-flow block.
]  # Continue or close the surrounding multiline expression or collection.
# Encode attached-hydrogen counts plus an overflow bin.
TOTAL_HYDROGENS = [0, 1, 2, 3, 4]  # Bind this name to an intermediate value, configuration setting, or result.
# Encode specified tetrahedral chirality and an unspecified state.
CHIRAL_TAGS = [  # Bind this name to an intermediate value, configuration setting, or result.
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,  # Perform this step of the surrounding calculation or control-flow block.
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,  # Perform this step of the surrounding calculation or control-flow block.
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,  # Perform this step of the surrounding calculation or control-flow block.
]  # Continue or close the surrounding multiline expression or collection.
# Encode the principal chemical bond orders plus an overflow bin.
BOND_TYPES = [  # Construct or transform graph topology, geometry, or molecular feature data.
    Chem.rdchem.BondType.SINGLE,  # Perform this step of the surrounding calculation or control-flow block.
    Chem.rdchem.BondType.DOUBLE,  # Perform this step of the surrounding calculation or control-flow block.
    Chem.rdchem.BondType.TRIPLE,  # Perform this step of the surrounding calculation or control-flow block.
    Chem.rdchem.BondType.AROMATIC,  # Perform this step of the surrounding calculation or control-flow block.
]  # Continue or close the surrounding multiline expression or collection.
# Encode common double-bond stereochemical states plus an overflow bin.
BOND_STEREO = [  # Construct or transform graph topology, geometry, or molecular feature data.
    Chem.rdchem.BondStereo.STEREONONE,  # Perform this step of the surrounding calculation or control-flow block.
    Chem.rdchem.BondStereo.STEREOZ,  # Perform this step of the surrounding calculation or control-flow block.
    Chem.rdchem.BondStereo.STEREOE,  # Perform this step of the surrounding calculation or control-flow block.
    Chem.rdchem.BondStereo.STEREOANY,  # Perform this step of the surrounding calculation or control-flow block.
]  # Continue or close the surrounding multiline expression or collection.


# FUNCTION: one_hot_with_unknown — see its docstring and inline comments.
def one_hot_with_unknown(value, choices: list) -> list[float]:  # Define this callable; its indented block implements the documented operation.
    """Encode one categorical value with a final unknown/overflow position."""
    # Allocate one position for every known choice and one for unknown values.
    encoded = [0.0] * (len(choices) + 1)  # Bind this name to an intermediate value, configuration setting, or result.
    # Use the known category position when present.
    if value in choices:  # Evaluate this condition before executing the associated branch.
        encoded[choices.index(value)] = 1.0  # Bind this name to an intermediate value, configuration setting, or result.
    # Use the final overflow position for uncommon or unsupported values.
    else:  # Handle the remaining case not covered by earlier conditions.
        encoded[-1] = 1.0  # Bind this name to an intermediate value, configuration setting, or result.
    # Return the fixed-length floating-point vector.
    return encoded  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: atom_features — see its docstring and inline comments.
def atom_features(atom: Chem.Atom) -> list[float]:  # Define this callable; its indented block implements the documented operation.
    """Create a fixed molecular-context feature vector for one RDKit atom."""
    # Encode elemental identity.
    features = one_hot_with_unknown(atom.GetAtomicNum(), ATOMIC_NUMBERS)  # Construct or transform graph topology, geometry, or molecular feature data.
    # Encode the number of directly bonded neighbours.
    features += one_hot_with_unknown(atom.GetDegree(), ATOM_DEGREES)  # Construct or transform graph topology, geometry, or molecular feature data.
    # Preserve the atom's standardized formal charge.
    features += one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGES)  # Construct or transform graph topology, geometry, or molecular feature data.
    # Encode local orbital hybridization.
    features += one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATIONS)  # Construct or transform graph topology, geometry, or molecular feature data.
    # Mark aromatic atoms explicitly.
    features += [float(atom.GetIsAromatic())]  # Construct or transform graph topology, geometry, or molecular feature data.
    # Mark atoms belonging to at least one ring.
    features += [float(atom.IsInRing())]  # Construct or transform graph topology, geometry, or molecular feature data.
    # Encode implicit and explicit attached hydrogen count.
    features += one_hot_with_unknown(atom.GetTotalNumHs(), TOTAL_HYDROGENS)  # Construct or transform graph topology, geometry, or molecular feature data.
    # Preserve specified tetrahedral chirality.
    features += one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAGS)  # Construct or transform graph topology, geometry, or molecular feature data.
    # Return the complete atom vector.
    return features  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: bond_features — see its docstring and inline comments.
def bond_features(bond: Chem.Bond) -> list[float]:  # Define this callable; its indented block implements the documented operation.
    """Create a fixed chemical feature vector for one RDKit bond."""
    # Encode single, double, triple, aromatic, or uncommon bond type.
    features = one_hot_with_unknown(bond.GetBondType(), BOND_TYPES)  # Construct or transform graph topology, geometry, or molecular feature data.
    # Mark conjugated bonds.
    features += [float(bond.GetIsConjugated())]  # Construct or transform graph topology, geometry, or molecular feature data.
    # Mark bonds belonging to rings.
    features += [float(bond.IsInRing())]  # Construct or transform graph topology, geometry, or molecular feature data.
    # Preserve double-bond stereochemistry where specified.
    features += one_hot_with_unknown(bond.GetStereo(), BOND_STEREO)  # Construct or transform graph topology, geometry, or molecular feature data.
    # Return the complete bond vector.
    return features  # Return this computed tensor, metric, object, or collection to the caller.


# Calculate feature dimensions from the functions' categorical definitions.
ATOM_FEATURE_DIM = (  # Construct or transform graph topology, geometry, or molecular feature data.
    len(ATOMIC_NUMBERS)  # Perform this step of the surrounding calculation or control-flow block.
    + 1  # Perform this step of the surrounding calculation or control-flow block.
    + len(ATOM_DEGREES)  # Perform this step of the surrounding calculation or control-flow block.
    + 1  # Perform this step of the surrounding calculation or control-flow block.
    + len(FORMAL_CHARGES)  # Perform this step of the surrounding calculation or control-flow block.
    + 1  # Perform this step of the surrounding calculation or control-flow block.
    + len(HYBRIDIZATIONS)  # Perform this step of the surrounding calculation or control-flow block.
    + 1  # Perform this step of the surrounding calculation or control-flow block.
    + 2  # Perform this step of the surrounding calculation or control-flow block.
    + len(TOTAL_HYDROGENS)  # Perform this step of the surrounding calculation or control-flow block.
    + 1  # Perform this step of the surrounding calculation or control-flow block.
    + len(CHIRAL_TAGS)  # Perform this step of the surrounding calculation or control-flow block.
    + 1  # Perform this step of the surrounding calculation or control-flow block.
)  # Continue or close the surrounding multiline expression or collection.
# Calculate the fixed chemical-bond feature dimension.
BOND_FEATURE_DIM = (  # Construct or transform graph topology, geometry, or molecular feature data.
    len(BOND_TYPES)  # Perform this step of the surrounding calculation or control-flow block.
    + 1  # Perform this step of the surrounding calculation or control-flow block.
    + 2  # Perform this step of the surrounding calculation or control-flow block.
    + len(BOND_STEREO)  # Perform this step of the surrounding calculation or control-flow block.
    + 1  # Perform this step of the surrounding calculation or control-flow block.
)  # Continue or close the surrounding multiline expression or collection.
