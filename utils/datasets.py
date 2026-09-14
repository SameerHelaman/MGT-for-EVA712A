# =============================================================================
# MODULE: utils/datasets.py
# PURPOSE: Original structure loader and local, line and wider graph construction with OpenBind nonperiodic compatibility.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Returns Python objects/tensors to its caller; this module does not directly define a persistent output artifact.
# CALCULATIONS: Distances derive from Cartesian coordinates; line-graph features are bond-angle cosines; Laplacian eigenvectors provide deterministic positional encodings; wider edges encode longer-range interactions.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Original MGT structure dataset with OpenBind non-periodic compatibility."""

import csv  # Load a standard-library, scientific, or local project dependency.
import glob  # Load a standard-library, scientific, or local project dependency.
import hashlib  # Load a standard-library, scientific, or local project dependency.
import json  # Load a standard-library, scientific, or local project dependency.
import random  # Load a standard-library, scientific, or local project dependency.
import warnings  # Load a standard-library, scientific, or local project dependency.
from typing import List, Tuple  # Import selected classes or functions from the named dependency.

import numpy as np  # Load a standard-library, scientific, or local project dependency.
import os.path as osp  # Load a standard-library, scientific, or local project dependency.

import dgl  # Load a standard-library, scientific, or local project dependency.
import torch.utils.data  # Load a standard-library, scientific, or local project dependency.
from dgl import load_graphs  # Import selected classes or functions from the named dependency.
from pymatgen.core import Structure, Molecule  # Import selected classes or functions from the named dependency.
from rdkit import Chem  # Import selected classes or functions from the named dependency.


# CLASS: AtomInitializer — reusable model/data abstraction.
class AtomInitializer(object):  # Define this reusable class and its inheritance contract.
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):  # Define this callable; its indented block implements the documented operation.
        """Initialize this object and its required state."""
        self.atom_types = set(atom_types)  # Store this configuration value or neural-network submodule on the instance.
        self._embedding = {}  # Store this configuration value or neural-network submodule on the instance.

    def get_atom_fea(self, atom_type):  # Define this callable; its indented block implements the documented operation.
        """Return the stored feature vector for one atomic number."""
        assert atom_type in self.atom_types  # Enforce an invariant required by the following calculation.
        return self._embedding[atom_type]  # Return this computed tensor, metric, object, or collection to the caller.

    def load_state_dict(self, state_dict):  # Define this callable; its indented block implements the documented operation.
        """Load atom embeddings and rebuild decoding metadata."""
        self._embedding = state_dict  # Store this configuration value or neural-network submodule on the instance.
        self.atom_types = set(self._embedding.keys())  # Store this configuration value or neural-network submodule on the instance.
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}  # Store this configuration value or neural-network submodule on the instance.

    def state_dict(self):  # Define this callable; its indented block implements the documented operation.
        """Return the atom-embedding mapping for serialization."""
        return self._embedding  # Return this computed tensor, metric, object, or collection to the caller.

    def decode(self, idx):  # Define this callable; its indented block implements the documented operation.
        """Map an encoded index back to its atom type."""
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in  # Store this configuration value or neural-network submodule on the instance.
                                self._embedding.items()}  # Perform this step of the surrounding calculation or control-flow block.
        return self._decodedict[idx]  # Return this computed tensor, metric, object, or collection to the caller.


# CLASS: AtomCustomJSONInitializer — reusable model/data abstraction.
class AtomCustomJSONInitializer(AtomInitializer):  # Define this reusable class and its inheritance contract.
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):  # Define this callable; its indented block implements the documented operation.
        """Initialize this object and its required state."""
        with open(elem_embedding_file) as f:  # Enter a managed context so resources and graph state are cleaned up safely.
            elem_embedding = json.load(f)  # Create or apply a trainable neural-network component.
        elem_embedding = {int(float(key)): value for key, value in elem_embedding.items()}  # Create or apply a trainable neural-network component.
        atom_types = set(elem_embedding.keys())  # Create or apply a trainable neural-network component.

        super(AtomCustomJSONInitializer, self).__init__(atom_types)  # Initialize or delegate to the parent class implementation.
        for key, value in elem_embedding.items():  # Iterate over the stated records, layers, batches, or graph elements.
            self._embedding[key] = np.array(value, dtype=float)  # Store this configuration value or neural-network submodule on the instance.


# FUNCTION: load_nonperiodic_molecule — see its docstring and inline comments.
def load_nonperiodic_molecule(structure_path):  # Define this callable; its indented block implements the documented operation.
    """Load a molecule, with an RDKit fallback for OpenBind ligand PDBs."""
    try:  # Begin protected execution for an operation that may raise an exception.
        return Molecule.from_file(structure_path)  # Return this computed tensor, metric, object, or collection to the caller.
    except (AttributeError, OSError, ValueError):  # Handle the specified failure without losing diagnostic context.
        if not str(structure_path).lower().endswith(".pdb"):
            raise  # Perform this step of the surrounding calculation or control-flow block.
        rdkit_molecule = Chem.MolFromPDBFile(  # Bind this name to an intermediate value, configuration setting, or result.
            str(structure_path), sanitize=True, removeHs=False  # Bind this name to an intermediate value, configuration setting, or result.
        )  # Continue or close the surrounding multiline expression or collection.
        if rdkit_molecule is None:  # Evaluate this condition before executing the associated branch.
            raise ValueError(f"RDKit could not parse PDB file: {structure_path}")  # Reject invalid input or state with an explicit exception.
        if rdkit_molecule.GetNumConformers() != 1:  # Evaluate this condition before executing the associated branch.
            raise ValueError(  # Reject invalid input or state with an explicit exception.
                f"PDB must contain exactly one conformer: {structure_path}"  # Perform this step of the surrounding calculation or control-flow block.
            )  # Continue or close the surrounding multiline expression or collection.
        conformer = rdkit_molecule.GetConformer()  # Bind this name to an intermediate value, configuration setting, or result.
        species = [atom.GetSymbol() for atom in rdkit_molecule.GetAtoms()]  # Bind this name to an intermediate value, configuration setting, or result.
        coordinates = [  # Bind this name to an intermediate value, configuration setting, or result.
            list(conformer.GetAtomPosition(index))  # Perform this step of the surrounding calculation or control-flow block.
            for index in range(rdkit_molecule.GetNumAtoms())  # Iterate over the stated records, layers, batches, or graph elements.
        ]  # Continue or close the surrounding multiline expression or collection.
        return Molecule(species, coordinates)  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: deterministic_laplacian_pe — see its docstring and inline comments.
def deterministic_laplacian_pe(graph, dimension, file_id, seed):  # Define this callable; its indented block implements the documented operation.
    """Run DGL LapPE with a stable per-structure eigenvector sign choice."""
    seed_material = f"{seed}:{file_id}".encode("utf-8")
    local_seed = int.from_bytes(  # Bind this name to an intermediate value, configuration setting, or result.
        hashlib.sha256(seed_material).digest()[:4], byteorder="big"
    )  # Continue or close the surrounding multiline expression or collection.
    numpy_state = np.random.get_state()  # Bind this name to an intermediate value, configuration setting, or result.
    try:  # Begin protected execution for an operation that may raise an exception.
        np.random.seed(local_seed)  # Invoke the relevant tensor, graph, numerical, or tabular operation.
        return dgl.lap_pe(graph, dimension, padding=True)  # Return this computed tensor, metric, object, or collection to the caller.
    finally:  # Run cleanup regardless of whether the protected operation succeeded.
        np.random.set_state(numpy_state)  # Invoke the relevant tensor, graph, numerical, or tabular operation.


# FUNCTION: compute_bond_cosines — see its docstring and inline comments.
def compute_bond_cosines(edges):  # Define this callable; its indented block implements the documented operation.
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (  # Compute or store a loss, error, residual, or regression evaluation statistic.
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)  # Compute or store a loss, error, residual, or regression evaluation statistic.
    )  # Continue or close the surrounding multiline expression or collection.
    bond_cosine = torch.clamp(bond_cosine, -1, 1)  # Construct or transform graph topology, geometry, or molecular feature data.
    return {"angle_feats": bond_cosine.unsqueeze(1)}


# CLASS: StructureDataset — reusable model/data abstraction.
class StructureDataset(torch.utils.data.Dataset):  # Define this reusable class and its inheritance contract.
    ''' Dataset for Molecular Graph Representations '''

    def __init__(self, args, process: bool = False, random_seed: int = 123, transform=None):  # Define this callable; its indented block implements the documented operation.
        
        """Initialize this object and its required state."""
        self.root = args.root  # Store this configuration value or neural-network submodule on the instance.
        self.random_seed = random_seed  # Store this configuration value or neural-network submodule on the instance.
        self.transform = transform  # Store this configuration value or neural-network submodule on the instance.

        self.process = process  # Store this configuration value or neural-network submodule on the instance.
        if self.process:  # Evaluate this condition before executing the associated branch.
            self.raw_dir = osp.join(self.root, 'raw')
            self.max_nei_num = args.max_nei_num  # Store this configuration value or neural-network submodule on the instance.
            self.pe_dim = args.num_pe_fea  # Store this configuration value or neural-network submodule on the instance.
            self.radius = args.local_radius  # Store this configuration value or neural-network submodule on the instance.
            self.random_seed = random_seed  # Store this configuration value or neural-network submodule on the instance.
            self.periodic = args.periodic  # Store this configuration value or neural-network submodule on the instance.
            if args.periodic:  # Evaluate this condition before executing the associated branch.
                self.periodic_radius = args.periodic_radius  # Store this configuration value or neural-network submodule on the instance.

            atom_init_file = osp.join(self.root, 'atom_init.json')
            assert osp.exists(atom_init_file), 'atom_init.json file does not exist!'
            self.cai = AtomCustomJSONInitializer(atom_init_file)  # Store this configuration value or neural-network submodule on the instance.
        else:  # Handle the remaining case not covered by earlier conditions.
            self.proc_dir = osp.join(self.root, 'processed')

        id_prop_file = osp.join(self.root, 'id_prop.csv')
        assert osp.exists(id_prop_file), 'id_prop.csv file does not exist'
        with open(id_prop_file) as f:  # Enter a managed context so resources and graph state are cleaned up safely.
            reader = csv.reader(f)  # Bind this name to an intermediate value, configuration setting, or result.
            self.id_prop_data = [row for row in reader]  # Store this configuration value or neural-network submodule on the instance.

    def __len__(self):  # Define this callable; its indented block implements the documented operation.
        """Return the number of dataset records."""
        return len(self.id_prop_data)  # Return this computed tensor, metric, object, or collection to the caller.

    def shuffle(self):  # Define this callable; its indented block implements the documented operation.
        """Shuffle records reproducibly with the dataset seed."""
        random.seed(self.random_seed)  # Perform this step of the surrounding calculation or control-flow block.
        random.shuffle(self.id_prop_data)  # Perform this step of the surrounding calculation or control-flow block.
        return  # Return control to the caller without an explicit value.

    def __getitem__(self, idx):  # Define this callable; its indented block implements the documented operation.
        """Load and return one indexed dataset record."""
        cif_id = self.id_prop_data[idx][0]  # Bind this name to an intermediate value, configuration setting, or result.

        if self.process:  # Evaluate this condition before executing the associated branch.
            g, lg, fg = self._construct_graph(cif_id)  # Construct or transform graph topology, geometry, or molecular feature data.
        else:  # Handle the remaining case not covered by earlier conditions.
            g, lg, fg = load_graphs(osp.join(self.proc_dir, f'{cif_id}.bin'))[0]  # Construct or transform graph topology, geometry, or molecular feature data.

        if self.transform:  # Evaluate this condition before executing the associated branch.
            g = self.transform(g)  # Bind this name to an intermediate value, configuration setting, or result.

        props = [float(x) for x in self.id_prop_data[idx][1:]]  # Bind this name to an intermediate value, configuration setting, or result.
        if props:  # Evaluate this condition before executing the associated branch.
            props = np.array(props)  # Bind this name to an intermediate value, configuration setting, or result.
            return g, lg, fg, torch.tensor(props, dtype=torch.float32), cif_id  # Return this computed tensor, metric, object, or collection to the caller.
        else:  # Handle the remaining case not covered by earlier conditions.
            return g, lg, fg, cif_id  # Return this computed tensor, metric, object, or collection to the caller.

    def _construct_graph(self, file_id):  # Define this callable; its indented block implements the documented operation.
        # Check if file exists and there are no duplicates
        """Parse one structure and build local, line and Coulomb DGL graphs."""
        if osp.exists(osp.join(self.raw_dir, file_id)):  # Evaluate this condition before executing the associated branch.
            structure_path = osp.join(self.raw_dir, file_id)  # Bind this name to an intermediate value, configuration setting, or result.
        elif glob.glob(osp.join(self.raw_dir, f'{file_id}.*')):  # Test this additional condition when earlier branches were not selected.
            files = glob.glob(osp.join(self.raw_dir, f'{file_id}.*'))  # Bind this name to an intermediate value, configuration setting, or result.
            if len(files) > 1:  # Evaluate this condition before executing the associated branch.
                warnings.warn(f'More than one file with the name {file_id} exists in the raw directory')  # Perform this step of the surrounding calculation or control-flow block.
                exit()  # Perform this step of the surrounding calculation or control-flow block.
            elif len(files) == 0:  # Test this additional condition when earlier branches were not selected.
                warnings.warn(f'No file with the name {file_id} exists in the raw directory')  # Perform this step of the surrounding calculation or control-flow block.
                exit()  # Perform this step of the surrounding calculation or control-flow block.
            else:  # Handle the remaining case not covered by earlier conditions.
                structure_path = files[0]  # Bind this name to an intermediate value, configuration setting, or result.
        else:  # Handle the remaining case not covered by earlier conditions.
            warnings.warn(f'No file with the name {file_id} exists in the raw directory')  # Perform this step of the surrounding calculation or control-flow block.
            exit()  # Perform this step of the surrounding calculation or control-flow block.

        # Load structure and transform into molecule if needed
        structure = (  # Bind this name to an intermediate value, configuration setting, or result.
            Structure.from_file(structure_path)  # Perform this step of the surrounding calculation or control-flow block.
            if self.periodic  # Evaluate this condition before executing the associated branch.
            else load_nonperiodic_molecule(structure_path)  # Handle the remaining case not covered by earlier conditions.
        )  # Continue or close the surrounding multiline expression or collection.

        # Get atom features
        atom_fea = np.vstack([self.cai.get_atom_fea(structure[i].specie.number) for i in range(len(structure))])  # Bind this name to an intermediate value, configuration setting, or result.
        atom_fea = torch.Tensor(atom_fea)  # Bind this name to an intermediate value, configuration setting, or result.

        nbr_idx, nbr_fea, nbr_disp, fc_idx, fc_coulomb = [], [], [], [], []  # Bind this name to an intermediate value, configuration setting, or result.

        # Get edges
        for idx, atm in enumerate(structure):  # Iterate over the stated records, layers, batches, or graph elements.

            # Get Neighbours for the atom
            if self.periodic:  # Evaluate this condition before executing the associated branch.
                nbr = sorted(structure.get_neighbors(atm, r=self.radius, include_index=True), key=lambda x: x[1])  # Bind this name to an intermediate value, configuration setting, or result.

                a, b, c = structure.lattice.abc  # Bind this name to an intermediate value, configuration setting, or result.
                diag = (a ** 2 + b ** 2 + c ** 2) ** 0.5  # Bind this name to an intermediate value, configuration setting, or result.

                if diag > self.periodic_radius:  # Evaluate this condition before executing the associated branch.
                    diag = self.periodic_radius  # Bind this name to an intermediate value, configuration setting, or result.

                full_nbrs = structure.get_neighbors(atm, diag, include_index=True)  # Bind this name to an intermediate value, configuration setting, or result.
            else:  # Handle the remaining case not covered by earlier conditions.
                nbr = sorted(structure.get_neighbors(atm, r=self.radius), key=lambda x: x[1])  # Bind this name to an intermediate value, configuration setting, or result.
                full_nbrs = structure.get_neighbors(atm, r=np.inf)  # Bind this name to an intermediate value, configuration setting, or result.

            # Compile local and global edges
            if len(nbr) < 12:  # Evaluate this condition before executing the associated branch.
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(file_id))
                nbr_idx.extend(list(map(lambda x: (idx, x[2]), nbr)))  # Perform this step of the surrounding calculation or control-flow block.
                nbr_fea.extend(list(map(lambda x: x[1], nbr)))  # Perform this step of the surrounding calculation or control-flow block.
                nbr_disp.extend(list(map(lambda x: x.coords - atm.coords, nbr)))  # Perform this step of the surrounding calculation or control-flow block.

                fc_idx.extend(list(map(lambda x: (idx, x[2]), full_nbrs)))  # Perform this step of the surrounding calculation or control-flow block.
                distances = np.array(list(map(lambda x: x[1], full_nbrs)))  # Construct or transform graph topology, geometry, or molecular feature data.
                charges = np.array(list(map(lambda x: x.specie.Z * atm.specie.Z, full_nbrs)))  # Bind this name to an intermediate value, configuration setting, or result.
                fc_coulomb.extend(charges / distances)  # Perform this step of the surrounding calculation or control-flow block.
            else:  # Handle the remaining case not covered by earlier conditions.
                nbr_idx.extend(list(map(lambda x: (idx, x[2]), nbr[:12])))  # Perform this step of the surrounding calculation or control-flow block.
                nbr_fea.extend(list(map(lambda x: x[1], nbr[:12])))  # Perform this step of the surrounding calculation or control-flow block.
                nbr_disp.extend(list(map(lambda x: x.coords - atm.coords, nbr[:12])))  # Perform this step of the surrounding calculation or control-flow block.

                fc_idx.extend(list(map(lambda x: (idx, x[2]), full_nbrs)))  # Perform this step of the surrounding calculation or control-flow block.
                distances = np.array(list(map(lambda x: x[1], full_nbrs)))  # Construct or transform graph topology, geometry, or molecular feature data.
                charges = np.array(list(map(lambda x: x.specie.Z * atm.specie.Z, full_nbrs)))  # Bind this name to an intermediate value, configuration setting, or result.
                fc_coulomb.extend(charges / distances)  # Perform this step of the surrounding calculation or control-flow block.

        edge_idx, edge_fea, fc_index, fc_fea, edge_disp = torch.LongTensor(np.array(nbr_idx)).t().contiguous(), \
                                                          torch.tensor(np.array(nbr_fea), dtype=torch.float32).unsqueeze(dim=1), \
                                                          torch.LongTensor(np.array(fc_idx)).t().contiguous(), \
                                                          torch.tensor(np.array(fc_coulomb), dtype=torch.float32).unsqueeze(dim=1), \
                                                          torch.tensor(np.array(nbr_disp), dtype=torch.float32)  # Bind this name to an intermediate value, configuration setting, or result.

        # Construct Local Graph
        G = dgl.graph(data=(edge_idx[0], edge_idx[1]), num_nodes=atom_fea.shape[0])  # Construct or transform graph topology, geometry, or molecular feature data.
        G.ndata['node_feats'] = atom_fea
        G.edata['edge_feats'] = edge_fea
        G.edata['r'] = edge_disp
        # Get Positional Encodings
        G.ndata['pes'] = deterministic_laplacian_pe(
            G, self.pe_dim, file_id, self.random_seed  # Perform this step of the surrounding calculation or control-flow block.
        )  # Continue or close the surrounding multiline expression or collection.

        # Construct Full Graph
        FG = dgl.graph(data=(fc_index[0], fc_index[1]), num_nodes=atom_fea.shape[0])  # Construct or transform graph topology, geometry, or molecular feature data.
        FG.edata['fc_feats'] = fc_fea

        # Construct Line Graph
        LG = G.line_graph(shared=True)  # Construct or transform graph topology, geometry, or molecular feature data.
        LG.apply_edges(compute_bond_cosines)  # Perform this step of the surrounding calculation or control-flow block.

        return G, LG, FG  # Return this computed tensor, metric, object, or collection to the caller.

    @staticmethod  # Apply this decorator to the following class or function.
    def collate_run(samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph, str]]):  # Define this callable; its indented block implements the documented operation.
        """Batch unlabeled graph triplets for inference."""
        graphs, line_graphs, full_graphs, labels, ids = map(list, zip(*samples))  # Construct or transform graph topology, geometry, or molecular feature data.
        batched_graph = dgl.batch(graphs)  # Construct or transform graph topology, geometry, or molecular feature data.
        batched_line_graph = dgl.batch(line_graphs)  # Construct or transform graph topology, geometry, or molecular feature data.
        batched_full_graph = dgl.batch(full_graphs)  # Construct or transform graph topology, geometry, or molecular feature data.
        if len(labels[0].size()) > 0:  # Evaluate this condition before executing the associated branch.
            return batched_graph, batched_line_graph, batched_full_graph, ids  # Return this computed tensor, metric, object, or collection to the caller.
        else:  # Handle the remaining case not covered by earlier conditions.
            return batched_graph, batched_line_graph, batched_full_graph, ids  # Return this computed tensor, metric, object, or collection to the caller.

    @staticmethod  # Apply this decorator to the following class or function.
    def collate_tt(samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph, torch.Tensor, str]]):  # Define this callable; its indented block implements the documented operation.
        """Batch labeled graph triplets, labels and identifiers."""
        graphs, line_graphs, full_graphs, labels, ids = map(list, zip(*samples))  # Construct or transform graph topology, geometry, or molecular feature data.
        batched_graph = dgl.batch(graphs)  # Construct or transform graph topology, geometry, or molecular feature data.
        batched_line_graph = dgl.batch(line_graphs)  # Construct or transform graph topology, geometry, or molecular feature data.
        batched_full_graph = dgl.batch(full_graphs)  # Construct or transform graph topology, geometry, or molecular feature data.
        if len(labels[0].size()) > 0:  # Evaluate this condition before executing the associated branch.
            return batched_graph, batched_line_graph, batched_full_graph, torch.stack(labels), ids  # Return this computed tensor, metric, object, or collection to the caller.
        else:  # Handle the remaining case not covered by earlier conditions.
            return batched_graph, batched_line_graph, batched_full_graph, torch.tensor(labels), ids  # Return this computed tensor, metric, object, or collection to the caller.

    @staticmethod  # Apply this decorator to the following class or function.
    def collate_pre(samples: List[Tuple[Tuple, dgl.DGLGraph, dgl.DGLGraph, str]]):  # Define this callable; its indented block implements the documented operation.
        """Batch masked graphs and retain global masked-node indices."""
        graphs, line_graphs, full_graphs, ids = map(list, zip(*samples))  # Construct or transform graph topology, geometry, or molecular feature data.
        graphs, nodes_sub = map(list, zip(*graphs))  # Construct or transform graph topology, geometry, or molecular feature data.
        cum_n = 0  # Bind this name to an intermediate value, configuration setting, or result.
        for i, g in enumerate(graphs):  # Iterate over the stated records, layers, batches, or graph elements.
            nodes_sub[i].ndata[dgl.NID] = nodes_sub[i].ndata[dgl.NID] + cum_n  # Construct or transform graph topology, geometry, or molecular feature data.
            cum_n += g.num_nodes()  # Construct or transform graph topology, geometry, or molecular feature data.
        batched_graph = dgl.batch(graphs)  # Construct or transform graph topology, geometry, or molecular feature data.
        batched_nodes = dgl.batch(nodes_sub)  # Construct or transform graph topology, geometry, or molecular feature data.
        batched_line_graph = dgl.batch(line_graphs)  # Construct or transform graph topology, geometry, or molecular feature data.
        batched_full_graph = dgl.batch(full_graphs)  # Construct or transform graph topology, geometry, or molecular feature data.
        return (batched_graph, batched_nodes), batched_line_graph, batched_full_graph, ids  # Return this computed tensor, metric, object, or collection to the caller.
