# =============================================================================
# MODULE: utils/masker.py
# PURPOSE: DGL transform that masks atom feature vectors and records reconstruction targets.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Returns Python objects/tensors to its caller; this module does not directly define a persistent output artifact.
# CALCULATIONS: No additional project-specific equation beyond the operations identified in the line annotations and called modules.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Original DGL transform for self-supervised random atom masking."""

import dgl  # Load a standard-library, scientific, or local project dependency.
import torch  # Load a standard-library, scientific, or local project dependency.
import random  # Load a standard-library, scientific, or local project dependency.

from dgl import BaseTransform  # Import selected classes or functions from the named dependency.


# CLASS: MaskAtom — reusable model/data abstraction.
class MaskAtom(BaseTransform):  # Define this reusable class and its inheritance contract.
    """Zero randomly selected node features while retaining their reconstruction labels."""
    def __init__(self, num_atom_fea, mask_rate, node_feat_name):  # Define this callable; its indented block implements the documented operation.
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        :param num_atom_fea:
        :param mask_rate: % of atoms to be masked
        masked atoms
        """
        self.num_atom_fea = num_atom_fea  # Store this configuration value or neural-network submodule on the instance.
        self.node_feat_name = node_feat_name  # Store this configuration value or neural-network submodule on the instance.
        self.mask_rate = mask_rate  # Store this configuration value or neural-network submodule on the instance.

    def __call__(self, g, masked_atom_indices=None):  # Define this callable; its indented block implements the documented operation.
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                      [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples (num_atoms * mask rate) number of atom indices, 
            otherwise a list of atom idx that sets the atoms to be masked 
            (for debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        """

        if masked_atom_indices == None:  # Evaluate this condition before executing the associated branch.
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = g.num_nodes()  # Construct or transform graph topology, geometry, or molecular feature data.
            sample_size = int(num_atoms * self.mask_rate + 1)  # Bind this name to an intermediate value, configuration setting, or result.
            masked_atom_indices = random.sample(range(num_atoms), sample_size)  # Bind this name to an intermediate value, configuration setting, or result.

        nsg = dgl.node_subgraph(g, masked_atom_indices, relabel_nodes=True, store_ids=True)  # Construct or transform graph topology, geometry, or molecular feature data.
        nodes = nsg.ndata.pop(self.node_feat_name)  # Construct or transform graph topology, geometry, or molecular feature data.
        for id in masked_atom_indices:  # Iterate over the stated records, layers, batches, or graph elements.
            g.ndata[self.node_feat_name][id] = torch.zeros(self.num_atom_fea)  # Construct or transform graph topology, geometry, or molecular feature data.
        nsg.ndata[self.node_feat_name] = nodes  # Construct or transform graph topology, geometry, or molecular feature data.
        return g, nsg  # Return this computed tensor, metric, object, or collection to the caller.

    def __repr__(self):  # Define this callable; its indented block implements the documented operation.
        """Return a concise printable transform configuration."""
        return '{}(num_atom_fea={}, mask_rate={})'.format(self.__class__.__name__, self.num_atom_fea, self.mask_rate)