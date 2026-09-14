# =============================================================================
# MODULE: pre-training.py
# PURPOSE: Original Fabric atom-masking pretraining entry point.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Lightning CSV logs and pretrained checkpoints.
# CALCULATIONS: No additional project-specific equation beyond the operations identified in the line annotations and called modules.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Original Fabric entry point for masked-atom MGT pretraining."""

import os  # Load a standard-library, scientific, or local project dependency.
import time  # Load a standard-library, scientific, or local project dependency.
import pathlib  # Load a standard-library, scientific, or local project dependency.
import argparse  # Load a standard-library, scientific, or local project dependency.

import dgl  # Load a standard-library, scientific, or local project dependency.
import torch  # Load a standard-library, scientific, or local project dependency.
import numpy as np  # Load a standard-library, scientific, or local project dependency.
import os.path as osp  # Load a standard-library, scientific, or local project dependency.
import torch.nn as nn  # Load a standard-library, scientific, or local project dependency.
import torch.optim as optim  # Load a standard-library, scientific, or local project dependency.

from model.transformer import multiheaded  # Import selected classes or functions from the named dependency.
from model.alignn import EdgeGatedGraphConv  # Import selected classes or functions from the named dependency.
from model.graphformer import Graphformer, encoder  # Import selected classes or functions from the named dependency.
from utils.datasets import StructureDataset  # Import selected classes or functions from the named dependency.
from utils.masker import MaskAtom  # Import selected classes or functions from the named dependency.

from torch.nn import Linear  # Import selected classes or functions from the named dependency.
from lightning.fabric import Fabric  # Import selected classes or functions from the named dependency.
from torch.utils.data import DataLoader  # Import selected classes or functions from the named dependency.
from lightning.fabric.loggers import CSVLogger  # Import selected classes or functions from the named dependency.
from lightning.fabric.strategies import FSDPStrategy  # Import selected classes or functions from the named dependency.


# FUNCTION: pre_train — see its docstring and inline comments.
def pre_train(args, loader, main_model, atom_model, main_optim, atom_optim, criterion, fabric: Fabric):  # Define this callable; its indented block implements the documented operation.
    """Run one masked-node reconstruction training epoch and return updated states."""
    main_model.train(), atom_model.train()  # Switch the model to training behaviour.
    main_optim.zero_grad(), atom_optim.zero_grad()  # Clear accumulated gradients before the next optimization update.
    epoch_loss = torch.zeros(2).to(fabric.local_rank)  # Compute or store a loss, error, residual, or regression evaluation statistic.

    for iteration, (graphs, lg, fg, _) in enumerate(loader):  # Iterate over the stated records, layers, batches, or graph elements.

        is_accumulating = iteration % args.n_cum != 0  # Bind this name to an intermediate value, configuration setting, or result.

        g = graphs[0]  # Construct or transform graph topology, geometry, or molecular feature data.
        nsg = graphs[1]  # Construct or transform graph topology, geometry, or molecular feature data.
        node_idxs = nsg.ndata[dgl.NID].tolist()  # Construct or transform graph topology, geometry, or molecular feature data.
        node_truths = nsg.ndata['node_feats']

        with fabric.no_backward_sync(main_model, enabled=is_accumulating), fabric.no_backward_sync(atom_model, enabled=is_accumulating):  # Enter a managed context so resources and graph state are cleaned up safely.
            _, node_rep, _, _, _ = main_model(g, lg, fg)  # Construct or transform graph topology, geometry, or molecular feature data.
            pred_node = atom_model(node_rep[node_idxs])  # Construct or transform graph topology, geometry, or molecular feature data.
            loss = criterion(pred_node, node_truths)  # Compute or store a loss, error, residual, or regression evaluation statistic.
            fabric.backward(loss)  # Backpropagate the loss to compute gradients for trainable parameters.

        if not is_accumulating:  # Evaluate this condition before executing the associated branch.
            main_optim.step()  # Advance the optimizer or learning-rate scheduler by one update.
            main_optim.zero_grad()  # Clear accumulated gradients before the next optimization update.
            atom_optim.step()  # Advance the optimizer or learning-rate scheduler by one update.
            atom_optim.zero_grad()  # Clear accumulated gradients before the next optimization update.

        # Save Loss
        epoch_loss[0] += (loss.item() * args.n_cum)  # Compute or store a loss, error, residual, or regression evaluation statistic.
        epoch_loss[1] += args.batch_size  # Compute or store a loss, error, residual, or regression evaluation statistic.

    fabric.all_reduce(epoch_loss, reduce_op='sum')
    epoch_loss = epoch_loss[0] / epoch_loss[1]  # Compute or store a loss, error, residual, or regression evaluation statistic.
    fabric.print('Epoch loss: %.4f' % epoch_loss)
    return main_model, atom_model, main_optim, atom_optim, epoch_loss  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: validate — see its docstring and inline comments.
def validate(args, loader, main_model, atom_model, criterion, fabric):  # Define this callable; its indented block implements the documented operation.
    """Evaluate masked-node reconstruction loss and exact-vector accuracy."""
    main_model.eval(), atom_model.eval()  # Switch the model to deterministic evaluation behaviour.
    epoch_loss = torch.zeros(2).to(fabric.local_rank)  # Compute or store a loss, error, residual, or regression evaluation statistic.
    epoch_matches = torch.empty(0).to(fabric.local_rank)  # Bind this name to an intermediate value, configuration setting, or result.

    for graphs, lg, fg, _ in loader:  # Iterate over the stated records, layers, batches, or graph elements.

        g = graphs[0]  # Construct or transform graph topology, geometry, or molecular feature data.
        nsg = graphs[1]  # Construct or transform graph topology, geometry, or molecular feature data.
        node_idxs = nsg.ndata[dgl.NID].tolist()  # Construct or transform graph topology, geometry, or molecular feature data.
        node_truths = nsg.ndata['node_feats']

        with torch.no_grad():  # Enter a managed context so resources and graph state are cleaned up safely.
            _, node_rep, _, _, _ = main_model(g, lg, fg)  # Construct or transform graph topology, geometry, or molecular feature data.
            pred_node = atom_model(node_rep[node_idxs])  # Construct or transform graph topology, geometry, or molecular feature data.
            loss = criterion(pred_node, node_truths)  # Compute or store a loss, error, residual, or regression evaluation statistic.

        # Save Loss
        epoch_loss[0] += (loss.item() * args.n_cum)  # Compute or store a loss, error, residual, or regression evaluation statistic.
        epoch_loss[1] += args.batch_size  # Compute or store a loss, error, residual, or regression evaluation statistic.

        # Get accuracy of current iteration
        pred_atoms = (torch.sigmoid(pred_node) > 0.5).float()  # Construct or transform graph topology, geometry, or molecular feature data.
        correct_atoms = torch.all(pred_atoms == node_truths, dim=1)  # Construct or transform graph topology, geometry, or molecular feature data.
        epoch_matches = torch.cat((epoch_matches, correct_atoms), dim=0)  # Bind this name to an intermediate value, configuration setting, or result.

    # Get overall accuracy accross all iterations
    accuracy = epoch_matches.to(torch.float32).mean()  # Bind this name to an intermediate value, configuration setting, or result.
    fabric.print(f'Validation accuracy: {accuracy}')  # Perform this step of the surrounding calculation or control-flow block.

    # Get overall loss and return it
    fabric.all_reduce(epoch_loss, reduce_op='sum')
    epoch_loss = epoch_loss[0] / epoch_loss[1]  # Compute or store a loss, error, residual, or regression evaluation statistic.
    return epoch_loss, accuracy  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: main — see its docstring and inline comments.
def main(args):  # Define this callable; its indented block implements the documented operation.

    """Configure Fabric, datasets, models, optimizers and the pretraining loop."""
    if not osp.exists(args.model_path):  # Evaluate this condition before executing the associated branch.
        os.makedirs(args.model_path)  # Perform this step of the surrounding calculation or control-flow block.

    # ------------------------------------- FABRIC SETUP -------------------------------------
    logger = CSVLogger(  # Bind this name to an intermediate value, configuration setting, or result.
        root_dir=args.save_dir,  # Bind this name to an intermediate value, configuration setting, or result.
        name=args.run_name,  # Bind this name to an intermediate value, configuration setting, or result.
        flush_logs_every_n_steps=1  # Bind this name to an intermediate value, configuration setting, or result.
    )  # Continue or close the surrounding multiline expression or collection.
    policy = {encoder, EdgeGatedGraphConv, multiheaded}  # Construct or transform graph topology, geometry, or molecular feature data.
    fsdp_strategy = FSDPStrategy(auto_wrap_policy=policy, activation_checkpointing_policy=policy, state_dict_type='full')
    if args.accelerator == 'cpu' or args.accelerator == 'mps':
        fabric = Fabric(accelerator=args.accelerator, devices=args.n_devices, num_nodes=args.n_nodes, loggers=logger)  # Construct or transform graph topology, geometry, or molecular feature data.
    elif args.accelerator == 'gpu' or args.accelerator == 'cuda':
        fabric = Fabric(accelerator=args.accelerator, devices=args.n_devices, num_nodes=args.n_nodes, strategy=fsdp_strategy, loggers=logger)  # Construct or transform graph topology, geometry, or molecular feature data.
    else:  # Handle the remaining case not covered by earlier conditions.
        fabric = Fabric(accelerator='auto', devices=args.n_devices, num_nodes=args.n_nodes, loggers=logger)
    fabric.launch()  # Perform this step of the surrounding calculation or control-flow block.

    # ------------------------------------- DATASET SETUP -------------------------------------
    data = StructureDataset(args, transform=MaskAtom(  # Prepare dataset membership or batched data access for the experiment.
        num_atom_fea=args.num_atom_fea, node_feat_name='node_feats', mask_rate=args.mask_rate
    ))  # Continue or close the surrounding multiline expression or collection.
    training_data, validation_data = torch.utils.data.random_split(data, [args.train_split, args.val_split])  # Prepare dataset membership or batched data access for the experiment.
    training_loader = DataLoader(training_data, collate_fn=data.collate_pre, batch_size=args.batch_size, shuffle=True)  # Prepare dataset membership or batched data access for the experiment.
    validation_loader = DataLoader(validation_data, collate_fn=data.collate_pre, batch_size=args.batch_size, shuffle=True)  # Prepare dataset membership or batched data access for the experiment.
    training_loader, validation_loader = fabric.setup_dataloaders(training_loader), fabric.setup_dataloaders(validation_loader)  # Prepare dataset membership or batched data access for the experiment.

    # ------------------------------------- MODEL, OPTIMIZER AND LOSS/ERROR FUNCTION SETUP -------------------------------------
    main_model = Graphformer(args=args)  # Construct or transform graph topology, geometry, or molecular feature data.
    main_model.freeze_pretrain()  # Perform this step of the surrounding calculation or control-flow block.
    atom_model = nn.Sequential(Linear(args.hidden_dims, args.num_atom_fea), nn.Sigmoid())  # Create or apply a trainable neural-network component.
    main_model, atom_model = fabric.setup_module(main_model), fabric.setup_module(atom_model)  # Create or apply a trainable neural-network component.

    main_optim = optim.Adam(filter(lambda p: p.requires_grad, main_model.parameters()), lr=args.lr, weight_decay=args.decay)  # Create or apply a trainable neural-network component.
    atom_optim = optim.Adam(atom_model.parameters(), lr=args.lr, weight_decay=args.decay)  # Create or apply a trainable neural-network component.
    main_optim, atom_optim = fabric.setup_optimizers(main_optim), fabric.setup_optimizers(atom_optim)  # Bind this name to an intermediate value, configuration setting, or result.

    if args.load_model == 1:  # Evaluate this condition before executing the associated branch.
        # Check if there are model checkpoints
        pt_main_path = osp.join(args.model_path, f'mm_checkpoint.{args.begin_epoch}epochs.ckpt')  # Create or apply a trainable neural-network component.
        pt_atom_path = osp.join(args.model_path, f'am_checkpoint.{args.begin_epoch}epochs.ckpt')  # Create or apply a trainable neural-network component.
        assert osp.exists(pt_main_path) and osp.exists(pt_atom_path), f'No models checkpoint for epoch {args.begin_epoch} exist in path {str(args.model_path)}'  # Enforce an invariant required by the following calculation.
        # Load models
        main_state = {'model': main_model, 'optim_state': main_optim}
        fabric.load(pt_main_path, state=main_state)  # Bind this name to an intermediate value, configuration setting, or result.
        atom_state = {'node_model': atom_model, 'node_optim': atom_optim}
        fabric.load(pt_atom_path, state=atom_state)  # Bind this name to an intermediate value, configuration setting, or result.

    criterion = nn.BCEWithLogitsLoss()  # Compute or store a loss, error, residual, or regression evaluation statistic.

    fabric.print('-------------------- Pre-Training Started --------------------', flush=True)
    lowest_error = np.inf  # Compute or store a loss, error, residual, or regression evaluation statistic.
    per_epoch_times = []  # Bind this name to an intermediate value, configuration setting, or result.
    start_time = time.time()  # Bind this name to an intermediate value, configuration setting, or result.

    for epoch in range(args.begin_epoch, args.epochs + 1):  # Iterate over the stated records, layers, batches, or graph elements.
        # -------------------- TRAINING --------------------
        training_start_time = time.time()  # Bind this name to an intermediate value, configuration setting, or result.
        main_model, atom_model, main_optim, atom_optim, epoch_loss = pre_train(args, training_loader, main_model, atom_model, main_optim, atom_optim, criterion, fabric)  # Compute or store a loss, error, residual, or regression evaluation statistic.
        fabric.print(f'Training time: {time.time() - training_start_time} seconds')  # Perform this step of the surrounding calculation or control-flow block.

        # ------------------- VALIDATION -------------------
        validation_start_time = time.time()  # Bind this name to an intermediate value, configuration setting, or result.
        epoch_error, epoch_accuracy = validate(args, validation_loader, main_model, atom_model, criterion, fabric)  # Compute or store a loss, error, residual, or regression evaluation statistic.
        fabric.print(f'Validation time: {time.time() - validation_start_time} seconds')  # Perform this step of the surrounding calculation or control-flow block.

        # ------------------- LOG RESULTS ------------------
        per_epoch_time = time.time() - training_start_time  # Bind this name to an intermediate value, configuration setting, or result.
        fabric.print('Completed Epoch %d of %d in %.2f s' % (epoch, args.epochs, per_epoch_time), flush=True)
        per_epoch_times.append(per_epoch_time)  # Perform this step of the surrounding calculation or control-flow block.

        fabric.log_dict({'Pre-Training Loss': epoch_loss, 'Pre-Training Error': epoch_error, 'Pre-Training Accuracy': epoch_accuracy},  step=epoch)

        # ----------------- CHECKPOINT MODEL ---------------
        if epoch % 10 == 0:  # Evaluate this condition before executing the associated branch.
            main_state = {'model': main_model, 'optim_state': main_optim}
            fabric.save(osp.join(args.model_path, f'mm_checkpoint.{epoch}epochs.ckpt'), main_state)  # Perform this step of the surrounding calculation or control-flow block.
            atom_state = {'node_model': atom_model, 'node_optim': atom_optim}
            fabric.save(osp.join(args.model_path, f'am_checkpoint.{epoch}epochs.ckpt'), atom_state)  # Perform this step of the surrounding calculation or control-flow block.

        # ------------- SAVE LOWEST ERROR MODEL ------------
        if epoch_error < lowest_error:  # Evaluate this condition before executing the associated branch.
            lowest_error = epoch_error  # Compute or store a loss, error, residual, or regression evaluation statistic.
            main_state = {'model': main_model, 'optim_state': main_optim}
            fabric.save(osp.join(args.model_path, args.pretrain_model), main_state)  # Perform this step of the surrounding calculation or control-flow block.

    fabric.print(f'Average per epoch time: {np.mean(per_epoch_times)} seconds, Total {args.epochs} epochs time: {time.time() - start_time} seconds')  # Perform this step of the surrounding calculation or control-flow block.
    fabric.print('-------------------- Pre-Training Finished --------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of Pre-Training strategy for the Molecular Graph Transformer")
    # Fabric Arguments
    parser.add_argument('--n_devices', type=int, default=8, help='number of gpus/cpus that the code has access to (default: 8)')
    parser.add_argument('--n_nodes', type=int, default=1, help='number of nodes/computers on which the model is being trained (default: 1)')
    parser.add_argument('--accelerator', type=str, default='cuda', choices=['cpu', 'gpu', 'mps', 'cuda', 'tpu'],
                        help='device type on which the training is happening [cpu, gpu, mps (apple M1/M2 only), cuda (NVIDIA GPUs only), tpu] (default: cuda)')
    # Save and Load Arguments
    parser.add_argument('--root', type=str, help='root directory for all datasets (default: None)', required=True)
    parser.add_argument('--model_path', type=str, help='directory in which to save the trained model', required=True)
    parser.add_argument('--run_name', type=str, default=None, help='name of run for logging purposes')
    parser.add_argument('--save_dir', type=str, default=None, help='directory in which to save the test results')
    parser.add_argument('--pretrain_model', type=str, default='pretrain.ckpt', help='name with which to save the pretrained model')
    parser.add_argument('--load_model', type=int, default=0, choices=[0, 1], help='whether there is a model to be loaded (0: no model loading, 1: load checkpoint)')
    parser.add_argument('--n_ckpt', type=int, default=10, help='number of epochs to wait before saving a checkpoint of the model (default: 10)')
    # Training Arguments
    parser.add_argument('--n_cum', type=int, default=8, help='number of batches to accumulate the error for')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training (default: 2)')
    parser.add_argument('--train_split', type=float, default=0.8, help='number of items in dataset to be used for training (default: 0.8)')
    parser.add_argument('--val_split', type=float, default=0.2, help='number of items in dataset to be used for validation (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for (default: 100)')
    parser.add_argument('--begin_epoch', type=int, default=1, help='set to restart training from a specific epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=1e-5, help='weight decay for the optimizers (default: 1e-5)')
    # Model Arguments
    parser.add_argument('--process', type=int, default=1, choices=[0, 1], help='whether the graphs for the structures/molecules need to be created during dataset loading (default: True)')
    parser.add_argument('--max_nei_num', type=int, default=12, help='maximum number of neighbour allowed for each atom in the local graph (default: 12)')
    parser.add_argument('--local_radius', type=int, default=8, help='radius used to form the local graph (default: 8)')
    parser.add_argument('--periodic', type=int, default=1, choices=[0, 1], help='whether the input structure is a periodic structure or not (default: True)')
    parser.add_argument('--periodic_radius', type=int, default=12, help='radius used to form the fully connected graph (default: 12)')
    parser.add_argument('--num_atom_fea', type=int, default=90, help='length of feature vector for atoms (default: 90)')
    parser.add_argument('--num_edge_fea', type=int, default=1, help='length of feature vector for edges in local graph (default: 1)')
    parser.add_argument('--num_angle_fea', type=int, default=1, help='length of feature vector for edges in line graph (default: 1)')
    parser.add_argument('--num_pe_fea', type=int, default=10, help='length of feature vector for atom\'s positional encoding (default: 10)')
    parser.add_argument('--num_clmb_fea', type=int, default=1, help='length of feature vector for edges in fully connected graph (default: 1)')
    parser.add_argument('--num_edge_bins', type=int, default=80, help='number of bins for RBF expansion of edges in local graph (default: 80)')
    parser.add_argument('--num_angle_bins', type=int, default=40, help='number of bins for RBF expansion of edges in line graph (default: 40)')
    parser.add_argument('--num_clmb_bins', type=int, default=120, help='number of bins for RBF expansion of edges in fully connected graph (default: 120)')
    parser.add_argument('--embedding_dims', type=int, default=128, help='dimension of embedding layer (default: 128)')
    parser.add_argument('--hidden_dims', type=int, default=512, help='dimensions of each hidden layer (default: 512)')
    parser.add_argument('--out_dims', type=int, default=3, help='length of output vector of the network (default: 3)')
    parser.add_argument('--num_layers', type=int, default=1, help='number of encoders in the network (default: 1)')
    parser.add_argument('--n_mha', type=int, default=1, help='number of attention layers in each encoder (default: 1)')
    parser.add_argument('--n_alignn', type=int, default=3, help='number of graph convolutions in each encoder (default: 3)')
    parser.add_argument('--n_gnn', type=int, default=3, help='number of graph convolutions in each encoder (default: 3)')
    parser.add_argument('--n_heads', type=int, default=4, help='number of attention heads (default: 4)')
    parser.add_argument('--residual', type=int, default=1, choices=[0, 1], help='whether to add residuality to the network or not (default: True)')
    parser.add_argument('--mask_rate', type=float, default=0.2, help='percentage of node to be masked (default: 0.2)')

    args = parser.parse_args()  # Bind this name to an intermediate value, configuration setting, or result.

    args.residual = bool(args.residual)  # Compute or store a loss, error, residual, or regression evaluation statistic.
    args.periodic = bool(args.periodic)  # Bind this name to an intermediate value, configuration setting, or result.
    args.process = bool(args.process)  # Bind this name to an intermediate value, configuration setting, or result.

    if args.save_dir is None:  # Evaluate this condition before executing the associated branch.
        args.save_dir = osp.join(os.getcwd(), 'output', 'pre-train')
        if not osp.exists(args.save_dir):  # Evaluate this condition before executing the associated branch.
            directory = pathlib.Path(args.save_dir)  # Bind this name to an intermediate value, configuration setting, or result.
            directory.mkdir(parents=True, exist_ok=True)  # Bind this name to an intermediate value, configuration setting, or result.

    if args.run_name is None:  # Evaluate this condition before executing the associated branch.
        args.run_name = f'{args.num_layers}_{args.n_mha}_{args.n_alignn}_{args.n_gnn}'  # Create or apply a trainable neural-network component.

    main(args)  # Perform this step of the surrounding calculation or control-flow block.
