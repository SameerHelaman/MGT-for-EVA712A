# =============================================================================
# MODULE: training.py
# PURPOSE: Original Fabric supervised MGT training and validation entry point.
# LIBRARIES: Imports below identify Python standard-library, third-party scientific,
#            and local MGT/OpenBind modules used by this file.
# OUTPUT: Training/validation CSV logs plus lowest/end checkpoints.
# CALCULATIONS: No additional project-specific equation beyond the operations identified in the line annotations and called modules.
# NOTE: Comments document the implementation; executable statements are unchanged.
# =============================================================================
"""Original Fabric entry point for supervised MGT training and validation."""

import os  # Load a standard-library, scientific, or local project dependency.
import time  # Load a standard-library, scientific, or local project dependency.
import pathlib  # Load a standard-library, scientific, or local project dependency.
import argparse  # Load a standard-library, scientific, or local project dependency.
import warnings  # Load a standard-library, scientific, or local project dependency.

import numpy as np  # Load a standard-library, scientific, or local project dependency.
import os.path as osp  # Load a standard-library, scientific, or local project dependency.

from model.transformer import multiheaded  # Import selected classes or functions from the named dependency.
from model.alignn import EdgeGatedGraphConv  # Import selected classes or functions from the named dependency.
from model.graphformer import Graphformer, encoder  # Import selected classes or functions from the named dependency.
from utils.datasets import StructureDataset  # Import selected classes or functions from the named dependency.

import torch  # Load a standard-library, scientific, or local project dependency.
import torch.nn as nn  # Load a standard-library, scientific, or local project dependency.
from lightning.fabric import Fabric  # Import selected classes or functions from the named dependency.
from torch.utils.data import DataLoader  # Import selected classes or functions from the named dependency.
from lightning.fabric.loggers import CSVLogger  # Import selected classes or functions from the named dependency.
from lightning.fabric.strategies import FSDPStrategy  # Import selected classes or functions from the named dependency.


# FUNCTION: train — see its docstring and inline comments.
def train(args, model, loader, optimizer, criterion, fabric):  # Define this callable; its indented block implements the documented operation.
    """Run one original supervised training epoch with gradient accumulation."""
    model.train()  # Switch the model to training behaviour.
    optimizer.zero_grad()  # Clear accumulated gradients before the next optimization update.
    epoch_loss = torch.zeros(2).to(fabric.local_rank)  # Compute or store a loss, error, residual, or regression evaluation statistic.

    for iteration, (g, lg, fg, target, _) in enumerate(loader):  # Iterate over the stated records, layers, batches, or graph elements.

        is_accumulating = iteration % args.n_cum != 0  # Bind this name to an intermediate value, configuration setting, or result.

        with fabric.no_backward_sync(model, enabled=is_accumulating):  # Enter a managed context so resources and graph state are cleaned up safely.
            output, _, _, _, _ = model(g, lg, fg)  # Create or apply a trainable neural-network component.
            loss = criterion(output, target) / args.n_cum  # Compute or store a loss, error, residual, or regression evaluation statistic.
            fabric.backward(loss)  # Backpropagate the loss to compute gradients for trainable parameters.

        if not is_accumulating:  # Evaluate this condition before executing the associated branch.
            optimizer.step()  # Advance the optimizer or learning-rate scheduler by one update.
            optimizer.zero_grad()  # Clear accumulated gradients before the next optimization update.

        # Save Loss
        epoch_loss[0] += (loss.item() * args.n_cum)  # Compute or store a loss, error, residual, or regression evaluation statistic.
        epoch_loss[1] += args.batch_size  # Compute or store a loss, error, residual, or regression evaluation statistic.

    fabric.all_reduce(epoch_loss, reduce_op='sum')
    epoch_loss = epoch_loss[0] / epoch_loss[1]  # Compute or store a loss, error, residual, or regression evaluation statistic.
    fabric.print('Epoch loss: %.4f' % epoch_loss)
    return model, optimizer, epoch_loss  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: validate — see its docstring and inline comments.
def validate(args, model, loader, criterion, fabric):  # Define this callable; its indented block implements the documented operation.
    """Calculate overall and per-output validation MAE without gradients."""
    model.eval()  # Switch the model to deterministic evaluation behaviour.
    epoch_error = torch.zeros(2).to(fabric.local_rank)  # Compute or store a loss, error, residual, or regression evaluation statistic.
    epoch_indiv_error = [torch.zeros(2).to(fabric.local_rank) for _ in range(args.out_dims)]  # Compute or store a loss, error, residual, or regression evaluation statistic.

    for g, lg, fg, target, _ in loader:  # Iterate over the stated records, layers, batches, or graph elements.
        with torch.no_grad():  # Enter a managed context so resources and graph state are cleaned up safely.
            output, _, _, _, _ = model(g, lg, fg)  # Create or apply a trainable neural-network component.

        # Get overall loss and error
        error = criterion(output, target)  # Compute or store a loss, error, residual, or regression evaluation statistic.
        epoch_error[0] += error.item()  # Compute or store a loss, error, residual, or regression evaluation statistic.
        epoch_error[1] += args.batch_size  # Compute or store a loss, error, residual, or regression evaluation statistic.

        # Get individual errors
        if args.out_dims > 1:  # Evaluate this condition before executing the associated branch.
            targets = torch.hsplit(target, int(target.shape[1]))  # Prepare dataset membership or batched data access for the experiment.
            outputs = torch.hsplit(output, int(output.shape[1]))  # Prepare dataset membership or batched data access for the experiment.
            individual_errors = [criterion(outputs[i], targets[i]) for i in range(len(outputs))]  # Compute or store a loss, error, residual, or regression evaluation statistic.
            for i, error in enumerate(individual_errors):  # Iterate over the stated records, layers, batches, or graph elements.
                epoch_indiv_error[i][0] += error.item()  # Compute or store a loss, error, residual, or regression evaluation statistic.
                epoch_indiv_error[i][1] += args.batch_size  # Compute or store a loss, error, residual, or regression evaluation statistic.

    fabric.all_reduce(epoch_error, reduce_op='sum')
    epoch_error = epoch_error[0] / epoch_error[1]  # Compute or store a loss, error, residual, or regression evaluation statistic.
    if args.out_dims > 1:  # Evaluate this condition before executing the associated branch.
        for i, error in enumerate(epoch_indiv_error):  # Iterate over the stated records, layers, batches, or graph elements.
            error = fabric.all_reduce(error, reduce_op='sum')
            epoch_indiv_error[i] = error[0] / error[1]  # Compute or store a loss, error, residual, or regression evaluation statistic.

    error_str = 'Validation error: %.4f' % epoch_error
    if args.out_dims > 1:  # Evaluate this condition before executing the associated branch.
        for i in range(args.out_dims):  # Iterate over the stated records, layers, batches, or graph elements.
            error_str += str(f' | {args.out_names[i]} Error: {epoch_indiv_error[i]}')  # Compute or store a loss, error, residual, or regression evaluation statistic.
    fabric.print(error_str)  # Perform this step of the surrounding calculation or control-flow block.
    return epoch_error, epoch_indiv_error  # Return this computed tensor, metric, object, or collection to the caller.


# FUNCTION: main — see its docstring and inline comments.
def main(args):  # Define this callable; its indented block implements the documented operation.

    """Configure Fabric, split data, train Graphformer and save checkpoints."""
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
    dataset = StructureDataset(args, process=args.process)  # Prepare dataset membership or batched data access for the experiment.
    training_data, validation_data = torch.utils.data.random_split(dataset, [args.train_split, args.val_split])  # Prepare dataset membership or batched data access for the experiment.
    training_loader = DataLoader(training_data, collate_fn=dataset.collate_tt, batch_size=args.batch_size, shuffle=True)  # Prepare dataset membership or batched data access for the experiment.
    validation_loader = DataLoader(validation_data, collate_fn=dataset.collate_tt, batch_size=args.batch_size, shuffle=True)  # Prepare dataset membership or batched data access for the experiment.
    training_loader, validation_loader = fabric.setup_dataloaders(training_loader), fabric.setup_dataloaders(validation_loader)  # Prepare dataset membership or batched data access for the experiment.

    # ------------------------------------- MODEL, OPTIMIZER AND LOSS/ERROR FUNCTION SETUP -------------------------------------
    model = Graphformer(args=args)  # Construct or transform graph topology, geometry, or molecular feature data.
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())  # Create or apply a trainable neural-network component.
    num_params = sum(p.numel() for p in model_parameters)  # Create or apply a trainable neural-network component.
    fabric.print(f'ARCHITECTURE: \n'  # Perform this step of the surrounding calculation or control-flow block.
                 f'\tLayers - {args.num_layers} \n'  # Perform this step of the surrounding calculation or control-flow block.
                 f'\tMHAs - {args.n_mha} \n'  # Perform this step of the surrounding calculation or control-flow block.
                 f'\tALIGNNs - {args.n_alignn} \n'  # Perform this step of the surrounding calculation or control-flow block.
                 f'\tGNNs - {args.n_gnn} \n'  # Perform this step of the surrounding calculation or control-flow block.
                 f'\tParameters - {num_params}')  # Perform this step of the surrounding calculation or control-flow block.

    if args.load_model == 1:  # Evaluate this condition before executing the associated branch.
        warnings.warn('Pre-Training Strategy not yet Implemented')
        exit()  # Perform this step of the surrounding calculation or control-flow block.
        # model.freeze_train()

    model = fabric.setup_module(model)  # Create or apply a trainable neural-network component.

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)  # Create or apply a trainable neural-network component.
    optimizer = fabric.setup_optimizers(optimizer)  # Bind this name to an intermediate value, configuration setting, or result.

    if args.load_model == 1:  # Evaluate this condition before executing the associated branch.
        pretrain_path = osp.join(args.model_path, args.pretrain_model)  # Create or apply a trainable neural-network component.
        assert osp.exists(pretrain_path), f'No model save as {args.pretrain_model} exists in path {str(args.model_path)}'  # Enforce an invariant required by the following calculation.
        state_dicts = {'model': model}
        fabric.load(pretrain_path, state=state_dicts)  # Bind this name to an intermediate value, configuration setting, or result.
    elif args.load_model == 2:  # Test this additional condition when earlier branches were not selected.
        lowest_path = osp.join(args.model_path, args.lowest_model)  # Create or apply a trainable neural-network component.
        assert osp.exists(lowest_path), f'No model save as {args.lowest_model} exists in path {str(args.model_path)}'  # Enforce an invariant required by the following calculation.
        state_dicts = {'model': model, 'optimizer': optimizer}
        fabric.load(lowest_path, state=state_dicts)  # Bind this name to an intermediate value, configuration setting, or result.

    train_loss = nn.MSELoss()  # Compute or store a loss, error, residual, or regression evaluation statistic.
    val_loss = nn.L1Loss()  # Compute or store a loss, error, residual, or regression evaluation statistic.

    # -------------------------------- TRAINING AND VALIDATION --------------------------------
    fabric.print('-------------------- Training and Validation Started --------------------')
    lowest_error = 1000000  # Compute or store a loss, error, residual, or regression evaluation statistic.
    per_epoch_times = []  # Bind this name to an intermediate value, configuration setting, or result.
    start_time = time.time()  # Bind this name to an intermediate value, configuration setting, or result.

    for epoch in range(args.begin_epoch, args.epochs + 1):  # Iterate over the stated records, layers, batches, or graph elements.
        # -------------------- TRAINING --------------------
        training_start_time = time.time()  # Bind this name to an intermediate value, configuration setting, or result.
        model, optimizer, epoch_loss = train(args, model, training_loader, optimizer, train_loss, fabric)  # Compute or store a loss, error, residual, or regression evaluation statistic.
        fabric.print(f'Training time: {time.time() - training_start_time} seconds')  # Perform this step of the surrounding calculation or control-flow block.

        # ------------------- VALIDATION -------------------
        validation_start_time = time.time()  # Bind this name to an intermediate value, configuration setting, or result.
        epoch_error, epoch_indiv_error = validate(args, model, validation_loader, val_loss, fabric)  # Compute or store a loss, error, residual, or regression evaluation statistic.
        fabric.print(f'Validation time: {time.time() - validation_start_time} seconds')  # Perform this step of the surrounding calculation or control-flow block.

        # ------------------- LOG RESULTS ------------------
        per_epoch_time = time.time() - training_start_time  # Bind this name to an intermediate value, configuration setting, or result.
        per_epoch_times.append(per_epoch_time)  # Perform this step of the surrounding calculation or control-flow block.
        fabric.print('Completed Epoch %d of %d in %.2f s' % (epoch, args.epochs, per_epoch_time))

        results = {'Train Loss': epoch_loss, 'Validation Error': epoch_error}
        if args.out_dims > 1:  # Evaluate this condition before executing the associated branch.
            for i in range(args.out_dims):  # Iterate over the stated records, layers, batches, or graph elements.
                results[f'{args.out_names[i]} Error'] = epoch_indiv_error[i]  # Compute or store a loss, error, residual, or regression evaluation statistic.
        fabric.log_dict(results, step=epoch)  # Report progress, predictions, or metrics to the selected output/logging backend.

        # ------------- SAVE LOWEST ERROR MODEL ------------
        if epoch_error < lowest_error:  # Evaluate this condition before executing the associated branch.
            lowest_error = epoch_error  # Compute or store a loss, error, residual, or regression evaluation statistic.
            state = {'model': model, 'optimizer': optimizer}
            fabric.save(osp.join(args.model_path, args.lowest_model), state)  # Perform this step of the surrounding calculation or control-flow block.

        # ------------------- VISUALIZE -------------------
        if epoch % 10 == 0 or epoch == 1:  # Evaluate this condition before executing the associated branch.
            model.eval()  # Switch the model to deterministic evaluation behaviour.
            g, lg, fg, target, _ = next(iter(validation_loader))  # Prepare dataset membership or batched data access for the experiment.
            with torch.no_grad():  # Enter a managed context so resources and graph state are cleaned up safely.
                output, _, _, _, _ = model(g, lg, fg)  # Create or apply a trainable neural-network component.

    fabric.print(f'Average per epoch time: {np.mean(per_epoch_times)} seconds, Total {args.epochs} epochs time: {time.time() - start_time} seconds')  # Perform this step of the surrounding calculation or control-flow block.
    fabric.print('-------------------- Training and Validation Finished --------------------')

    state = {'model': model, 'optimizer': optimizer}
    fabric.save(osp.join(args.model_path, args.final_model), state)  # Perform this step of the surrounding calculation or control-flow block.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of Training strategy for the Molecular Graph Transformer")
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
    parser.add_argument('--final_model', type=str, default='end_model.ckpt', help='name with which to save the model')
    parser.add_argument('--lowest_model', type=str, default='lowest.ckpt', help='name with which to save the model with the best performance')
    parser.add_argument('--load_model', type=int, default=0, choices=[0, 1, 2], help='whether there is a model to be loaded (0: no model loading, 1: load pretrained, 2: load checkpoint)')
    parser.add_argument('--out_names', nargs='+', type=str, default=None, help='names of the outputs [for logging purposes only]')
    # Training Arguments
    parser.add_argument('--n_cum', type=int, default=8, help='number of batches to accumulate the error for')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training (default: 2)')
    parser.add_argument('--train_split', type=float, default=0.8, help='number of items in dataset to be used for training (default: 0.8)')
    parser.add_argument('--val_split', type=float, default=0.2, help='number of items in dataset to be used for validation (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for (default: 100)')
    parser.add_argument('--begin_epoch', type=int, default=1, help='set to restart training from a specific epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--decay', type=float, default=1e-5, help='weight decay for the optimizers (default: 1e-5)')
    # Model and Dataset Arguments
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

    args = parser.parse_args()  # Bind this name to an intermediate value, configuration setting, or result.

    if args.out_names is None:  # Evaluate this condition before executing the associated branch.
        args.out_names = []  # Bind this name to an intermediate value, configuration setting, or result.
        for i in range(args.out_dims):  # Iterate over the stated records, layers, batches, or graph elements.
            args.out_names.append(str(i + 1))  # Perform this step of the surrounding calculation or control-flow block.
    assert len(args.out_names) == args.out_dims, 'number of outputs and output names not the same'
    args.residual = bool(args.residual)  # Compute or store a loss, error, residual, or regression evaluation statistic.
    args.periodic = bool(args.periodic)  # Bind this name to an intermediate value, configuration setting, or result.
    args.process = bool(args.process)  # Bind this name to an intermediate value, configuration setting, or result.

    if args.save_dir is None:  # Evaluate this condition before executing the associated branch.
        args.save_dir = osp.join(os.getcwd(), 'output', 'train')
        if not osp.exists(args.save_dir):  # Evaluate this condition before executing the associated branch.
            directory = pathlib.Path(args.save_dir)  # Bind this name to an intermediate value, configuration setting, or result.
            directory.mkdir(parents=True, exist_ok=True)  # Bind this name to an intermediate value, configuration setting, or result.

    if args.run_name is None:  # Evaluate this condition before executing the associated branch.
        args.run_name = f'{args.num_layers}_{args.n_mha}_{args.n_alignn}_{args.n_gnn}'  # Create or apply a trainable neural-network component.

    main(args)  # Perform this step of the surrounding calculation or control-flow block.
