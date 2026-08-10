#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:33:06 2020

@author: jacobb
"""

import json
import numpy as np
from numpy.dtypes import StringDType
import torch
import copy
from scipy.sparse.csgraph import shortest_path
from tem import model, parameters
import polars as pl
from pathlib import Path
import time
from types import SimpleNamespace
from tem import utils
from torch.utils.tensorboard import SummaryWriter
from tem import analyse




def generate_diagnostic_walk(environment, multiplier=100):
    """Generate a long random walk purely for extracting rate-map
    representations (not used for training). multiplier=100 was verified
    to give stable non-zero visit counts for all 6 locations even in the
    second half of the walk (which is what rate_map() averages over)."""
    walk = environment.generate_walks(environment.n_locations * multiplier, 1)[0]
    for step in walk:
        step[0] = [step[0]]
        step[1] = step[1].unsqueeze(dim=0)
        step[2] = [step[2]]
    return walk


def walks_operators(design, env, actions):
    """Create walks from a learning phase design.

    Returns (walks, blocks) where blocks[i] is the block number of walks[i].
    Boundary markers get the block they follow, so they travel with it when a
    block is repeated.
    """
    walks = []
    blocks = []
    nodes = [f"node_{n}" for n in range(1, 7)]
    rows = list(design.iter_rows(named=True))

    prev_block = None
    for i, row in enumerate(rows):
        current_block = row["block"]
        if prev_block is not None and current_block != prev_block:
            prev_target_ind = nodes.index(rows[i - 1]["target_node"])
            prev_obs = env.get_observation(env.locations[prev_target_ind])
            boundary_step = [
                [{"id": prev_target_ind, "shiny": None}],
                torch.stack([prev_obs], dim=0),
                [None],
            ]
            walks.append([boundary_step])
            blocks.append(prev_block)

        prev_block = current_block

        steps = []
        if row["trial_type"] == "integration":
            start_ind = nodes.index(row["start_node"])
            start_obs = env.get_observation(env.locations[start_ind])
            steps.append(
                [
                    [{"id": start_ind, "shiny": None}],
                    [start_obs],
                    [actions[row["move_direction"]]],
                ]
            )
        cue_ind = nodes.index(row["cue_node"])
        cue_obs = env.get_observation(env.locations[cue_ind])
        steps.append(
            [
                [{"id": cue_ind, "shiny": None}],
                [cue_obs],
                [actions[row["direction"]]],
            ]
        )
        target_ind = nodes.index(row["target_node"])
        target_obs = env.get_observation(env.locations[target_ind])
        steps.append(
            [
                [{"id": target_ind, "shiny": None}],
                [target_obs],
                [0],
            ]
        )
        for i_step, step in enumerate(steps):
            steps[i_step][1] = torch.stack(step[1], dim=0)
        walks.append(steps)
        blocks.append(current_block)

    return walks, blocks


def reset_perceptual_weights(tem_model, adam=None, reset_optimizer_state=False):
    """
    Re-initializes ONLY the perceptual decoding pathway's learned weights
    (w_x, b_x, MLP_c_star) back to their startup values, leaving every
    other trained weight (MLP_D_a, g_init, M, etc.) untouched.

    reset_optimizer_state: if True, also clear Adam's exp_avg/exp_avg_sq for
    these parameters. In-place weight modification keeps the same Parameter
    objects, so without this the freshly re-initialized weights are still
    driven by momentum accumulated during Initial-map training. Default False
    reproduces study-39's behavior exactly.
    """
    with torch.no_grad():
        tem_model.w_x.fill_(1.0)
        tem_model.b_x.zero_()
        for from_layer in range(2):
            for n in range(tem_model.MLP_c_star.N):
                torch.nn.init.xavier_normal_(tem_model.MLP_c_star.w[n][from_layer].weight)
                if tem_model.MLP_c_star.w[n][from_layer].bias is not None:
                    tem_model.MLP_c_star.w[n][from_layer].bias.fill_(0.0)

    if reset_optimizer_state and adam is not None:
        for param in [tem_model.w_x, tem_model.b_x, *tem_model.MLP_c_star.parameters()]:
            adam.state.pop(param, None)

def reinit_transition_weights(tem_model, scale=1.0):
    """
    Re-initialize MLP_D_a's hidden->output layer with xavier (times `scale`),
    replacing the exact zeros set by Model.init_trainable via
    set_weights(1, 0.0).

    Why: with W2 = 0 the transition delta is 0, so g_gen == g_prev at every
    step. At the target step g_prev is the cue's g, so gen_p(g_gen, M)
    retrieves the CUE's memory and the model predicts the cue's object.
    Measured on study-40 (target-step, 45-way argmax): block 1 accuracy falls
    from 16.2% (chance = 1/6 among the map's objects) in trials 1-20 to 8.1%
    in trials 61-100, while cue-copying rises from 18% to 76%. The model is
    converging on "predict no movement", and a sharper M makes that wrong
    answer sharper.

    W2 = 0 also freezes the first layer, since dL/dW1 = W2^T dL/ddelta = 0.
    Measured on study-40 (scale=0): after training, w.{n}.0.weight norms are still
    at their xavier init values (2.3-2.8 vs xavier ~2.58), confirming it barely moves.

    Original TEM can afford this warm start over 10000 iterations. Here block 1
    is 168 trials, and climbing W2 off zero consumes most of it.

    scale=0.0 leaves the zero init in place, reproducing the pre-study-44 behavior.
    """
    with torch.no_grad():
        for n in range(tem_model.MLP_D_a.N):
            w = tem_model.MLP_D_a.w[n][1].weight
            torch.nn.init.xavier_normal_(w)
            w.mul_(scale)


def parameter_iteration_decoupled(i_memory, i_weights, params):
    """
    Same six outputs as parameters.parameter_iteration(), but eta/lambda
    (Hebbian memory write/forget rate) are computed from i_memory, while
    everything else (lr, p2g_scale_offset, walk_length_center, loss_weights)
    is computed from i_weights. Lets the memory schedule be reset at a
    design transition without also re-maximizing the network's learning
    rate/loss weights.
    """
    eta, lamb, _, _, _, _ = parameters.parameter_iteration(i_memory, params)
    _, _, p2g_scale_offset, lr, walk_length_center, loss_weights = parameters.parameter_iteration(i_weights, params)
    return eta, lamb, p2g_scale_offset, lr, walk_length_center, loss_weights


# def learn_walks(walks, env, tem_model, adam, params, out_dir, i, run=1,
#                  save_representations=20, prev_iter=None, accumulate_steps=1,
#                  memory_iteration_offset=0):
#     """
#     memory_iteration_offset: subtracted from `i` when computing eta/lambda
#     (NOT lr/loss weights), so the memory-write schedule can be independently
#     reset at a design transition. Default 0 = memory schedule follows i
#     normally (current behavior). Set to the value of `i` at the start of
#     the new design to make eta/lambda restart from ~0 there, while lr and
#     loss weights keep following the unmodified, continuous i.
#     """
#     visited = [[False for _ in range(env.n_locations)]]
#
#     str_dir = str(out_dir) + "/"
#     writer = SummaryWriter(str_dir)
#     logger = utils.make_logger(str_dir)
#     log_interval = 1
#
#     logits_list = []
#     representation_snapshots = []
#     per_trial_correct = []
#
#     accum_counter = 0
#     adam.zero_grad()
#
#     n_walks = len(walks)
#     for walk_i, walk in enumerate(walks):
#         i += 1
#         i_memory = i - memory_iteration_offset
#         is_last_walk = (walk_i == n_walks - 1)
#
#         (eta_new, lambda_new, p2g_scale_offset, lr, walk_length_center,
#          loss_weights) = parameter_iteration_decoupled(i_memory, i, params)
#         start_time = time.time()
#         tem_model.hyper["eta"] = eta_new
#         tem_model.hyper["lambda"] = lambda_new
#         tem_model.hyper["p2g_scale_offset"] = p2g_scale_offset
#         for param_group in adam.param_groups:
#             param_group["lr"] = lr
#
#         forward = tem_model(walk, prev_iter)
#
#         loss = torch.tensor(0.0, requires_grad=True)
#         plot_loss = 0
#         for step in forward:
#             step_loss = []
#             for env_i, env_visited in enumerate(visited):
#                 if env_visited[step.g[env_i]["id"]]:
#                     step_loss.append(loss_weights * torch.stack([l[env_i] for l in step.L]))
#                 else:
#                     env_visited[step.g[env_i]["id"]] = True
#             step_loss = (
#                 torch.tensor(0) if not step_loss
#                 else torch.mean(torch.stack(step_loss, dim=0), dim=0)
#             )
#             plot_loss = plot_loss + step_loss.detach().numpy()
#             loss = loss + torch.sum(step_loss)
#
#         # Scale by accumulate_steps so accumulated gradients represent an
#         # average across trials, not a SUM — keeps effective learning rate
#         # consistent with the non-accumulated case.
#         (loss / accumulate_steps).backward(retain_graph=True)
#         accum_counter += 1
#
#         if accum_counter >= accumulate_steps or is_last_walk:
#             adam.step()
#             adam.zero_grad()
#             accum_counter = 0
#
#         # Iteration.detach() (in model.py) covers L, M, g_gen, p_gen, x_gen,
#         # x_inf, g_inf, p_inf — but NOT x_logits. So we detach here for
#         # everything it covers, then explicitly .detach() x_logits wherever
#         # we touch it below.
#         for step in forward:
#             step.detach()
#
#         # --- x_gt_logits (explicit .detach()) ---
#         last_step = forward[-1]
#         logits_gt = last_step.x_logits[2].detach().numpy()[0]
#         logits_list.append({"iteration": i, "logits": logits_gt.tolist()})
#
#         # --- x_p_logits / x_g_logits (explicit .detach()) ---
#         logits_p = last_step.x_logits[0].detach().numpy()[0]
#         logits_g = last_step.x_logits[1].detach().numpy()[0]
#
#         # --- per-trial correct() for all 3 pathways ---
#         is_boundary = len(walk) == 1 and walk[0][2] == [None]
#         if not is_boundary:
#             trial_correct = np.mean(
#                 [[np.mean(a) for a in step.correct()] for step in forward], axis=0
#             )
#             per_trial_correct.append({
#                 "iteration": i,
#                 "correct_p": float(trial_correct[0]),
#                 "correct_g": float(trial_correct[1]),
#                 "correct_gt": float(trial_correct[2]),
#                 "logits_p": logits_p.tolist(),
#                 "logits_g": logits_g.tolist(),
#             })
#
#         # --- periodic representation + M snapshot, using a  dedicated
#         # diagnostic walk (full location coverage) branched off the current
#         # prev_iter, NOT the just-trained walk (which only visits 1-2
#         # locations and would leave the rest as zero-filled placeholders) ---
#         if save_representations is not None and (
#             i % save_representations == 0 or is_last_walk
#         ):
#             diag_walk = generate_diagnostic_walk(env, multiplier=100)
#             with torch.no_grad():
#                 diag_forward = tem_model(diag_walk, prev_iter)
#             g_snap, p_snap = analyse.rate_map(diag_forward, tem_model, [env])
#             representation_snapshots.append({
#                 "iteration": i,
#                 "g": [freq_mat.tolist() for freq_mat in g_snap[0]],
#                 "p": [freq_mat.tolist() for freq_mat in p_snap[0]],
#                 "M_generative": forward[-1].M[0].numpy().tolist(),
#             })
#
#         prev_iter = [forward[-1]]
#
#         if isinstance(plot_loss, np.int64):
#             plot_loss = None
#
#         acc_p, acc_g, acc_gt = np.mean(
#             [[np.mean(a) for a in step.correct()] for step in forward], axis=0
#         )
#         acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]
#
#         if i % log_interval == 0:
#             logger.info("Finished backprop iter {:d} in {:.2f} seconds.".format(i, time.time() - start_time))
#             if plot_loss is not None:
#                 logger.info(
#                     "Loss: {:.2f}. <p_g> {:.2f} <p_x> {:.2f} <x_gen> {:.2f} <x_g> {:.2f} <x_p> {:.2f} <g> {:.2f} <reg_g> {:.2f} <reg_p> {:.2f}".format(
#                         loss.detach().numpy(), *plot_loss
#                     )
#                 )
#             logger.info("Accuracy: <p> {:.2f}% <g> {:.2f}% <gt> {:.2f}%".format(acc_p, acc_g, acc_gt))
#             logger.info(
#                 "Parameters: <max_hebb> {:.2f} <eta> {:.2f} <lambda> {:.2f} <p2g_scale_offset> {:.2f}".format(
#                     np.max(np.abs(prev_iter[0].M[0].numpy())),
#                     tem_model.hyper["eta"], tem_model.hyper["lambda"], tem_model.hyper["p2g_scale_offset"],
#                 )
#             )
#             logger.info("Weights:" + str([w for w in loss_weights.numpy()]))
#             logger.info(" ")
#             writer.add_scalar("Losses/Total", loss.detach().numpy(), i)
#             if plot_loss is not None:
#                 writer.add_scalar("Losses/p_g", plot_loss[0], i)
#                 writer.add_scalar("Losses/p_x", plot_loss[1], i)
#                 writer.add_scalar("Losses/x_gen", plot_loss[2], i)
#                 writer.add_scalar("Losses/x_g", plot_loss[3], i)
#                 writer.add_scalar("Losses/x_p", plot_loss[4], i)
#                 writer.add_scalar("Losses/g", plot_loss[5], i)
#                 writer.add_scalar("Losses/reg_g", plot_loss[6], i)
#                 writer.add_scalar("Losses/reg_p", plot_loss[7], i)
#                 writer.add_scalar("Accuracies/p", acc_p, i)
#                 writer.add_scalar("Accuracies/g", acc_g, i)
#                 writer.add_scalar("Accuracies/gt", acc_gt, i)
#
#     writer.close()
#
#     return tem_model, adam, params, i, logits_list, representation_snapshots, per_trial_correct, prev_iter
#

def learn_walks(walks, env, tem_model, adam, params, out_dir, i, run=1,
                 save_representations=20, prev_iter=None,
                 memory_iteration_offset=0, schedule_index=None):
    """
    memory_iteration_offset: subtracted from the schedule position when
    computing eta/lambda (NOT lr/loss weights), so the memory-write schedule
    can be independently reset at a design transition. Default 0.

    schedule_index: optional list, same length as `walks`, giving each walk's
    position on the parameter schedule. When a block is repeated, every copy
    reuses the same positions, so eta/lambda/lr take the same values on a given
    trial regardless of how many times its block is repeated. Without this the
    schedules stretch with the number of repeats, which would change the
    learning conditions inside the very block being manipulated. Defaults to
    the walk's ordinal position, i.e. the original behaviour.

    `i` remains the global step counter used for logging and for TensorBoard,
    so it still increments once per weight update.
    """
    visited = [[False for _ in range(env.n_locations)]]

    str_dir = str(out_dir) + "/"
    writer = SummaryWriter(str_dir)
    logger = utils.make_logger(str_dir)
    log_interval = 1

    logits_list = []
    representation_snapshots = []
    per_trial_correct = []

    n_walks = len(walks)
    for walk_i, walk in enumerate(walks):
        i += 1
        sched = i if schedule_index is None else schedule_index[walk_i]
        sched_memory = sched - memory_iteration_offset
        is_last_walk = (walk_i == n_walks - 1)

        (eta_new, lambda_new, p2g_scale_offset, lr, walk_length_center,
         loss_weights) = parameter_iteration_decoupled(sched_memory, sched, params)
        start_time = time.time()
        tem_model.hyper["eta"] = eta_new
        tem_model.hyper["lambda"] = lambda_new
        tem_model.hyper["p2g_scale_offset"] = p2g_scale_offset
        for param_group in adam.param_groups:
            param_group["lr"] = lr

        forward = tem_model(walk, prev_iter)

        loss = torch.tensor(0.0, requires_grad=True)
        plot_loss = 0
        for step in forward:
            step_loss = []
            for env_i, env_visited in enumerate(visited):
                if env_visited[step.g[env_i]["id"]]:
                    step_loss.append(loss_weights * torch.stack([l[env_i] for l in step.L]))
                else:
                    env_visited[step.g[env_i]["id"]] = True
            step_loss = (
                torch.tensor(0) if not step_loss
                else torch.mean(torch.stack(step_loss, dim=0), dim=0)
            )
            plot_loss = plot_loss + step_loss.detach().numpy()
            loss = loss + torch.sum(step_loss)

        adam.zero_grad()
        loss.backward(retain_graph=True)
        adam.step()

        for step in forward:
            step.detach()

        last_step = forward[-1]
        logits_gt = last_step.x_logits[2].detach().numpy()[0]
        logits_list.append({"iteration": i, "logits": logits_gt.tolist()})

        logits_p = last_step.x_logits[0].detach().numpy()[0]
        logits_g = last_step.x_logits[1].detach().numpy()[0]

        is_boundary = len(walk) == 1 and walk[0][2] == [None]
        if not is_boundary:
            trial_correct = np.mean(
                [[np.mean(a) for a in step.correct()] for step in forward], axis=0
            )
            per_trial_correct.append({
                "iteration": i,
                "schedule_position": int(sched),
                "correct_p": float(trial_correct[0]),
                "correct_g": float(trial_correct[1]),
                "correct_gt": float(trial_correct[2]),
                "logits_p": logits_p.tolist(),
                "logits_g": logits_g.tolist(),
            })

        if save_representations is not None and (
            i % save_representations == 0 or is_last_walk
        ):
            diag_walk = generate_diagnostic_walk(env, multiplier=100)
            with torch.no_grad():
                diag_forward = tem_model(diag_walk, prev_iter)
            g_snap, p_snap = analyse.rate_map(diag_forward, tem_model, [env])
            representation_snapshots.append({
                "iteration": i,
                "g": [freq_mat.tolist() for freq_mat in g_snap[0]],
                "p": [freq_mat.tolist() for freq_mat in p_snap[0]],
                "M_generative": forward[-1].M[0].numpy().tolist(),
            })

        prev_iter = [forward[-1]]

        if isinstance(plot_loss, np.int64):
            plot_loss = None

        acc_p, acc_g, acc_gt = np.mean(
            [[np.mean(a) for a in step.correct()] for step in forward], axis=0
        )
        acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]

        if i % log_interval == 0:
            logger.info("Finished backprop iter {:d} (schedule {:d}) in {:.2f} seconds.".format(
                i, int(sched), time.time() - start_time))
            if plot_loss is not None:
                logger.info(
                    "Loss: {:.2f}. <p_g> {:.2f} <p_x> {:.2f} <x_gen> {:.2f} <x_g> {:.2f} <x_p> {:.2f} <g> {:.2f} <reg_g> {:.2f} <reg_p> {:.2f}".format(
                        loss.detach().numpy(), *plot_loss
                    )
                )
            logger.info("Accuracy: <p> {:.2f}% <g> {:.2f}% <gt> {:.2f}%".format(acc_p, acc_g, acc_gt))
            logger.info(
                "Parameters: <max_hebb> {:.2f} <eta> {:.2f} <lambda> {:.2f} <p2g_scale_offset> {:.2f}".format(
                    np.max(np.abs(prev_iter[0].M[0].numpy())),
                    tem_model.hyper["eta"], tem_model.hyper["lambda"], tem_model.hyper["p2g_scale_offset"],
                )
            )
            logger.info("Weights:" + str([w for w in loss_weights.numpy()]))
            logger.info(" ")
            writer.add_scalar("Losses/Total", loss.detach().numpy(), i)
            if plot_loss is not None:
                writer.add_scalar("Losses/p_g", plot_loss[0], i)
                writer.add_scalar("Losses/p_x", plot_loss[1], i)
                writer.add_scalar("Losses/x_gen", plot_loss[2], i)
                writer.add_scalar("Losses/x_g", plot_loss[3], i)
                writer.add_scalar("Losses/x_p", plot_loss[4], i)
                writer.add_scalar("Losses/g", plot_loss[5], i)
                writer.add_scalar("Losses/reg_g", plot_loss[6], i)
                writer.add_scalar("Losses/reg_p", plot_loss[7], i)
                writer.add_scalar("Accuracies/p", acc_p, i)
                writer.add_scalar("Accuracies/g", acc_g, i)
                writer.add_scalar("Accuracies/gt", acc_gt, i)

    writer.close()

    return tem_model, adam, params, i, logits_list, representation_snapshots, per_trial_correct, prev_iter


def learn_operators(env_files, design_files, out_dir, subject, run, override_file,
                     walks_multiplier=1, reset_iteration_at_transfer=False,
                     reset_perceptual_weights_at_transfer=True,
                     reset_perceptual_optimizer_state=False,
                     transition_init_scale=1.0, block_epochs=None,
                     save_representations=20):
    """
    block_epochs: dict mapping block number to how many times that block's
    trials are repeated, e.g. {2: 5, 4: 5}. Blocks not listed run once. The
    trial order inside a repeated block is unchanged, and no boundary marker is
    inserted between copies, matching how walks_multiplier repeats were handled
    in study-24.

    Repeating a block adds gradient updates without moving the parameter
    schedules: each copy reuses the same schedule positions as the first, so
    eta, lambda and the learning rate take the same value on a given trial
    however many times its block repeats.
    """
    designs = [pl.read_csv(file) for file in design_files]
    out_dir = Path(out_dir)
    params = parameters.parameters()
    with open(override_file) as f:
        params.update(json.load(f))
    tem_model = model.Model(params)
    if transition_init_scale != 0.0:
        reinit_transition_weights(tem_model, scale=transition_init_scale)
    adam = torch.optim.Adam(tem_model.parameters(), lr=params["lr_max"])
    i = 0
    schedule_base = 0

    for d, design in enumerate(designs):
        env = World(env_files[d], randomise_observations=True, shiny=None)

        design_out_dir = out_dir / f"design-{d}"
        design_out_dir.mkdir(parents=True, exist_ok=True)

        obs_mapping = {
            f"node_{loc['id'] + 1}": loc["observation"]
            for loc in env.locations
        }
        with open(design_out_dir / f"sub-{subject}_run-{run}_design-{d}_obs_mapping.json", "w") as f:
            json.dump(obs_mapping, f)

        actions = {"south": 1, "east": 2, "north": 3, "west": 4}
        walks, blocks = walks_operators(design, env, actions)

        # Schedule position of each walk in the unrepeated sequence.
        base_sched = [schedule_base + k + 1 for k in range(len(walks))]

        if block_epochs:
            expanded_walks, expanded_sched = [], []
            k = 0
            while k < len(walks):
                blk = blocks[k]
                j = k
                while j < len(walks) and blocks[j] == blk:
                    j += 1
                segment = walks[k:j]
                segment_sched = base_sched[k:j]
                for _ in range(block_epochs.get(blk, 1)):
                    expanded_walks.extend(segment)
                    expanded_sched.extend(segment_sched)
                k = j
            walks, schedule_index = expanded_walks, expanded_sched
        else:
            schedule_index = base_sched

        if walks_multiplier > 1:
            walks = walks * walks_multiplier
            schedule_index = schedule_index * walks_multiplier

        schedule_base += len(base_sched)

        if reset_perceptual_weights_at_transfer and d == 1:
            reset_perceptual_weights(tem_model, adam, reset_perceptual_optimizer_state)

        memory_offset = schedule_base - len(base_sched) if (
            reset_iteration_at_transfer and d == 1) else 0



        tem_model, adam, params, i, logits_list, representation_snapshots, per_trial_correct, _ = learn_walks(
            walks, env, tem_model, adam, params, design_out_dir, i, run,
            save_representations=save_representations,
            memory_iteration_offset=memory_offset,
            schedule_index=schedule_index,
        )

        pl.DataFrame(logits_list).write_parquet(
            design_out_dir / f"sub-{subject}_run-{run}_design-{d}_x_gt_logits.parquet"
        )

        import pickle
        with open(design_out_dir / f"sub-{subject}_run-{run}_design-{d}_representations.pkl", "wb") as f:
            pickle.dump(representation_snapshots, f)

        pl.DataFrame(per_trial_correct).write_parquet(
            design_out_dir / f"sub-{subject}_run-{run}_design-{d}_per_trial.parquet"
        )

        torch.save(
            tem_model.state_dict(),
            design_out_dir / f"sub-{subject}_run-{run}_design-{d}_tem.pt",
        )
        torch.save(
            tem_model.hyper,
            design_out_dir / f"sub-{subject}_run-{run}_design-{d}_params.pt",
        )
    return tem_model


# This is the tentative function for testing whether repeated exposure cause the g-representation to gradually converge?
def learn_operators_repeated(env_files, design_files, out_dir, subject, run, override_file,
                               n_repeats=3, save_representations=20):
    """
    Repeated-environment variant: alternates training between Initial (design 0)
    and Transfer (design 1) for n_repeats full passes each (Initial→Transfer→
    Initial→Transfer...), instead of training each design once sequentially.

    - obs_mapping is randomized ONCE per design (on first exposure) and reused
      for every subsequent repetition, so representations stay comparable.
    - M (and g_inf/x_inf) persist across every design switch and repetition —
      prev_iter is threaded continuously through the whole sequence.
    - tem_model/adam are created once and never recreated.
    """
    designs = [pl.read_csv(file) for file in design_files]
    out_dir = Path(out_dir)
    params = parameters.parameters()
    with open(override_file) as f:
        params.update(json.load(f))
    tem_model = model.Model(params)
    adam = torch.optim.Adam(tem_model.parameters(), lr=params["lr_max"])
    i = 0
    prev_iter = None

    actions = {"south": 1, "east": 2, "north": 3, "west": 4}

    envs = [World(env_files[d], randomise_observations=True, shiny=None) for d in range(len(designs))]

    for d, env in enumerate(envs):
        design_out_dir = out_dir / f"design-{d}"
        design_out_dir.mkdir(parents=True, exist_ok=True)
        obs_mapping = {f"node_{loc['id'] + 1}": loc["observation"] for loc in env.locations}
        with open(design_out_dir / f"sub-{subject}_run-{run}_design-{d}_obs_mapping.json", "w") as f:
            json.dump(obs_mapping, f)

    all_logits = {d: [] for d in range(len(designs))}
    all_representations = {d: [] for d in range(len(designs))}
    all_per_trial = {d: [] for d in range(len(designs))}

    for rep in range(n_repeats):
        for d, design in enumerate(designs):
            env = envs[d]
            design_out_dir = out_dir / f"design-{d}"
            walks, _ = walks_operators(design, env, actions)

            (tem_model, adam, params, i, logits_list, representation_snapshots,
             per_trial_correct, prev_iter) = learn_walks(
                walks, env, tem_model, adam, params, design_out_dir, i, run,
                save_representations=save_representations,
                prev_iter=prev_iter,
            )

            for entry in logits_list:
                entry["repeat"] = rep
            for entry in representation_snapshots:
                entry["repeat"] = rep
            for entry in per_trial_correct:
                entry["repeat"] = rep

            all_logits[d].extend(logits_list)
            all_representations[d].extend(representation_snapshots)
            all_per_trial[d].extend(per_trial_correct)

    import pickle
    for d in range(len(designs)):
        design_out_dir = out_dir / f"design-{d}"
        pl.DataFrame(all_logits[d]).write_parquet(
            design_out_dir / f"sub-{subject}_run-{run}_design-{d}_x_gt_logits_repeated.parquet"
        )
        with open(design_out_dir / f"sub-{subject}_run-{run}_design-{d}_representations_repeated.pkl", "wb") as f:
            pickle.dump(all_representations[d], f)
        pl.DataFrame(all_per_trial[d]).write_parquet(
            design_out_dir / f"sub-{subject}_run-{run}_design-{d}_per_trial_repeated.parquet"
        )

    torch.save(tem_model.state_dict(), out_dir / f"sub-{subject}_run-{run}_tem_final_repeated.pt")
    torch.save(tem_model.hyper, out_dir / f"sub-{subject}_run-{run}_params_repeated.pt")

    return tem_model


def generate_env(spec, n_obs, observations):
    """Generate an environment from a specification."""
    env = {
        "n_locations": spec["n_locations"],
        "n_observations": n_obs,
        "n_actions": spec["n_actions"],
        "adjacency": spec["adjacency"],
        "locations": [],
    }

    for i, loc in enumerate(spec["locations"]):
        dest = [a["dest"] for a in loc["actions"] if a["dest"] != "null"]
        actions = []
        for src in loc["actions"]:
            t = np.zeros(env["n_locations"], dtype=int)
            t = [int(i) for i in t]
            if src["dest"] != "null":
                t[src["dest"]] = 1
                p = 1 / len(dest)
            else:
                p = 0
            action = {"id": src["id"], "transition": t, "probability": p}
            actions.append(action)
        d = {
            "id": loc["id"],
            "observation": int(observations[i]),
            "x": loc["x"],
            "y": loc["y"],
            "in_locations": dest,
            "in_degree": len(dest),
            "out_locations": dest,
            "out_degree": len(dest),
            "actions": actions,
        }
        env["locations"].append(d)
    return env


class World:
    def __init__(
        self, env, randomise_observations=False, randomise_policy=False, shiny=None
    ):
        # If the environment is provided as a filename: load the corresponding file. If it's no filename, it's assumed to be an environment dictionary
        if type(env) == str or type(env) == np.str_:
            # Filename provided, load graph from json file
            file = open(env, "r")
            json_text = file.read()
            env = json.loads(json_text)
            file.close()

        # Now env holds a dictionary that describes this world
        try:
            # Copy expected fiels to object attributes
            self.adjacency = env["adjacency"]
            self.locations = env["locations"]
            self.n_actions = env["n_actions"]
            self.n_locations = env["n_locations"]
            self.n_observations = env["n_observations"]
        except (KeyError, TypeError) as e:
            # If any of the expected fields is missing: treat this as an invalid environment
            print("Invalid environment: bad dictionary\n", e)
            # Initialise all environment fields for an empty environment
            self.adjacency = []
            self.locations = []
            self.n_actions = 0
            self.n_locations = 0
            self.n_observations = 0

        # If requested: shuffle observations from original assignments
        if randomise_observations:
            self.observations_randomise()

        # If requested: randomise policy by setting equal probability for each action
        if randomise_policy:
            self.policy_random()

        # Copy the shiny input
        self.shiny = copy.deepcopy(shiny)
        # If there's no shiny data provided: initialise this world as a non-shiny environement
        if self.shiny is None:
            # TEM needs to know that this is a non-shiny environment (e.g. for providing actions to generative model), so set shiny to None for each location
            for location in self.locations:
                location["shiny"] = None
        # If shiny data is provided: initialise shiny properties
        else:
            # Initially make all locations non-shiny
            for location in self.locations:
                location["shiny"] = False
            # Calculate all graph distances, since shiny objects aren't allowed to be too close together
            dist_matrix = shortest_path(
                csgraph=np.array(self.adjacency), directed=False
            )
            # Initialise the list of shiny locations as empty
            self.shiny["locations"] = []
            # Then select shiny locations by adding them one-by-one, with the constraint that they can't be too close to each other
            while len(self.shiny["locations"]) < self.shiny["n"]:
                new = np.random.randint(self.n_locations)
                too_close = [
                    dist_matrix[new, existing] < np.max(dist_matrix) / self.shiny["n"]
                    for existing in self.shiny["locations"]
                ]
                if not any(too_close):
                    self.shiny["locations"].append(new)
            # Set those locaitons to be shiny
            for shiny_location in self.shiny["locations"]:
                self.locations[shiny_location]["shiny"] = True
            # Get objects at shiny locations
            self.shiny["objects"] = [
                self.locations[location]["observation"]
                for location in self.shiny["locations"]
            ]
            # Make list of objects that are not shiny
            not_shiny = [
                observation
                for observation in range(self.n_observations)
                if observation not in self.shiny["objects"]
            ]
            # Update observations so there is no non-shiny occurence of the shiny objects
            for location in self.locations:
                # Update a non-shiny location if it has a shiny object observation
                if (
                    location["id"] not in self.shiny["locations"]
                    and location["observation"] in self.shiny["objects"]
                ):
                    # Pick new observation from non-shiny objects
                    location["observation"] = np.random.choice(not_shiny)
            # Generate a policy towards each of the shiny objects
            self.shiny["policies"] = [
                self.policy_distance(shiny_location)
                for shiny_location in self.shiny["locations"]
            ]

    # def observations_randomise(self):
    #     # Run through every abstract location
    #     for location in self.locations:
    #         # Pick random observation from any of the observations
    #         location["observation"] = np.random.randint(self.n_observations)
    #     return self

    def observations_randomise(self):
        # Sample observations without replacement. The original TEM code used
        # np.random.randint per location, which allows two locations to share
        # the same observation. That is deliberate in original TEM above (sensory
        # aliasing forces the model to rely on structure), but it does not
        # match our task: the human design always assigns 6 distinct objects
        # to the 6 nodes. With 6 draws from 45 with replacement, ~29% of runs
        # contained at least one duplicated observation, which corrupts
        # memory-based g inference for the aliased pair.
        if self.n_locations <= self.n_observations:
            chosen = np.random.choice(
                self.n_observations, size=self.n_locations, replace=False
            )
        else:
            chosen = np.random.randint(self.n_observations, size=self.n_locations)
        for location, observation in zip(self.locations, chosen):
            location["observation"] = int(observation)
        return self

    def policy_random(self):
        # Run through every abstract location
        for location in self.locations:
            # Count the number of actions that can transition anywhere for this location
            count = sum(
                [sum(action["transition"]) > 0 for action in location["actions"]]
            )
            # Run through all actions at this location to update their probability
            for action in location["actions"]:
                # If this action transitions anywhere: it is an avaiable action, so set its probability to 1/count
                action["probability"] = (
                    1.0 / count if sum(action["transition"]) > 0 else 0
                )
        return self

    def policy_learned(self, reward_locations):
        # This generates a Q-learned policy towards reward locations.
        # Prepare new set of locations to hold policies towards reward locations
        new_locations, reward_locations = self.get_reward(reward_locations)
        # Initialise state-action values Q at 0
        for location in new_locations:
            for action in location["actions"]:
                action["Q"] = 0
        # Do value iteration in order to find a policy toward a given location
        iters = 10 * self.n_locations
        # Run value iterations by looping through all actions iteratively
        for i in range(iters):
            # Deepcopy the current Q-values so they are the same for all updates (don't update values that you later need)
            prev_locations = copy.deepcopy(new_locations)
            for location in new_locations:
                for action in location["actions"]:
                    # Q-value update from value iteration of Bellman equation: Q(s,a) <- sum_across_s'(p(s,a,s') * (r(s') + gamma * max_across_a'(Q(s', a'))))
                    action["Q"] = sum(
                        [
                            probability
                            * (
                                (new_location in reward_locations)
                                + self.shiny["gamma"]
                                * max(
                                    [
                                        new_action["Q"]
                                        for new_action in prev_locations[new_location][
                                            "actions"
                                        ]
                                    ]
                                )
                            )
                            for new_location, probability in enumerate(
                                action["transition"]
                            )
                        ]
                    )
        # Calculate policy from softmax over Q-values for every state
        for location in new_locations:
            exp = np.exp(
                self.shiny["beta"]
                * np.array(
                    [
                        action["Q"] if action["probability"] > 0 else -np.inf
                        for action in location["actions"]
                    ]
                )
            )
            for action, probability in zip(location["actions"], exp / sum(exp)):
                # Policy from softmax: p(a) = exp(beta*a)/sum_over_as(exp(beta*a_s))
                action["probability"] = probability
        # Return new locations with updated policy for given reward locations
        return new_locations

    def policy_distance(self, reward_locations):
        # This generates a distance-based policy towards reward locations, which is much faster than Q-learning but ignores policy and transition probabilities
        # Prepare new set of locations to hold policies towards reward locations
        new_locations, reward_locations = self.get_reward(reward_locations)
        # Create boolean vector of reward locations for matrix indexing
        is_reward_location = np.zeros(self.n_locations, dtype=bool)
        is_reward_location[reward_locations] = True
        # Calculate distances between all locations based on adjacency matrix - this doesn't take transition probabilities into account!
        dist_matrix = shortest_path(csgraph=np.array(self.adjacency), directed=True)
        # Fill out minumum distance to any reward state for each action
        for location in new_locations:
            for action in location["actions"]:
                action["d"] = (
                    np.min(
                        dist_matrix[
                            is_reward_location, np.array(action["transition"]) > 0
                        ]
                    )
                    if any(action["transition"])
                    else np.inf
                )
        # Calculate policy from softmax over negative distances for every action
        for location in new_locations:
            exp = np.exp(
                self.shiny["beta"]
                * np.array(
                    [
                        -action["d"] if action["probability"] > 0 else -np.inf
                        for action in location["actions"]
                    ]
                )
            )
            for action, probability in zip(location["actions"], exp / sum(exp)):
                # Policy from softmax: p(a) = exp(beta*a)/sum_over_as(exp(beta*a_s))
                action["probability"] = probability
        # Return new locations with updated policy for given reward locations
        return new_locations

    def generate_walks(self, walk_length=10, n_walk=100, repeat_bias_factor=2):
        # Generate walk by sampling actions accoring to policy, then next state according to graph
        walks = (
            []
        )  # This is going to contain a list of (state, observation, action) tuples
        for currWalk in range(n_walk):
            new_walk = []
            # If shiny hasn't been specified: there are no shiny objects, generate default policy
            if self.shiny is None:
                new_walk = self.walk_default(new_walk, walk_length, repeat_bias_factor)
            # If shiny was specified: use policy that uses shiny policy to approach shiny objects sequentially
            else:
                new_walk = self.walk_shiny(new_walk, walk_length, repeat_bias_factor)
            # Clean up walk a bit by only keep essential location dictionary entries
            for step in new_walk[:-1]:
                step[0] = {"id": step[0]["id"], "shiny": step[0]["shiny"]}
            # Append new walk to list of walks
            walks.append(new_walk)
        return walks

    def walk_default(self, walk, walk_length, repeat_bias_factor=2):
        # Finish the provided walk until it contains walk_length steps
        for curr_step in range(walk_length - len(walk)):
            # Get new location based on previous action and location
            new_location = self.get_location(walk)
            # Get new observation at new location
            new_observation = self.get_observation(new_location)
            # Get new action based on policy at new location
            new_action = self.get_action(new_location, walk)
            # Append location, observation, and action to the walk
            walk.append([new_location, new_observation, new_action])
        # Return the final walk
        return walk

    def walk_shiny(self, walk, walk_length, repeat_bias_factor=2):
        # Pick current shiny object to approach
        shiny_current = np.random.randint(self.shiny["n"])
        # Reset number of iterations to hang around an object once found
        shiny_returns = self.shiny["returns"]
        # Finish the provided walk until it contains walk_length steps
        for curr_step in range(walk_length - len(walk)):
            # Get new location based on previous action and location
            new_location = self.get_location(walk)
            # Check if the shiny object was found in this step
            if new_location["id"] == self.shiny["locations"][shiny_current]:
                # After shiny object is found, start counting down for hanging around
                shiny_returns -= 1
            # Check if it's time to select new object to approach
            if shiny_returns < 0:
                # Pick new current shiny object to approach
                shiny_current = np.random.randint(self.shiny["n"])
                # Reset number of iterations to hang around an object once found
                shiny_returns = self.shiny["returns"]
            # Get new observation at new location
            new_observation = self.get_observation(new_location)
            # Get new action based on policy of new location towards shiny object
            new_action = self.get_action(
                self.shiny["policies"][shiny_current][new_location["id"]], walk
            )
            # Append location, observation, and action to the walk
            walk.append([new_location, new_observation, new_action])
        # Return the final walk
        return walk

    def get_location(self, walk):
        # First step: start at random location
        if len(walk) == 0:
            new_location = np.random.randint(self.n_locations)
        # Any other step: get new location from previous location and action
        else:
            new_location = int(
                np.flatnonzero(
                    np.cumsum(walk[-1][0]["actions"][walk[-1][2]]["transition"])
                    > np.random.rand()
                )[0]
            )
        # Return the location dictionary of the new location
        return self.locations[new_location]

    def get_observation(self, new_location):
        # Find sensory observation for new state, and store it as one-hot vector
        new_observation = np.eye(self.n_observations)[new_location["observation"]]
        # Create a new observation by converting the new observation to a torch tensor
        new_observation = torch.tensor(new_observation, dtype=torch.float).view(
            (new_observation.shape[0])
        )
        # Return the new observation
        return new_observation

    def get_action(self, new_location, walk, repeat_bias_factor=2):
        # Build policy from action probability of each action of provided location dictionary
        policy = np.array([action["probability"] for action in new_location["actions"]])
        # Add a bias for repeating previous action to walk in straight lines, only if (this is not the first step) and (the previous action was a move)
        policy[
            (
                []
                if len(walk) == 0 or new_location["id"] == walk[-1][0]["id"]
                else walk[-1][2]
            )
        ] *= repeat_bias_factor
        # And renormalise policy (note that for unavailable actions, the policy was 0 and remains 0, so in that case no renormalisation needed)
        policy = policy / sum(policy) if sum(policy) > 0 else policy
        # Select action in new state
        new_action = int(np.flatnonzero(np.cumsum(policy) > np.random.rand())[0])
        # Return the new action
        return new_action

    def get_reward(self, reward_locations):
        # Stick reward location into a list if there is only one reward location. Use multiple reward locations simultaneously for e.g. wall attraction
        reward_locations = (
            [reward_locations]
            if type(reward_locations) is not list
            else reward_locations
        )
        # Copy locations for updated policy towards goal
        new_locations = copy.deepcopy(self.locations)
        # Disable self-actions at reward locations because they will be very attractive
        for reward_location in reward_locations:
            # Check for each action if it's a self-action
            for action in new_locations[reward_location]["actions"]:
                if action["transition"][reward_location] == 1:
                    action["probability"] = 0
            # Count total action probability to renormalise after disabling self-action
            total_probability = sum(
                [
                    action["probability"]
                    for action in new_locations[reward_location]["actions"]
                ]
            )
            # Renormalise action probabilities
            for action in new_locations[reward_location]["actions"]:
                action["probability"] = (
                    action["probability"] / total_probability
                    if total_probability > 0
                    else action["probability"]
                )
        return new_locations, reward_locations
