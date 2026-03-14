#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:33:06 2020

@author: jacobb
"""
import json
import numpy as np
import torch
import copy
from scipy.sparse.csgraph import shortest_path
from tem import model, parameters
import polars as pl
from pathlib import Path
import time
from tem import utils
from torch.utils.tensorboard import SummaryWriter


# Functions for generating data that TEM trains on: sequences of [state,observation,action] tuples
def design_walks(design, env, actions):
    """Create walks from a learning phase design."""
    walks = []
    nodes = [f"node_{n}" for n in range(1, 7)]
    for row in design.iter_rows(named=True):
        steps = []
        if row["trial_type"] == "integration":
            # start node (only applies in two-step trials)
            start_ind = nodes.index(row["start_node"])
            start_obs = env.get_observation(env.locations[start_ind])
            steps.append(
                [
                    [{"id": start_ind, "shiny": None}],
                    [start_obs],
                    [actions[row["move_direction"]]],
                ]
            )

        # cue node
        cue_ind = nodes.index(row["cue_node"])
        cue_obs = env.get_observation(env.locations[cue_ind])
        steps.append(
            [
                [{"id": cue_ind, "shiny": None}],
                [cue_obs],
                [actions[row["direction"]]],
            ]
        )

        # target node
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
    return walks


def learn_walks(walks, env, tem_model, adam, params, out_dir, i):
    """Learn a series of walks through an environment."""
    visited = [[False for _ in range(env.n_locations)]]
    prev_iter = None

    # Create a tensor board to stay updated on training progress. Start tensorboard with tensorboard --logdir=runs
    str_dir = str(out_dir) + '/'
    writer = SummaryWriter(str_dir)
    # Create a logger to write log output to file
    logger = utils.make_logger(str_dir)
    log_interval = 1
    for walk in walks:
        i += 1
        # Get updated parameters for this backprop iteration
        (
            eta_new, lambda_new, p2g_scale_offset, lr, walk_length_center, loss_weights
        ) = parameters.parameter_iteration(i, params)
        # Get start time for function timing
        start_time = time.time()
        # Update eta and lambda
        tem_model.hyper['eta'] = eta_new
        tem_model.hyper['lambda'] = lambda_new
        # Update scaling of offset for variance of inferred grounded position
        tem_model.hyper['p2g_scale_offset'] = p2g_scale_offset
        # Update learning rate (the neater torch-way of doing this would be a scheduler, but this is quick and easy)
        for param_group in adam.param_groups:
            param_group['lr'] = lr

        # Forward-pass this walk through the network
        forward = tem_model(walk, prev_iter)

        # Accumulate loss from forward pass
        loss = torch.tensor(0.0, requires_grad=True)
        # Collect all losses
        plot_loss = 0
        for step in forward:
            # Make list of losses included in this step
            step_loss = []
            # Only include loss for locations that have been visited before
            for env_i, env_visited in enumerate(visited):
                if env_visited[step.g[env_i]['id']]:
                    step_loss.append(loss_weights * torch.stack([l[env_i] for l in step.L]))
                else:
                    env_visited[step.g[env_i]['id']] = True
            # Stack losses in this step along first dimension, then average across that dimension to get mean loss for this step
            step_loss = torch.tensor(0) if not step_loss else torch.mean(torch.stack(step_loss, dim=0), dim=0)
            # Save all separate components of loss for monitoring
            plot_loss = plot_loss + step_loss.detach().numpy()
            # And sum all components, then add them to total loss of this step
            loss = loss + torch.sum(step_loss)

        # Reset gradients
        adam.zero_grad()
        # Do backward pass to calculate gradients with respect to total loss of this chunk
        loss.backward(retain_graph=True)
        # Then do optimiser step to update parameters of model
        adam.step()
        # Update the previous iteration for the next chunk with the final step of this chunk, removing all operation history
        prev_iter = [forward[-1].detach()]

        if isinstance(plot_loss, np.int64):
            plot_loss = None

        # Compute model accuracies
        acc_p, acc_g, acc_gt = np.mean([[np.mean(a) for a in step.correct()] for step in forward], axis=0)
        acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]

        # Log progress
        if i % log_interval == 0:
            # Write series of messages to logger from this backprop iteration
            logger.info('Finished backprop iter {:d} in {:.2f} seconds.'.format(i, time.time() - start_time))
            if plot_loss is not None:
                logger.info(
                    'Loss: {:.2f}. <p_g> {:.2f} <p_x> {:.2f} <x_gen> {:.2f} <x_g> {:.2f} <x_p> {:.2f} <g> {:.2f} <reg_g> {:.2f} <reg_p> {:.2f}'.format(
                        loss.detach().numpy(), *plot_loss))
            logger.info('Accuracy: <p> {:.2f}% <g> {:.2f}% <gt> {:.2f}%'.format(acc_p, acc_g, acc_gt))
            logger.info('Parameters: <max_hebb> {:.2f} <eta> {:.2f} <lambda> {:.2f} <p2g_scale_offset> {:.2f}'.format(
                np.max(np.abs(prev_iter[0].M[0].numpy())), tem_model.hyper['eta'], tem_model.hyper['lambda'],
                tem_model.hyper['p2g_scale_offset']))
            logger.info('Weights:' + str([w for w in loss_weights.numpy()]))
            logger.info(' ')
            # Also write progress to tensorboard, and all loss components. Order: [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p]
            writer.add_scalar('Losses/Total', loss.detach().numpy(), i)
            if plot_loss is not None:
                writer.add_scalar('Losses/p_g', plot_loss[0], i)
                writer.add_scalar('Losses/p_x', plot_loss[1], i)
                writer.add_scalar('Losses/x_gen', plot_loss[2], i)
                writer.add_scalar('Losses/x_g', plot_loss[3], i)
                writer.add_scalar('Losses/x_p', plot_loss[4], i)
                writer.add_scalar('Losses/g', plot_loss[5], i)
                writer.add_scalar('Losses/reg_g', plot_loss[6], i)
                writer.add_scalar('Losses/reg_p', plot_loss[7], i)
                writer.add_scalar('Accuracies/p', acc_p, i)
                writer.add_scalar('Accuracies/g', acc_g, i)
                writer.add_scalar('Accuracies/gt', acc_gt, i)
    return tem_model, adam, params, i


def learn_design(env_files, design_files, out_dir, run):
    """Perform learning of multiple designs."""
    designs = [pl.read_csv(file) for file in design_files]
    out_dir = Path(out_dir)
    params = parameters.parameters()
    tem_model = model.Model(params)
    adam = torch.optim.Adam(tem_model.parameters(), lr=params['lr_max'])
    i = 0  # iteration counter
    for d, design in enumerate(designs):
        env = World(env_files[d], randomise_observations=True, shiny=None)
        actions = {"south": 1, "east": 2, "north": 3, "west": 4}
        walks = design_walks(design, env, actions)
        design_out_dir = out_dir / f"design-{d}"
        tem_model, adam, params, i = learn_walks(walks, env, tem_model, adam, params, design_out_dir, i)
        torch.save(tem_model.state_dict(), design_out_dir / f"run-{run}_design-{d}_tem.pt")
        torch.save(tem_model.hyper, design_out_dir / f"run-{run}_design-{d}_params.pt")
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

    def observations_randomise(self):
        # Run through every abstract location
        for location in self.locations:
            # Pick random observation from any of the observations
            location["observation"] = np.random.randint(self.n_observations)
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
