#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:35:30 2020

@author: jacobb
"""

# Functions for plotting training and results of TEM

# Standard library imports
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import polars as pl
import seaborn as sns
from pathlib import Path

def plot_weights(models, params=None, steps=None, do_save=False):
    # If no parameter names specified: just take all of the trained ones from the model
    if params is None:
        params = [
            item[0]
            for item in models[0].named_parameters().items()
            if item[1].requires_grad
        ]
    # If no steps specified: just make them increase by 1 for each model
    if steps is None:
        steps = [i for i in range(len(models))]
    # Collect this parameter in each model as provided
    model_dicts = [
        {model_params[0]: model_params[1] for model_params in model.named_parameters()}
        for model in models
    ]
    # Plot each parameter separately
    for param in params:
        # Create figure and subplots
        fig, axs = plt.subplots(2, len(steps))
        # Set it's size to something that is stretched horizontally so you can read titles
        fig.set_size_inches(10, 4)
        # Figure overall title is the parameter name
        fig.suptitle(param)
        values = [model_params[param].detach().numpy() for model_params in model_dicts]
        # On the first line of this figure: plot params at each step
        for i, step in enumerate(steps):
            # Plot variable values in subplot
            axs[0, i].imshow(values[i])
            axs[0, i].set_title("Step " + str(step))
        # On the second line of this figure: plot change in params between steps
        for i in range(len(steps) - 1):
            # Plot the change in variables
            axs[1, i].imshow(values[i + 1] - values[i])
            axs[1, i].set_title(
                str(steps[i])
                + " to "
                + str(steps[i + 1])
                + ", "
                + "{:.2E}".format(
                    np.mean(np.abs(values[i + 1] - values[i]))
                    / (steps[i + 1] - steps[i])
                )
            )
        # On the very last axis: plot the total difference between the first and the last
        axs[1, -1].imshow(values[-1] - values[0])
        axs[1, -1].set_title(
            str(steps[0])
            + " to "
            + str(steps[-1])
            + ", "
            + "{:.2E}".format(np.mean(np.abs(values[-1] - values[0])))
        )
        # If you want to save this figure: do so
        if do_save:
            fig.savefig("./figs/plot_weights_" + param + ".png")


def plot_memory(iters, steps=None, do_save=False):
    # If no steps specified: just make them increase by 1 for each model
    if steps is None:
        steps = [i for i in range(len(iters))]
    # Set names of memory: inference and generative
    names = ["Generative", "Inference"]
    # Plot each parameter separately
    for mem in range(len(iters[0].M)):
        # Get current memory name
        name = names[mem]
        # Create figure and subplots
        fig, axs = plt.subplots(len(iters[0].M[0]), len(steps))
        # Set it's size to something that is stretched horizontally so you can read titles
        fig.set_size_inches(len(steps) * 2, len(iters[0].M[0]))
        # Figure overall title is the parameter name
        fig.suptitle(name + " memory")
        # Load the memory matrices - first on in each batch
        batches = [iteration.M[mem] for iteration in iters]
        # On the first line of this figure: plot params at each step
        for col, step in enumerate(steps):
            for row, batch in enumerate(batches[col]):
                if len(steps) == 1:
                    # Plot variable values in subplot
                    axs[row].imshow(batch.numpy())
                    axs[row].set_title("Step " + str(step) + ", batch " + str(row))
                else:
                    # Plot variable values in subplot
                    axs[row, col].imshow(batch.numpy())
                    axs[row, col].set_title("Step " + str(step) + ", batch " + str(row))
        # If you want to save this figure: do so
        if do_save:
            fig.savefig("./figs/plot_mem_" + name + ".png")


def plot_map(
    environment,
    values,
    ax=None,
    min_val=None,
    max_val=None,
    num_cols=100,
    location_cm="viridis",
    action_cm="Pastel1",
    do_plot_actions=False,
    shape="circle",
    radius=None,
):
    # If min_val and max_val are not specified: take the minimum and maximum of the supplied values
    min_val = np.min(values) if min_val is None else min_val
    max_val = np.max(values) if max_val is None else max_val
    # Create color map for locations: colour given by value input
    location_cm = cm.get_cmap(location_cm, num_cols)
    # Create color map for actions: colour given by action index
    action_cm = cm.get_cmap(action_cm, environment.n_actions)
    # Calculate colour corresponding to each value
    plotvals = (
        np.floor((values - min_val) / (max_val - min_val) * num_cols)
        if max_val != min_val
        else np.ones(values.shape)
    )
    # Calculate radius of location circles based on how many nodes there are
    radius = (
        2 * (0.01 + 1 / (10 * np.sqrt(environment.n_locations)))
        if radius is None
        else radius
    )
    # Initialise empty axis
    ax = initialise_axes(ax)
    # Create empty list of location patches and action patches
    location_patches, action_patches = [], []
    # Now start drawing locations and actions
    for i, location in enumerate(environment.locations):
        # Create patch for location
        location_patches.append(
            plt.Rectangle(
                (location["x"] - radius / 2, location["y"] - radius / 2),
                radius,
                radius,
                color=location_cm(int(plotvals[i])),
            )
            if shape == "square"
            else plt.Circle(
                (location["x"], location["y"]),
                radius,
                color=location_cm(int(plotvals[i])),
            )
        )
        # And create action patches, if action plotting is switched on
        if do_plot_actions:
            for a, action in enumerate(location["actions"]):
                # Only draw patch if action probability is larger than 0
                if action["probability"] > 0:
                    # Find where this action takes you
                    locations_to = [
                        environment.locations[loc_to]
                        for loc_to in np.where(np.array(action["transition"]) > 0)[0]
                    ]
                    # Create an action patch for each possible transition for this action
                    for loc_to in locations_to:
                        action_patches.append(
                            action_patch(
                                location, loc_to, radius, action_cm(action["id"])
                            )
                        )
    # After drawing all locations, add shiny patches
    for location in environment.locations:
        # For shiny locations, add big red patch to indicate shiny
        if location["shiny"]:
            # Create square patch for location
            location_patches.append(
                plt.Rectangle(
                    (location["x"] - radius / 2, location["y"] - radius / 2),
                    radius,
                    radius,
                    linewidth=1,
                    facecolor="none",
                    edgecolor=[1, 0, 0],
                )
                if shape == "square"
                else plt.Circle(
                    (location["x"], location["y"]),
                    radius,
                    linewidth=1,
                    facecolor="none",
                    edgecolor=[1, 0, 0],
                )
            )
    # Add patches to axes
    for patch in location_patches + action_patches:
        ax.add_patch(patch)
    # Return axes for further use
    return ax


def plot_actions(
    environment,
    field="probability",
    ax=None,
    min_val=None,
    max_val=None,
    num_cols=100,
    action_cm="viridis",
):
    # If min_val and max_val are not specified: take the minimum and maximum of the supplied values
    min_val = (
        min(
            [
                action[field]
                for location in environment.locations
                for action in location["actions"]
            ]
        )
        if min_val is None
        else min_val
    )
    max_val = (
        max(
            [
                action[field]
                for location in environment.locations
                for action in location["actions"]
            ]
        )
        if max_val is None
        else max_val
    )
    # Create color map for locations: colour given by value input
    action_cm = cm.get_cmap(action_cm, num_cols)
    # Calculate radius of location circles based on how many nodes there are
    radius = 2 * (0.01 + 1 / (10 * np.sqrt(environment.n_locations)))
    # Initialise empty axis
    ax = initialise_axes(ax)
    # Create empty list of location patches and action patches
    location_patches, action_patches = [], []
    # Now start drawing locations and actions
    for i, location in enumerate(environment.locations):
        # Create circle patch for location
        location_patches.append(
            plt.Circle((location["x"], location["y"]), radius, color=[0, 0, 0])
        )
        # And create action patches
        for a, action in enumerate(location["actions"]):
            # Only draw patch if action probability is larger than 0
            if action["probability"] > 0:
                # Calculate colour for this action from colour map
                action_colour = action_cm(
                    int(
                        np.floor(
                            (action[field] - min_val) / (max_val - min_val) * num_cols
                        )
                    )
                )
                # Find where this action takes you
                locations_to = [
                    environment.locations[loc_to]
                    for loc_to in np.where(np.array(action["transition"]) > 0)[0]
                ]
                # Create an action patch for each possible transition for this action
                for loc_to in locations_to:
                    action_patches.append(
                        action_patch(location, loc_to, radius, action_colour)
                    )
    # Add patches to axes
    for patch in location_patches + action_patches:
        ax.add_patch(patch)
    # Return axes for further use
    return ax


def plot_walk(environment, walk, max_steps=None, n_steps=1, ax=None):
    # Set maximum number of steps if not provided
    max_steps = len(walk) if max_steps is None else min(max_steps, len(walk))
    # Initialise empty axis if axis wasn't provided
    if ax is None:
        ax = initialise_axes(ax)
    # Find all circle patches on current axis
    location_patches = [
        patch_i
        for patch_i, patch in enumerate(ax.patches)
        if type(patch) is plt.Circle or type(patch) is plt.Rectangle
    ]
    # Get radius of location circles on this map
    radius = (
        (
            ax.patches[location_patches[-1]].get_radius()
            if type(ax.patches[location_patches[-1]]) is plt.Circle
            else ax.patches[location_patches[-1]].get_width()
        )
        if len(location_patches) > 0
        else 0.02
    )
    # Initialise previous location: location of first location
    prev_loc = np.array(
        [
            environment.locations[walk[0][0]["id"]]["x"],
            environment.locations[walk[0][0]["id"]]["y"],
        ]
    )
    # Run through walk, creating lines
    for step_i in range(1, max_steps, n_steps):
        # Get location of current location, with some jitter so lines don't overlap
        new_loc = np.array(
            [
                environment.locations[walk[step_i][0]["id"]]["x"],
                environment.locations[walk[step_i][0]["id"]]["y"],
            ]
        )
        # Add jitter (need to unpack shape for rand - annoyingly np.random.rand takes dimensions separately)
        new_loc = new_loc + 0.8 * (
            -radius + 2 * radius * np.random.rand(*new_loc.shape)
        )
        # Plot line from previous location to current location
        plt.plot(
            [prev_loc[0], new_loc[0]],
            [prev_loc[1], new_loc[1]],
            color=[step_i / max_steps for _ in range(3)],
        )
        # Update new location to previous location
        prev_loc = new_loc
    # Return axes that this was plotted on
    return ax


def plot_cells(p, g, environment, n_f_ovc=0, columns=10):
    # Run through all hippocampal and entorhinal rate maps, big nested arrays arranged as [frequency][location][cell]
    for cells, names in zip([p, g], ["Hippocampal", "Entorhinal"]):
        # Calculate the number of rows that each frequency module requires
        n_rows_f = np.cumsum(
            [0] + [np.ceil(len(c[0]) * 1.0 / columns) for c in cells]
        ).astype(int)
        # Create subplots for cells across frequencies
        fig, ax = plt.subplots(nrows=n_rows_f[-1], ncols=columns)
        # Switch all axes off
        for row in ax:
            for col in row:
                col.axis("off")
        # And run through all frequencies to plot cells for that frequency
        for f, loc_rates in enumerate(cells):
            # Set title for current axis
            ax[n_rows_f[f], int(columns / 2)].set_title(
                names
                + ("" if f < len(cells) - n_f_ovc else " object vector ")
                + " cells, frequency "
                + str(f if f < len(cells) - n_f_ovc else f - (len(cells) - n_f_ovc))
            )
            # Plot map for each cell
            for c in range(len(loc_rates[0])):
                # Get current row and column
                row = int(n_rows_f[f] + np.floor(c / columns))
                col = int(c % columns)
                # Plot rate map for this cell by collection firing rate at each location
                plot_map(
                    environment,
                    np.array([loc_rates[l][c] for l in range(len(loc_rates))]),
                    ax[row, col],
                    shape="square",
                    radius=1 / np.sqrt(len(loc_rates)),
                )


def initialise_axes(ax=None):
    # If no axes specified: create new figure with new empty axes
    if ax is None:
        plt.figure()
        ax = plt.axes()
    # Set axes limits to 0, 1 as this is how the positions in the environment are setup
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # Force axes to be square to keep proper aspect ratio
    ax.set_aspect(1)
    # Revert y-axes so y position increases downwards (as it usually does in graphics/pixels)
    ax.invert_yaxis()
    # And don't show any axes
    ax.axis("off")
    # Return axes object
    return ax


def action_patch(location_from, location_to, radius, colour):
    # Set patch coordinates
    if location_to["id"] == location_from["id"]:
        # If this is a transition to self: action will point down (y-axis is reversed so pi/2 degrees is up)
        a_dir = np.pi / 2
        # Set the patch coordinates to point from this location to transition location (but shifted upward for self transition)
        xdat = location_from["x"] + radius * np.array(
            [
                2 * np.cos((a_dir - np.pi / 6)),
                2 * np.cos((a_dir + np.pi / 6)),
                3 * np.cos((a_dir)),
            ]
        )
        ydat = (
            location_from["y"]
            - radius * 3
            + radius
            * np.array(
                [
                    2 * np.sin((a_dir - np.pi / 6)),
                    2 * np.sin((a_dir + np.pi / 6)),
                    3 * np.sin((a_dir)),
                ]
            )
        )
    else:
        # This is not a transition to self. Find out the direction between current location and transitioned location
        xvec = location_to["x"] - location_from["x"]
        yvec = location_from["y"] - location_to["y"]
        a_dir = np.arctan2(xvec * 0 - yvec * 1, xvec * 1 + yvec * 0)
        # Set the patch coordinates to point from this location to transition location
        xdat = location_from["x"] + radius * np.array(
            [
                2 * np.cos((a_dir - np.pi / 6)),
                2 * np.cos((a_dir + np.pi / 6)),
                3 * np.cos((a_dir)),
            ]
        )
        ydat = location_from["y"] + radius * np.array(
            [
                2 * np.sin((a_dir - np.pi / 6)),
                2 * np.sin((a_dir + np.pi / 6)),
                3 * np.sin((a_dir)),
            ]
        )
    # Return action patch for provided data
    return plt.Polygon(np.stack([xdat, ydat], axis=1), color=colour)


def load_grid_results(sim_dir, study_prefix="study-grid-", tag="Accuracies/g", peak_fraction=0.1):
    """Load and combine aggregated parquet files from all grid search studies.

    Reads each study folder matching study_prefix, parses lambda and eta from
    the folder name, and returns a single combined dataframe with parameter
    labels attached.

    Parameters
    ----------
    sim_dir : Path
        Path to the simulation directory containing study folders.
    study_prefix : str
        Prefix used to identify grid search study folders.
    tag : str
        TensorBoard tag to filter (e.g. "Accuracies/g" for structural accuracy).
    peak_fraction : float
        Fraction of total trials at the end of training used to compute peak
        accuracy (e.g. 0.1 = last 10% of trials).

    Returns
    -------
    df_curves : polars.DataFrame
        Full trial-by-trial data for plotting learning curves, with columns:
        Trial, value, Graph, condition, lambda_val, eta_val, param_label.
    df_peaks : polars.DataFrame
        One row per study with peak accuracy summary, with columns:
        lambda_val, eta_val, Graph, condition, peak_accuracy.
    """
    sim_dir = Path(sim_dir)
    curve_dfs = []
    peak_rows = []

    for study_dir in sorted(sim_dir.glob(f"{study_prefix}*")):
        parquet_path = study_dir / "aggregated.parquet"
        if not parquet_path.exists():
            continue

        # Parse lambda and eta from folder name, e.g. "study-grid-l0_99-e0_5"
        # naming convention: underscores replace decimal points, so "l0_99" -> 0.99
        name = study_dir.name.replace(study_prefix, "")
        parts = name.split("-")
        try:
            lambda_str = [p for p in parts if p.startswith("l")][0][1:]
            eta_str = [p for p in parts if p.startswith("e")][0][1:]
            lambda_val = float(lambda_str.replace("_", "."))
            eta_val = float(eta_str.replace("_", "."))
        except (IndexError, ValueError):
            continue

        df = pl.read_parquet(parquet_path)
        df = (
            df.filter(pl.col("tag") == tag)
            .with_columns(
                pl.col("step").cum_count().over("subject", "run", "design").alias("Trial"),
                pl.col("design").replace({"0": "Initial", "1": "Transfer"}).alias("Graph"),
                pl.lit(lambda_val).alias("lambda_val"),
                pl.lit(eta_val).alias("eta_val"),
                pl.lit(f"λ={lambda_val}, η={eta_val}").alias("param_label"),
            )
        )
        curve_dfs.append(df)

        # Compute peak accuracy using last peak_fraction of trials
        max_trial = df["Trial"].max()
        trial_min = int(max_trial * (1 - peak_fraction))
        peak = (
            df.filter(pl.col("Trial") >= trial_min)
            .group_by("Graph", "condition", "lambda_val", "eta_val")
            .agg(pl.col("value").mean().alias("peak_accuracy"))
        )
        peak_rows.append(peak)

    df_curves = pl.concat(curve_dfs) if curve_dfs else pl.DataFrame()
    df_peaks = pl.concat(peak_rows) if peak_rows else pl.DataFrame()

    return df_curves, df_peaks


def plot_learning_curves(df_curves, graph="Transfer", condition=None,
                         window_size=200, ax=None, do_save=False, save_path=None):
    """Plot learning curves for multiple parameter combinations on one axis.

    Each line represents one (lambda, eta) combination. Lines are averaged
    across subjects and runs, with a shaded confidence interval.

    Parameters
    ----------
    df_curves : polars.DataFrame
        Output from load_grid_results().
    graph : str
        "Initial" or "Transfer" — which graph to plot.
    condition : str or None
        "PI", "AL", or None to plot both conditions together (averaged).
    window_size : int
        Rolling mean window for smoothing.
    ax : matplotlib axis or None
        Axis to plot on. Creates new figure if None.
    do_save : bool
        Whether to save the figure.
    save_path : Path or None
        Path to save the figure to if do_save is True.

    Returns
    -------
    ax : matplotlib axis
    """


    filtered = df_curves.filter(pl.col("Graph") == graph)
    if condition is not None:
        filtered = filtered.filter(pl.col("condition") == condition)

    # Apply rolling mean per subject/run/design/param combination
    smoothed = filtered.with_columns(
        pl.col("value")
        .rolling_mean(window_size=window_size)
        .over("subject", "run", "design", "param_label")
    )

    df_pd = smoothed.to_pandas()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    sns.lineplot(
        data=df_pd,
        x="Trial",
        y="value",
        hue="param_label",
        ax=ax,
        errorbar="se",
    )
    ax.set_ylabel("Structural accuracy")
    ax.set_title(f"Learning curves: {graph} map" + (f", {condition} group" if condition else ""))
    ax.legend(title="Parameters", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)

    if do_save and save_path is not None:
        ax.get_figure().savefig(save_path, bbox_inches="tight")

    return ax


def plot_param_heatmap(df_peaks, graph="Transfer", condition="PI",
                       ax=None, do_save=False, save_path=None):
    """Plot a heatmap of peak structural accuracy across lambda and eta values.

    Each cell shows the mean accuracy in the peak trial window for that
    (lambda, eta) combination.

    Parameters
    ----------
    df_peaks : polars.DataFrame
        Output from load_grid_results().
    graph : str
        "Initial" or "Transfer".
    condition : str
        "PI" or "AL".
    ax : matplotlib axis or None
        Axis to plot on. Creates new figure if None.
    do_save : bool
        Whether to save the figure.
    save_path : Path or None
        Path to save the figure to if do_save is True.

    Returns
    -------
    ax : matplotlib axis
    """

    filtered = (
        df_peaks
        .filter((pl.col("Graph") == graph) & (pl.col("condition") == condition))
        .to_pandas()
    )

    # Pivot into matrix form for heatmap: rows=lambda, cols=eta
    pivot = filtered.pivot_table(
        index="lambda_val",
        columns="eta_val",
        values="peak_accuracy",
        aggfunc="mean",
    )

    # Sort so highest lambda is at top
    pivot = pivot.sort_index(ascending=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Peak structural accuracy (%)"},
    )
    ax.set_xlabel("eta (η)")
    ax.set_ylabel("lambda (λ)")
    ax.set_title(f"Peak accuracy: {graph} map, {condition} group")

    if do_save and save_path is not None:
        ax.get_figure().savefig(save_path, bbox_inches="tight")

    return ax


## Just for convenience: all parameters in TEM
# for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)
"""
w_x
b_x
w_p.0
w_p.1
w_p.2
w_p.3
w_p.4
MLP_D_a.w.0.0.weight
MLP_D_a.w.0.0.bias
MLP_D_a.w.0.1.weight
MLP_D_a.w.0.1.bias
MLP_D_a.w.1.0.weight
MLP_D_a.w.1.0.bias
MLP_D_a.w.1.1.weight
MLP_D_a.w.1.1.bias
MLP_D_a.w.2.0.weight
MLP_D_a.w.2.0.bias
MLP_D_a.w.2.1.weight
MLP_D_a.w.2.1.bias
MLP_D_a.w.3.0.weight
MLP_D_a.w.3.0.bias
MLP_D_a.w.3.1.weight
MLP_D_a.w.3.1.bias
MLP_D_a.w.4.0.weight
MLP_D_a.w.4.0.bias
MLP_D_a.w.4.1.weight
MLP_D_a.w.4.1.bias
MLP_sigma_g_path.w.0.0.weight
MLP_sigma_g_path.w.0.0.bias
MLP_sigma_g_path.w.0.1.weight
MLP_sigma_g_path.w.0.1.bias
MLP_sigma_g_path.w.1.0.weight
MLP_sigma_g_path.w.1.0.bias
MLP_sigma_g_path.w.1.1.weight
MLP_sigma_g_path.w.1.1.bias
MLP_sigma_g_path.w.2.0.weight
MLP_sigma_g_path.w.2.0.bias
MLP_sigma_g_path.w.2.1.weight
MLP_sigma_g_path.w.2.1.bias
MLP_sigma_g_path.w.3.0.weight
MLP_sigma_g_path.w.3.0.bias
MLP_sigma_g_path.w.3.1.weight
MLP_sigma_g_path.w.3.1.bias
MLP_sigma_g_path.w.4.0.weight
MLP_sigma_g_path.w.4.0.bias
MLP_sigma_g_path.w.4.1.weight
MLP_sigma_g_path.w.4.1.bias
MLP_sigma_p.w.0.0.weight
MLP_sigma_p.w.0.0.bias
MLP_sigma_p.w.0.1.weight
MLP_sigma_p.w.0.1.bias
MLP_sigma_p.w.1.0.weight
MLP_sigma_p.w.1.0.bias
MLP_sigma_p.w.1.1.weight
MLP_sigma_p.w.1.1.bias
MLP_sigma_p.w.2.0.weight
MLP_sigma_p.w.2.0.bias
MLP_sigma_p.w.2.1.weight
MLP_sigma_p.w.2.1.bias
MLP_sigma_p.w.3.0.weight
MLP_sigma_p.w.3.0.bias
MLP_sigma_p.w.3.1.weight
MLP_sigma_p.w.3.1.bias
MLP_sigma_p.w.4.0.weight
MLP_sigma_p.w.4.0.bias
MLP_sigma_p.w.4.1.weight
MLP_sigma_p.w.4.1.bias
MLP_mu_g_mem.w.0.0.weight
MLP_mu_g_mem.w.0.0.bias
MLP_mu_g_mem.w.0.1.weight
MLP_mu_g_mem.w.0.1.bias
MLP_mu_g_mem.w.1.0.weight
MLP_mu_g_mem.w.1.0.bias
MLP_mu_g_mem.w.1.1.weight
MLP_mu_g_mem.w.1.1.bias
MLP_mu_g_mem.w.2.0.weight
MLP_mu_g_mem.w.2.0.bias
MLP_mu_g_mem.w.2.1.weight
MLP_mu_g_mem.w.2.1.bias
MLP_mu_g_mem.w.3.0.weight
MLP_mu_g_mem.w.3.0.bias
MLP_mu_g_mem.w.3.1.weight
MLP_mu_g_mem.w.3.1.bias
MLP_mu_g_mem.w.4.0.weight
MLP_mu_g_mem.w.4.0.bias
MLP_mu_g_mem.w.4.1.weight
MLP_mu_g_mem.w.4.1.bias
MLP_sigma_g_mem.w.0.0.weight
MLP_sigma_g_mem.w.0.0.bias
MLP_sigma_g_mem.w.0.1.weight
MLP_sigma_g_mem.w.0.1.bias
MLP_sigma_g_mem.w.1.0.weight
MLP_sigma_g_mem.w.1.0.bias
MLP_sigma_g_mem.w.1.1.weight
MLP_sigma_g_mem.w.1.1.bias
MLP_sigma_g_mem.w.2.0.weight
MLP_sigma_g_mem.w.2.0.bias
MLP_sigma_g_mem.w.2.1.weight
MLP_sigma_g_mem.w.2.1.bias
MLP_sigma_g_mem.w.3.0.weight
MLP_sigma_g_mem.w.3.0.bias
MLP_sigma_g_mem.w.3.1.weight
MLP_sigma_g_mem.w.3.1.bias
MLP_sigma_g_mem.w.4.0.weight
MLP_sigma_g_mem.w.4.0.bias
MLP_sigma_g_mem.w.4.1.weight
MLP_sigma_g_mem.w.4.1.bias
MLP_c_star.w.0.0.weight
MLP_c_star.w.0.0.bias
MLP_c_star.w.0.1.weight
MLP_c_star.w.0.1.bias
"""
