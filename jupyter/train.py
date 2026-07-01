"""
Train TEM operators model with configurable hyperparameters.

Usage examples:
    # Single run
    uv run jupyter/train.py --lambda-val 0.99 --eta-val 0.5 --study-name study-19 --n-subj 10

    # With custom walks multiplier
    uv run jupyter/train.py --lambda-val 0.99 --eta-val 0.5 --study-name study-19 --walks-multiplier 10 --n-subj 10
"""

import click
import json
from pathlib import Path
import numpy as np
import torch
import polars as pl
from tem import world


@click.command()
@click.option("--lambda-val", type=float, required=True, help="Hebbian forgetting rate (e.g. 0.99)")
@click.option("--eta-val", type=float, required=True, help="Hebbian learning rate (e.g. 0.5)")
@click.option("--lr-decay-steps", type=int, default=2000, help="Learning rate decay steps")
@click.option("--walks-multiplier", type=int, default=10, help="Number of times to repeat walks (epochs)")
@click.option("--study-name", type=str, required=True, help="Output study folder name, e.g. study-19")
@click.option("--start-subj", type=int, default=501, help="First subject ID")
@click.option("--n-subj", type=int, default=10, help="Number of subjects to train")
@click.option("--n-runs", type=int, default=10, help="Number of runs per subject")
@click.option("--seed", type=int, default=0, help="Random seed")

def train(lambda_val, eta_val, lr_decay_steps, walks_multiplier, study_name, start_subj, n_subj, n_runs, seed):
    # --- Setup paths ---
    cluster_path = Path("/home/uwm/linzan/Data_morton/linzan/torch_tem")
    root_dir = cluster_path if cluster_path.exists() else Path("~/PycharmProjects/torch_tem").expanduser()

    version = "v5"
    base_dir = root_dir / "tem_simulation" / "operators" / version
    design_dir = root_dir / version / "design"
    env_files = [
        str(root_dir / "envs" / "2x3_env1.json"),
        str(root_dir / "envs" / "2x3_env2.json"),
    ]

    # Build parameter override
    # Load base parameters from parameters.json, then override the ones we're tuning
    with open(root_dir / "params" / "parameters.json") as f:
        base_params = json.load(f)
    base_params["lambda"] = lambda_val
    base_params["eta"] = eta_val
    base_params["lr_decay_steps"] = lr_decay_steps

    # Write override to a temp file named after the study, so parallel jobs don't conflict
    override_file = root_dir / "params" / f"_override_{study_name}.json"
    with open(override_file, "w") as f:
        json.dump(base_params, f)

    # Setup output directories
    np.random.seed(seed)
    torch.manual_seed(seed)

    sim_dir = base_dir / "simulation"
    study_dir = sim_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = base_dir / "figs" / study_name
    fig_dir.mkdir(exist_ok=True, parents=True)

    # Train
    subjects = [f"{n:03d}" for n in range(start_subj, start_subj + n_subj)]

    for subject in subjects:
        raw_file = design_dir / f"sub-{subject}" / f"sub-{subject}_task-learning_design.csv"
        raw = pl.read_csv(raw_file)
        trials = raw.filter(
            ~(
                pl.col("trial_type").str.starts_with("practice")
                | pl.col("trial_type").str.contains("instruction")
                | pl.col("trial_type").str.contains("feedback")
            )
        )
        out_dir = study_dir / f"sub-{subject}"
        out_dir.mkdir(exist_ok=True)
        design_files = [
            out_dir / f"sub-{subject}_graph-{graph}_design.csv" for graph in [1, 2]
        ]
        trials.filter(graph="graph_1").write_csv(design_files[0])
        trials.filter(graph="graph_2").write_csv(design_files[1])

        for run in range(1, n_runs + 1):
            world.learn_operators(
                env_files, design_files, out_dir, subject, run,
                str(override_file), walks_multiplier=walks_multiplier
            )

    # Clean up temp override file
    override_file.unlink(missing_ok=True)
    print(f"Done: {study_name} (lambda={lambda_val}, eta={eta_val}, lr_decay_steps={lr_decay_steps}, walks×{walks_multiplier})")


if __name__ == "__main__":
    train()