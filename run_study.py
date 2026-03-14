"""Run TEM through a specific experimental design."""

import numpy as np
import polars as pl
import torch
from pathlib import Path
from tem import world, parameters

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

design_dir = Path("~/VSCode/operators/design").expanduser()
env_files = ['./envs/2x3_env1.json', './envs/2x3_env2.json']
n_subj = 30
n_runs = 10

sim_dir = design_dir / "simulation"
sim_dir.mkdir(exist_ok=True)

study_dir = sim_dir / "study-1"
while study_dir.exists():
    number = int(study_dir.name.split("-")[1])
    study_dir = sim_dir / f"study-{number + 1}"
study_dir.mkdir()

for number in range(1, n_subj + 1):
    subject = f"{number:03d}"

    raw_file = design_dir / f"sub-{subject}" / f"sub-{subject}_task-learning_design.csv"
    raw = pl.read_csv(raw_file)
    trials = raw.filter(
        ~(
                pl.col("trial_type").str.starts_with('practice')
                | pl.col("trial_type").str.contains("instruction")
                | pl.col("trial_type").str.contains("feedback")
        )
    )
    out_dir = study_dir / f"sub-{subject}"
    out_dir.mkdir()
    design_files = [out_dir / f"sub-{subject}_graph-{graph}_design.csv" for graph in [1, 2]]
    trials.filter(graph="graph_1").write_csv(design_files[0])
    trials.filter(graph="graph_2").write_csv(design_files[1])

    out_dir.mkdir(exist_ok=True)
    for run in range(1, n_runs + 1):
        tem_model = world.learn_design(env_files, design_files, out_dir, subject, run)
