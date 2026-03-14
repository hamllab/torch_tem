"""Run TEM through a specific experimental design."""

import numpy as np
import torch
from pathlib import Path
from tem import world, parameters

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

design_dir = Path("~/VSCode/operators/design/simulation").expanduser()
env_files = ['./envs/2x3_env1.json', './envs/2x3_env2.json']
subjects = ["001", "002"]
conditions = {"001": "PI", "002": "AL"}
n_runs = 100

study_dir = design_dir / "study-1"
while study_dir.exists():
    number = int(study_dir.name.split("-")[1])
    study_dir = design_dir / f"study-{number + 1}"
study_dir.mkdir()

for subject in subjects:
    cond = conditions[subject]
    design_files = [design_dir / f"sub-{subject}_graph-{graph}_design.csv" for graph in [1, 2]]
    out_dir = study_dir / f"cond-{cond}"
    out_dir.mkdir(exist_ok=True)
    for run in range(1, n_runs + 1):
        tem_model = world.learn_design(env_files, design_files, out_dir, run)
