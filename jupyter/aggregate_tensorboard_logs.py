"""
Aggregate TensorBoard training logs from a TEM simulation study into a single file.

During training, each subject/run/design combination writes its own set of
TensorBoard log files (events.out.tfevents.*) recording loss and accuracy at
every training iteration. For 50 subjects x 10 runs x 2 designs, this means
hundreds of small log files scattered across many folders.

This script reads all of those log files for one study, combines them into a
single tidy table (one row per training iteration per subject/run/design),
and saves the result as a single .parquet file. This makes it much faster and
easier to download and analyze the results later on a local laptop.

This script is exclusively used upon the completion of the model's training on the HPC cluster.

Usage (run after training finishes, on the cluster):
    python aggregate_tensorboard_logs.py study-18
"""


import sys
from pathlib import Path
import polars as pl
from tbparse import SummaryReader

# The study folder to aggregate, passed as a command-line argument
# (e.g. "study-18" -> reads from .../simulation/study-18/)
study_name = sys.argv[1]

# Path is hardcoded to the cluster location, since this script only runs on the cluster
root_dir = Path("/home/uwm/linzan/Data_morton/linzan/torch_tem")
version = "v5"
base_dir = root_dir / "tem_simulation" / "operators" / version
sim_dir = base_dir / "simulation"
study_dir = sim_dir / study_name

start_subj = 501
n_subj = 50
subjects = [f"{number:03d}" for number in range(start_subj, start_subj + n_subj)]

# Read raw TensorBoard logs for every subject/design
# Each subject has two designs: "0" = Initial map, "1" = Transfer map.
# Subject condition (PI vs AL) is determined by whether the subject number
# is odd or even, per the yoked PI/AL design.
raw_dfs = []
for subject in subjects:
    number = int(subject)
    condition = "AL" if number % 2 == 0 else "PI"
    for design in ["0", "1"]:
        design_path = study_dir / f"sub-{subject}" / f"design-{design}"
        if not design_path.exists():
            continue
        # SummaryReader parses all tfevents files in this folder into a single table of (step, tag, value) rows
        reader = SummaryReader(str(design_path))
        raw = (
            pl.DataFrame(reader.scalars)
            .with_columns(
                subject=pl.lit(subject),
                condition=pl.lit(condition),
                design=pl.lit(design),
                # Each TensorBoard "step" can have multiple runs logged to it;
                # this assigns a run number (1, 2, 3, ...) per step
                run=pl.col("value").cum_count().over(pl.col("step")),
            )
        )
        raw_dfs.append(raw)

# Combine every subject/design into one long table
df = pl.concat(raw_dfs)

# Add human-readable trial number and graph label columns
# "Trial" = position within this subject/run/design's training sequence (i.e. which trial number this logged value corresponds to)
# "Graph" = readable label for the design ("Initial" or "Transfer")
df = df.with_columns(
    pl.col("step").cum_count().over("subject", "run", "design").alias("Trial"),
    pl.col("design").replace({"0": "Initial", "1": "Transfer"}).alias("Graph"),
)

# Save the aggregated table as a single parquet file
out_path = study_dir / "aggregated.parquet"
df.write_parquet(out_path)
print(f"Saved aggregated df to {out_path}, shape={df.shape}")
