#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import polars as pl
import seaborn as sns
from pathlib import Path
from tbparse import SummaryReader
from tem import world
import socket
import matplotlib.pyplot as plt
import pingouin as pg

version = "v5"

if socket.gethostname() == "submit":  # cluster
    root_dir = Path("/home/uwm/linzan/torch_tem")
else:  # local laptop
    root_dir = Path("~/PycharmProjects/torch_tem").expanduser()

base_dir = root_dir / "tem_simulation" / "operators" / version
design_dir = root_dir / version / "design"
env_files = [
    str(root_dir / "envs" / "2x3_env1.json"),
    str(root_dir / "envs" / "2x3_env2.json")
]
override_file = str(root_dir / "params" / "parameters.json")

start_subj = 501
n_subj = 50
subjects = [f"{number:03d}" for number in range(start_subj, start_subj + n_subj)]
n_runs = 10


# In[2]:


# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

sim_dir = base_dir / "simulation"
sim_dir.mkdir(exist_ok=True)

study_dir = sim_dir / "study-8"
while study_dir.exists():
    number = int(study_dir.name.split("-")[1])
    study_dir = sim_dir / f"study-{number + 1}"
study_dir.mkdir()

fig_dir = base_dir / "figs" / study_dir.name
fig_dir.mkdir(exist_ok=True, parents=True)

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
    out_dir.mkdir()
    design_files = [
        out_dir / f"sub-{subject}_graph-{graph}_design.csv" for graph in [1, 2]
    ]
    trials.filter(graph="graph_1").write_csv(design_files[0])
    trials.filter(graph="graph_2").write_csv(design_files[1])

    out_dir.mkdir(exist_ok=True)

    for run in range(1, n_runs + 1):
        tem_model = world.learn_operators(env_files, design_files, out_dir, subject, run, override_file)


# In[ ]:


raw_dfs = []
for subject in subjects:
    number = int(subject)
    condition = "AL" if number % 2 == 0 else "PI"
    for design in ["0", "1"]:
        design_path = study_dir / f"sub-{subject}" / f"design-{design}"
        if not design_path.exists():
            continue
        reader = SummaryReader(str(design_path))
        raw = (
            pl.DataFrame(reader.scalars)
            .with_columns(
                subject=pl.lit(subject),
                condition=pl.lit(condition),
                design=pl.lit(design),
                run=pl.col("value").cum_count().over(pl.col("step")),
            )
        )
        raw_dfs.append(raw)
df = pl.concat(raw_dfs)


# In[ ]:


df = df.with_columns(
    pl.col("step").cum_count().over("subject", "run", "design").alias("Trial"),
    pl.col("design").replace({"0": "Initial", "1": "Transfer"}).alias("Graph"),
)


# In[ ]:
plt.style.use(str(root_dir / "src" / "tem" / "figures.mplstyle"))


# In[ ]:
g = sns.relplot(
    (
        df.filter(pl.col("tag") == "Accuracies/g")
        .with_columns(pl.col("value").rolling_mean(window_size=50).over("subject", "run", "Graph"))
    ),
    x="Trial",
    y="value",
    col="Graph",
    kind="line",
    hue="condition",
)
g.set(ylabel="Structural accuracy")
g.savefig(fig_dir / f"learning_{version}_structure.pdf")


# In[ ]:
g = sns.relplot(
    (
        df.filter(pl.col("tag") == "Accuracies/p")
        .with_columns(pl.col("value").rolling_mean(window_size=50).over("subject", "run", "Graph"))
    ),
    x="Trial",
    y="value",
    col="Graph",
    kind="line",
    hue="condition",
)
g.set(ylabel="Perceptual accuracy")
g.savefig(fig_dir / f"learning_{version}_perception.pdf")


# In[ ]:


m = (
    df.filter((pl.col("tag") == "Accuracies/g") & (pl.col("Trial").is_between(100, 190)))
    .group_by("subject", "condition", "Graph")
    .agg(pl.col("value").mean())
    .sort("subject", "condition", "Graph")
)
g = sns.catplot(m, x="Graph", y="value", hue="condition", kind="bar")
g.set(ylabel="Structural accuracy")
g.savefig(fig_dir / f"learning_{version}_structure_peak_contrast.pdf")


# In[ ]:
al = m.filter((pl.col("Graph") == "Transfer") & (pl.col("condition") == "AL"))
pi = m.filter((pl.col("Graph") == "Transfer") & (pl.col("condition") == "PI"))
print(pg.ttest(al["value"].to_numpy(), pi["value"].to_numpy()))


# In[ ]:
m = (
    df.filter((pl.col("tag") == "Accuracies/p") & (pl.col("Trial").is_between(100, 150)))
    .group_by("subject", "condition", "Graph")
    .agg(pl.col("value").mean())
    .sort("subject", "condition", "Graph")
)
g = sns.catplot(m, x="Graph", y="value", hue="condition", kind="bar")
g.set(ylabel="Perceptual accuracy")
g.savefig(fig_dir / f"learning_{version}_perception_peak_contrast.pdf")


# In[ ]:


al = m.filter((pl.col("Graph") == "Transfer") & (pl.col("condition") == "AL"))
pi = m.filter((pl.col("Graph") == "Transfer") & (pl.col("condition") == "PI"))
print(pg.ttest(al["value"].to_numpy(), pi["value"].to_numpy()))


