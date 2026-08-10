# Grid over block_epochs, the new independent variable. walks_multiplier is
# fixed at 1: it repeats the whole 216-trial sequence, which scales blocks 1
# and 2 together and leaves the manipulation's share unchanged. block_epochs
# repeats only blocks 2 and 4, raising block 2's share of the Initial map from
# 22% at 1 epoch to 72% at 9.
#
# lambda and eta are the same for PI and AL. study-24 is the only earlier grid
# where they were, and every group comparison from study-30 onwards is
# confounded because PI used 0.99/0.25 and AL 0.9999/0.2.
lambdas = [0.99, 0.999, 0.9999]
etas = [0.2, 0.25, 0.3, 0.5]
block_epochs_values = [1, 3, 5, 7, 9]

lines = ["# Auto-generated block-epoch grid\n"]
for l in lambdas:
    for e in etas:
        for be in block_epochs_values:
            l_str = str(l).replace(".", "_")
            e_str = str(e).replace(".", "_")
            study = f"study-be-l{l_str}-e{e_str}-b{be}"
            be_arg = "" if be == 1 else f'--block-epochs "2:{be},4:{be}" '
            cmd = (f"uv run jupyter/train.py "
                   f"--lambda-val {l} --eta-val {e} "
                   f"--walks-multiplier 1 "
                   f"{be_arg}"
                   f"--study-name {study} "
                   f"--n-subj 10\n")
            lines.append(cmd)

with open("commands.sh", "w") as f:
    f.writelines(lines)

print(f"Generated {len(lines) - 1} commands")