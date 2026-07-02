lambdas = [0.99, 0.999, 0.9999]
etas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
walks = range(1, 11)

lines = ["# Auto-generated grid search commands\n"]
for l in lambdas:
    for e in etas:
        for w in walks:
            l_str = str(l).replace(".", "_")
            e_str = str(e).replace(".", "_")
            study = f"study-grid2-l{l_str}-e{e_str}-w{w}"
            cmd = (f"uv run jupyter/train.py "
                   f"--lambda-val {l} --eta-val {e} "
                   f"--walks-multiplier {w} "
                   f"--study-name {study} "
                   f"--n-subj 10\n")
            lines.append(cmd)

with open("commands.sh", "w") as f:
    f.writelines(lines)

print(f"Generated {len(lines)-1} commands")