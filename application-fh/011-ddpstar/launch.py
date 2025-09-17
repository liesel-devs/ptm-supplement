import csv
import subprocess
import pandas as pd
from pathlib import Path
import os

env = os.environ.copy()


def main(jobdir, index):
    csv_path = Path(jobdir) / "params.csv"
    run_path = Path(jobdir) / "run.R"
    index = int(index)

    # Read row from CSV
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    row = rows[index]

    # Build CLI arguments from CSV headers
    args = []
    for key, value in row.items():
        args.append(f"--{key}")
        args.append(str(value))

    args.append("--jobdir")
    args.append(jobdir)

    args.append("--jobrow")
    args.append(str(index))

    # Call run.py with these arguments
    subprocess.run(["Rscript", run_path] + args, check=True, env=env)


if __name__ == "__main__":
    jobdir = Path("application-fh/011-ddpstar")
    params = pd.read_csv(jobdir / "params.csv")

    for index, row in params.iterrows():
        try:
            main(jobdir=jobdir, index=index + 1)
        except RuntimeError:
            pass
