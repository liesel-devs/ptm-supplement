import csv
import subprocess
import sys
from pathlib import Path
import os

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = (
    "/mnt/vast-standard/home/brachem1/u18549/conda/envs/r-renv/lib:"
    + env.get("LD_LIBRARY_PATH", "")
)


def main(jobdir, index):
    csv_path = Path(jobdir) / "params.csv"
    run_path = Path(jobdir) / "run.py"
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

    args.append("--jobid")
    args.append(f"job{index:04d}")

    args.append("--jobdir")
    args.append(jobdir)

    args.append("--jobrow")
    args.append(str(index))

    # Call run.py with these arguments
    subprocess.run([sys.executable, run_path] + args, check=True, env=env)


if __name__ == "__main__":
    main(jobdir=sys.argv[1], index=sys.argv[2])
