import csv
import subprocess
import sys
from pathlib import Path

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
    
    args.append("--jobid")
    args.append(f"job{index:04d}")

    args.append("--jobdir")
    args.append(jobdir)

    # Call run.py with these arguments
    subprocess.run(["Rscript", run_path] + args, check=True)

if __name__ == "__main__":
    main(jobdir=sys.argv[1], index=sys.argv[2])