#!/bin/bash
set -eo pipefail
module load apptainer/1.4.5
export APPTAINER_TMPDIR="$PWD/.apptainer/tmp"

apptainer exec --containall --cleanenv --no-home \
  --bind "$PWD:/a4" --pwd /a4/extra_credit \
  .apptainer/part2.sif python -B -u - <<'PY'
from pathlib import Path
import pickle
import shutil

from utils import compute_records, read_queries

source = Path(
    "results/t5_scr_ec_lr2e-4_bs8_cosine_maxep60_pat12_sqlops_test.sql"
)
submission_sql = Path("results/t5_ft_experiment_ec_test.sql")
submission_pkl = Path("records/t5_ft_experiment_ec_test.pkl")

if submission_sql.exists() or submission_pkl.exists():
    raise FileExistsError("A submission file already exists. Back it up before rerunning.")

expected = len(Path("data/test.nl").read_text().splitlines())
queries = read_queries(source)
if expected != 432 or len(queries) != expected:
    raise ValueError(f"Test inputs={expected}, predicted SQL={len(queries)}")

records, errors = compute_records(queries)
if len(records) != expected or len(errors) != expected:
    raise ValueError("The number of records or error entries does not match Test.")

timeouts = errors.count("Query timed out")
other_errors = sum(bool(error) for error in errors) - timeouts
print(f"Test records={len(records)}, timeouts={timeouts}, other SQL errors={other_errors}")
if timeouts:
    raise RuntimeError("Some queries timed out. Submission files were not created.")

with submission_pkl.open("xb") as f:
    pickle.dump((records, errors), f)

with submission_pkl.open("rb") as f:
    saved_records, saved_errors = pickle.load(f)
if len(saved_records) != expected or len(saved_errors) != expected:
    raise ValueError("Saved pickle failed verification.")

with source.open("rb") as src, submission_sql.open("xb") as dst:
    shutil.copyfileobj(src, dst)

print("READY: both extra credit submission files are verified.")
PY