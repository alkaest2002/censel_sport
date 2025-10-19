
from pathlib import Path

import orjson

from lib import MT100
from lib.data_cleaner import clean_data
from lib.data_loader import load_from_synthetic
from lib.percentiles_bootstrap import compute_bootstrap_percentiles
from lib.percentiles_table import create_normative_table

#################################################################################
# Initialize metric configuration
################################################################################
metric_config = {
    "name": MT100,
    "type": "time",
    "units": "seconds",
    "precision": 2,
    "higher_is_better": False,
    "requested_percentiles": [5, 25, 50, 75, 95],
    "bootstrap_n_replicates": 10000,
    "bootstrap_n_replicate_size": 300,
}

#################################################################################
# Load data
################################################################################
print("1. Loading data...")

# Load synthetic data
data = load_from_synthetic(metric_config=metric_config)


#################################################################################
# Clean data
#################################################################################
print("2. Cleaning data...")

# Clean data
data = clean_data(data)


#################################################################################
# Compute bootstrap percentiles
#################################################################################
print("3. Computing bootstrap percentiles...")

# Compute bootstrap percentiles
bootstrap_percentiles, bootstrap_estimates = (
    compute_bootstrap_percentiles(
        data["analysis_data"],
        requested_percentiles=metric_config["requested_percentiles"],
        n_replicates=metric_config["bootstrap_n_replicates"],
        n_replicate_size=metric_config["bootstrap_n_replicate_size"],
    )
)


#################################################################################
# Create normative table
#################################################################################
print("4. Creating normative table...")

# Create normative table
normative_table = create_normative_table(bootstrap_percentiles)

# Store results in data dictionary
data["normative_table"] = {
    "requested_percentiles": metric_config["requested_percentiles"],
    "bootstrap_percentiles": bootstrap_percentiles,
    "percentile_cutoffs": normative_table,
}


#################################################################################
# Save results
#################################################################################
print("5. Saving results...")

# Determine output folder
output_path = Path(f"./out/{metric_config['name']}")

# Make sure output folder exists or create it
output_path.mkdir(parents=True, exist_ok=True)

# Delete existing files in folder if exists
for child in output_path.iterdir():
    if child.is_file():
        child.unlink()

# Write results to JSON file
analysis_output_path = output_path / "analysis.json"
with analysis_output_path.open("w") as f:
    orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
    f.write(orjson.dumps(data, option=orjson_options).decode("utf-8"))

# Write bootstrap estimates to separate JSON file
bootstrap_output_path = output_path / "bootstrap_estimates.json"
with bootstrap_output_path.open("w") as f:
    orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
    f.write(orjson.dumps(bootstrap_estimates, option=orjson_options).decode("utf-8"))
