"""Download and prepare the IBM Telco churn demo without committing raw customer IDs."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def prepare(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(SOURCE_URL, timeout=60) as response:
        raw = response.read()

    source_path = output_dir / "_customer_churn_source.csv"
    source_path.write_bytes(raw)
    df = pd.read_csv(source_path)
    source_path.unlink(missing_ok=True)

    # CustomerID is an identifier, not a useful predictive feature or a demo artifact.
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.to_csv(output_dir / "customer_churn.csv", index=False)

    features = df.drop(columns=["Churn"])
    stable = features.sample(n=min(100, len(features)), random_state=42).reset_index(drop=True)
    stable.to_csv(output_dir / "incoming_churn_stable.csv", index=False)

    shifted = stable.copy()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] + 50
    shifted["tenure"] = (shifted["tenure"] - 18).clip(lower=0)
    if "Contract" in shifted:
        shifted["Contract"] = "Month-to-month"
    shifted.to_csv(output_dir / "incoming_churn_shifted.csv", index=False)

    print(f"Prepared {len(df)} rows in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/samples"))
    prepare(parser.parse_args().output_dir)
