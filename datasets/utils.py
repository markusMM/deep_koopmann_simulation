import os
from typing import Iterable
import zipfile
import logging

import pandas as pd


# Folder containing your compressed ERA5 or CSV archives
def unzip_csv_or_nc_batch(
    input_dir: str = "./test_data",
    output_dir: (str|None) = None,
    repl_suffix: (str|None) = "_compr.zip",
    overwrite: bool = False
) -> None:
    # check output path exists
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    for zip_name in os.listdir(input_dir):
        if not zip_name.endswith(".zip"):
            continue

        zip_path = os.path.join(input_dir, zip_name)
        if repl_suffix is not None:
            base_name = zip_name.replace(repl_suffix, "")

        with zipfile.ZipFile(zip_path, "r") as z:
            k = 0  # counter for multiple files
            for member in z.namelist():
                # Only extract NetCDF or CSV
                if member.endswith((".nc", ".csv")):
                    k += 1
                    # Choose extension dynamically
                    ext = ".nc" if member.endswith(".nc") else ".csv"
                    out_name = f"{base_name}_{k :2d}{ext}"
                    out_path = os.path.join(output_dir, out_name)

                    if os.path.exists(out_path) and not overwrite:
                        continue

                    # Write extracted content into new file
                    with z.open(member) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())

                    logging.info(f"Saved: {out_path}")


def add_specification(
    df: pd.DataFrame, 
    specification: (float|int|str|Iterable), 
    col_name: str = 'cluster'
) -> pd.DataFrame:
    if isinstance(specification, Iterable):
        assert len(specification) == len(df)  # type: ignore
    df[col_name] = specification
    return df
