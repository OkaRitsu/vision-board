import argparse
import os
from datetime import datetime

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a CSV file from the dataset directory."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="path/to/dataset",
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="dataset.csv",
        help="Path to the output CSV file.",
    )
    args = parser.parse_args()

    samples = []
    for root, dirs, files in os.walk(args.dataset_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                full_path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(full_path))
                splited = os.path.splitext(file)[0].replace("Fotos ", "").split("_")
                # Convert date string to datetime object
                date_str = splited[0]
                try:
                    date_obj = datetime.strptime(date_str, "%d-%m-%Y")
                except ValueError:
                    date_obj = None

                samples.append(
                    {
                        # "filepath": full_path,
                        "label": label,
                        "filename": file,
                        "date": date_obj,
                        "idx1": splited[2],
                        "idx2": splited[-1],
                    }
                )

    df = pd.DataFrame(samples)
    df.to_csv(args.output_csv, index=False)
