import pandas as pd
from pathlib import Path
from states_classification import STATE_TO_ID
import numpy as np
def clean_realtor_data(
    input_file: str,
    output_file: str,
    sold_status: str = "sold",
    drop_columns=None,
    split_zip: bool = True,
):


    if drop_columns is None:
        drop_columns = [
            "status",
            "brokered_by",
            "city",
            "street",
            "prev_sold_date",
        ]

    # Load data
    df = pd.read_csv(input_file)

    print(f"Original rows: {len(df)}")
    # 1. Keep only sold houses
    if "status" in df.columns:
        df = df[df["status"] == sold_status].copy()
        print(f"Rows after keeping only '{sold_status}': {len(df)}")

    # 2. Remove rows where house_size is missing
    if "house_size" in df.columns:
        df = df.dropna(subset=["house_size"])
        print(f"Rows after removing missing house_size: {len(df)}")
    
    # 3. Drop unwanted columns
    existing_drop_cols = [col for col in drop_columns if col in df.columns]
    df = df.drop(columns=existing_drop_cols)
    print(f"Dropped columns: {existing_drop_cols}")

    # 4. Fix zip_code to always be 5 digits
    if "zip_code" in df.columns:
        df["zip_code"] = (
            df["zip_code"]
            .astype(str)
            .str.split(".").str[0]   # removes .0 if zip was read as float
            .str.strip()
            .str.zfill(5)
        )
    # Change state names to ids
    if "state" in df.columns:
        df["state_id"] = df["state"].map(STATE_TO_ID) 
        df = df.drop(columns="state")

    #####Ask temmates
    bad_data = df[df['zip_code'].astype(str) == '00nan']
    print(bad_data)
    # 5. Split zip_code into separate sections
    if split_zip and "zip_code" in df.columns:
        # full digit split
        df["national_area"] = df["zip_code"].str[0]
        df["national_area"] = df["national_area"].astype(np.int8)
        df["sectional_center_facility"] = df["zip_code"].str[1:3] #convert to numpy type and convert 7 == 07, and there is no need to make categories instead
        df["sectional_center_facility"] = df["sectional_center_facility"].astype(int).astype(np.int16)
        df["delivery_area"] = df["zip_code"].str[3:] #same case as above
        df = df.drop(columns="zip_code")
    #saved cleaned file
    df.to_csv(output_file, index=False)

    print(f"Final rows: {len(df)}")
    print(f"Final columns: {list(df.columns)}")
    print(f"Saved cleaned dataset to: {output_file}")


if __name__ == "__main__":
    INPUT_FILE = "data/realtor_final_cleaned_zip5.csv"
    OUTPUT_FILE = "data/realtor_cleaned_ready.csv"

    clean_realtor_data(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
    )