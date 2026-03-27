import pandas as pd
from pathlib import Path
from states_classification import STATE_TO_ID
import numpy as np
import matplotlib.pyplot as plt
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
    # Keep data of only ones that have number of bedrooms and number of bathrooms
    if "bed" and "bath" in df.columns:
        df = df.dropna(subset=["bed"])
        print(f"after removing bedrooms missing rows: {len(df)}")
        df = df.dropna(subset=["bath"])
        print(f"after removing bathrooms missing rows: {len(df)}")
    #Remove data with missing acre lots
    if "acre_lot" in df.columns:
        df = df.dropna(subset=["acre_lot"])
        print(f"after removing acre_lot missing rows: {len(df)}")
    # 3. Drop unwanted columns
    existing_drop_cols = [col for col in drop_columns if col in df.columns]
    df = df.drop(columns=existing_drop_cols)
    print(f"Dropped columns: {existing_drop_cols}")
    if "price" in df.columns:
        df = df.dropna(subset=["price"])
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
        df["state_id"] = df["state"].map(STATE_TO_ID).astype(np.int8)
        df = df.drop(columns="state")
    # 5. Split zip_code into separate sections
    if split_zip and "zip_code" in df.columns:
        #remove missing values
        df = df[df['zip_code'].astype(str) != '00nan']
        #split by meaning
        df["national_area"] = df["zip_code"].str[0]
        df["national_area"] = df["national_area"].astype(np.int8)
        df["sectional_center_facility"] = df["zip_code"].str[1:3] #convert to numpy type and convert 7 == 07, and there is no need to make categories instead
        df["sectional_center_facility"] = df["sectional_center_facility"].astype(int).astype(np.int8)
        df["delivery_area"] = df["zip_code"].str[3:].astype(np.int8) #same case as above
        df = df.drop(columns="zip_code")
    # Calculate the 1st percentile
    lower_limit = df["price"].quantile(0.005)
    # Filter the data
    df = df[df["price"] > lower_limit]

    print(f"Removed prices below: ${lower_limit:,.2f}")
    #creating numpy compatible values
    df["price"] = df["price"].astype(np.float64)
    df["bed"] = df["bed"].astype(np.int8)
    df["bath"] = df["bath"].astype(np.int8)
    df["acre_lot"] = df["acre_lot"].astype(np.float64)
    df["house_size"] = df["house_size"].astype(np.float64)
    #saved cleaned file
    plot_show = input("Do you want to see price distribution? y/n ")
    if plot_show == "y":
        plt.figure(figsize=(10, 6))
        np.log10(df["price"]).hist(bins=100, color='skyblue', edgecolor='black')
        plt.title("Price Distribution (Log Scale)")
        plt.xlabel("Log10 of Price")
        plt.ylabel("Number of Houses")
        plt.show()
    #one-hot-encode states, and zips
    df = pd.get_dummies(df, columns = ["state_id", "national_area"], dtype = np.int8)
    df.to_csv(output_file, index=False)
   
    print(f"Final rows: {len(df)}")
    print(f"Final columns: {list(df.columns)}")
    print(f"Saved cleaned dataset to: {output_file}")


if __name__ == "__main__":
    INPUT_FILE = "realtor_final_cleaned_zip5.csv"
    OUTPUT_FILE = "clean_estate_data.csv"

    clean_realtor_data(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
    )