import pandas as pd
import numpy as np
import re

def main():
    # Load the dataset
    df = pd.read_csv("train set.csv")

    # Function to extract numeric value from string columns
    def extract_numeric(value):
        try:
            return float(str(value).split()[0])
        except:
            return np.nan

    # Convert textual numeric columns to float
    df['mileage'] = df['mileage'].apply(extract_numeric)
    df['engine'] = df['engine'].apply(extract_numeric)
    df['max_power'] = df['max_power'].apply(extract_numeric)

    # Extract torque in Nm from torque column
    def extract_nm(torque_str):
        if pd.isna(torque_str):
            return np.nan
        torque_str = str(torque_str).lower().replace(',', '')
        match = re.search(r'(\d+\.?\d*)\s*nm', torque_str)
        if match:
            return float(match.group(1))
        match_kgm = re.search(r'(\d+\.?\d*)\s*@?\s*\(?kgm', torque_str)
        if match_kgm:
            return float(match_kgm.group(1)) * 9.80665
        return np.nan

    df['torque_nm'] = df['torque'].apply(extract_nm)

    # Function to remove outliers using IQR method
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df[(df[column] >= lower) & (df[column] <= upper)]

    # Remove outliers from numeric columns
    numeric_cols = ['selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'torque_nm']
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)

    # Save cleaned data to CSV
    df.to_csv("train_cleaned.csv", index=False)
    print("✅ Data cleaned and saved to train_cleaned.csv")

if __name__ == "__main__":
    main()
