import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from google.colab import files


train = pd.read_csv("train set.csv")
test = pd.read_csv("test set.csv")

print( train.shape)
print( test.shape)

"====================="

def extract_numeric(value):
    try:
        return float(str(value).split()[0])
    except:
        return np.nan

def extract_nm(torque_str):
    if pd.isna(torque_str):
        return np.nan
    s = str(torque_str).lower().replace(',', '')
    m = re.search(r'(\d+\.?\d*)\s*nm', s)
    if m:
        return float(m.group(1))
    m = re.search(r'(\d+\.?\d*)\s*@?\s*\(?kgm', s)
    if m:
        return float(m.group(1)) * 9.80665
    return np.nan

def extract_rpm(torque_str):
    if pd.isna(torque_str):
        return np.nan
    s = str(torque_str).lower().replace(',', '')
    m_range = re.search(r'(\d{3,5})\s*[-–]\s*(\d{3,5})\s*rpm', s)
    if m_range:
        low, high = map(float, m_range.groups())
        return (low + high) / 2
    m_single = re.search(r'@?\s*(\d{3,5})\s*rpm', s)
    if m_single:
        return float(m_single.group(1))
    return np.nan

"=================================="

def preprocess(df):
    df = df.copy()
    df['mileage'] = df['mileage'].apply(extract_numeric)
    df['engine'] = df['engine'].apply(extract_numeric)
    df['max_power'] = df['max_power'].apply(extract_numeric)
    df['torque_nm'] = df['torque'].apply(extract_nm)
    df['torque_rpm'] = df['torque'].apply(extract_rpm)
    df.drop(columns=['torque'], inplace=True)
    return df


"===================================="

train_cleaned = preprocess(train)
test_cleaned = preprocess(test)

print("datacleaned")

"==================================="

label_maps = {}

for col in train_cleaned.select_dtypes(include='object').columns:
    le = LabelEncoder()
    all_values = pd.concat([train_cleaned[col], test_cleaned[col]], axis=0).astype(str)
    le.fit(all_values)

    train_cleaned[col] = le.transform(train_cleaned[col].astype(str))
    test_cleaned[col] = le.transform(test_cleaned[col].astype(str))

    label_maps[col] = dict(zip(le.transform(le.classes_), le.classes_))

explanation_df = pd.DataFrame([
    {"column": col, "code": code, "label": label}
    for col, mapping in label_maps.items()
    for code, label in mapping.items()
])

explanation_df.to_excel("label_explanation.xlsx", index=False)
files.download("label_explanation.xlsx")


"==================================="

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

outlier_columns = ['selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'torque_nm', 'torque_rpm']
for col in outlier_columns:
    train_cleaned = remove_outliers_iqr(train_cleaned, col)

print("outliers removed")


"======================================"

train_cleaned.to_csv("train_cleaned.csv", index=False)
test_cleaned.to_csv("test_cleaned.csv", index=False)

files.download("train_cleaned.csv")
files.download("test_cleaned.csv")
