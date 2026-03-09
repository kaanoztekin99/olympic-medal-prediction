from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# =========================================================
# Configuration
# =========================================================
DATASET_DIR = Path("../data")
INPUT_CSV = DATASET_DIR / "dataset_olympics.csv"
OUTPUT_CSV = DATASET_DIR / "dataset_olympics_preprocessed.csv"
CORRELATION_PNG = DATASET_DIR / "correlation_matrix.png"

# If True, create binary target:
# Medal -> 1, No Medal -> 0
CREATE_BINARY_TARGET = True

# If True, normalize numerical columns with Min-Max scaling [0, 1]
APPLY_NORMALIZATION = True

# If True, one-hot encode categorical columns
APPLY_ENCODING = True

# If True, create BMI feature
CREATE_BMI_FEATURE = True

# If True, save correlation matrix heatmap
SAVE_CORRELATION_MATRIX = True

# Drop rows missing these critical physical attributes
CRITICAL_NUMERIC_COLUMNS = ["Age", "Height", "Weight"]

# Optional columns to remove if they exist
IRRELEVANT_COLUMNS = ["Name", "ID"]


# =========================================================
# Helper functions
# =========================================================
def find_column_case_insensitive(df: pd.DataFrame, target_name: str):
    """
    Return the actual column name in the dataframe matching target_name
    case-insensitively. Returns None if not found.
    """
    for col in df.columns:
        if col.strip().lower() == target_name.strip().lower():
            return col
    return None


def fill_medal_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing/empty medal values with 'No Medal'.
    """
    medal_col = find_column_case_insensitive(df, "Medal")
    if medal_col is None:
        raise ValueError("Could not find a 'Medal' column in the dataset.")

    df[medal_col] = df[medal_col].replace(r"^\s*$", np.nan, regex=True)
    df[medal_col] = df[medal_col].fillna("No Medal")
    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Medal_Binary column:
    1 -> athlete has a medal
    0 -> No Medal
    """
    medal_col = find_column_case_insensitive(df, "Medal")
    if medal_col is None:
        raise ValueError("Could not find a 'Medal' column in the dataset.")

    df["Medal_Binary"] = df[medal_col].apply(
        lambda x: 0 if str(x).strip().lower() == "no medal" else 1
    )
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop optional irrelevant columns if they exist.
    """
    columns_to_drop = []
    for col_name in IRRELEVANT_COLUMNS:
        actual_col = find_column_case_insensitive(df, col_name)
        if actual_col is not None:
            columns_to_drop.append(actual_col)

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert known numeric columns to numeric type when possible.
    """
    for col_name in CRITICAL_NUMERIC_COLUMNS:
        actual_col = find_column_case_insensitive(df, col_name)
        if actual_col is not None:
            df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce")

    return df


def drop_critical_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing critical numeric columns such as Age, Height, Weight.
    """
    actual_critical_cols = []
    for col_name in CRITICAL_NUMERIC_COLUMNS:
        actual_col = find_column_case_insensitive(df, col_name)
        if actual_col is not None:
            actual_critical_cols.append(actual_col)

    if actual_critical_cols:
        df = df.dropna(subset=actual_critical_cols)

    return df


def add_bmi_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create BMI feature using:
    BMI = Weight / (Height_in_meters ^ 2)

    Assumes Height is in centimeters and Weight is in kilograms.
    """
    height_col = find_column_case_insensitive(df, "Height")
    weight_col = find_column_case_insensitive(df, "Weight")

    if height_col is None or weight_col is None:
        raise ValueError("Could not find Height and/or Weight columns to compute BMI.")

    height_m = df[height_col] / 100.0
    df["BMI"] = df[weight_col] / (height_m ** 2)

    # Prevent inf values if any unexpected zero height appears
    df["BMI"] = df["BMI"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["BMI"])

    return df


def normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numeric columns to [0, 1] using MinMaxScaler.
    Excludes binary target columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Do not normalize binary target columns
    excluded = {"Medal_Binary"}
    numeric_cols = [col for col in numeric_cols if col not in excluded]

    if not numeric_cols:
        return df

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns.
    Keeps Medal column unless removed manually later.
    """
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # Remove Medal from encoding if Medal_Binary exists
    if "Medal_Binary" in df.columns:
        medal_col = find_column_case_insensitive(df, "Medal")
        if medal_col in categorical_cols:
            categorical_cols.remove(medal_col)

    if not categorical_cols:
        return df

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    return df

def prepare_dataframe_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a simplified dataframe for correlation visualization.
    Keeps core columns and converts categorical columns into numeric codes
    so they appear as single columns in the correlation matrix.
    """
    corr_df = df.copy()

    selected_columns = []

    for col_name in ["Age", "Height", "Weight", "BMI", "Team", "Season", "Sex", "Medal_Binary"]:
        actual_col = find_column_case_insensitive(corr_df, col_name)
        if actual_col is not None:
            selected_columns.append(actual_col)

    corr_df = corr_df[selected_columns].copy()

    # Convert categorical columns to category codes
    for col_name in ["Team", "Season", "Sex"]:
        actual_col = find_column_case_insensitive(corr_df, col_name)
        if actual_col is not None:
            corr_df[actual_col] = corr_df[actual_col].astype("category").cat.codes

    return corr_df


def save_correlation_matrix(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a simplified correlation matrix similar to the desired visualization.
    """
    corr_df = prepare_dataframe_for_correlation(df)
    corr_matrix = corr_df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# Main pipeline
def preprocess_dataset(input_csv: Path, output_csv: Path):
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"Reading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"Original shape: {df.shape}")

    # Step 1: Fill medal column
    df = fill_medal_column(df)

    # Step 2: Drop irrelevant columns
    df = drop_irrelevant_columns(df)

    # Step 3: Convert numeric columns safely
    df = convert_numeric_columns(df)

    # Step 4: Drop rows missing critical physical data
    df = drop_critical_missing_rows(df)

    # Step 5: Create BMI feature
    if CREATE_BMI_FEATURE:
        df = add_bmi_feature(df)

    # Step 6: Create binary target if requested
    if CREATE_BINARY_TARGET:
        df = create_binary_target(df)

    # Step 7: Save correlation matrix BEFORE one-hot encoding
    if SAVE_CORRELATION_MATRIX:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        save_correlation_matrix(df, CORRELATION_PNG)

    # Step 8: Normalize numeric columns
    if APPLY_NORMALIZATION:
        df = normalize_numeric_columns(df)

    # Step 9: Encode categorical columns
    if APPLY_ENCODING:
        df = encode_categorical_columns(df)

    # Save output CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Preprocessed dataset saved to: {output_csv}")
    print(f"Correlation matrix saved to: {CORRELATION_PNG}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    preprocess_dataset(INPUT_CSV, OUTPUT_CSV)