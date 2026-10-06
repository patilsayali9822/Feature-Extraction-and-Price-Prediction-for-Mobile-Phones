import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

PLOTS_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)

MODELS_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)


# Create output directories automatically
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


DATA_PATH = os.path.join(
    DATA_DIR,
    "mobile_phone_data.csv"
)


# ============================================================
# 2. HELPER FUNCTION
# ============================================================

def extract_number(value):
    """
    Extract the first number from a value.

    Examples:
        50MP      -> 50
        16MP      -> 16
        7,299     -> 7299
        6000      -> 6000
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    value = value.replace(",", "")

    match = re.search(
        r"\d+(?:\.\d+)?",
        value
    )

    if match:
        return float(match.group())

    return np.nan


# ============================================================
# 3. START
# ============================================================

print("\n")
print("=" * 75)
print("MOBILE PHONE PRICE PREDICTION")
print("=" * 75)


# ============================================================
# 4. CHECK DATASET
# ============================================================

if not os.path.exists(DATA_PATH):

    print("\nERROR: Dataset not found.")

    print("\nPlease put your CSV file here:")

    print(
        DATA_PATH
    )

    print(
        "\nThe CSV file must be named:"
    )

    print(
        "mobile_phone_data.csv"
    )

    raise SystemExit


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n[1/15] Loading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    f"Dataset loaded successfully."
)

print(
    f"Rows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)


# ============================================================
# 6. CLEAN COLUMN NAMES
# ============================================================

print("\n[2/15] Cleaning column names...")

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.replace(
        " ",
        "_",
        regex=False
    )
    .str.replace(
        "-",
        "_",
        regex=False
    )
)

print(
    "\nOriginal/cleaned columns:"
)

print(
    df.columns.tolist()
)


# ============================================================
# 7. STANDARDIZE COLUMN NAMES
# ============================================================

column_mapping = {}

for column in df.columns:

    column_lower = column.lower()

    if column_lower in [
        "prize",
        "price"
    ]:
        column_mapping[column] = "Price"

    elif column_lower in [
        "colour",
        "color"
    ]:
        column_mapping[column] = "Colour"

    elif column_lower in [
        "battery",
        "battery_",
        "battery_capacity"
    ]:
        column_mapping[column] = "Battery"

    elif column_lower in [
        "processor",
        "processor_"
    ]:
        column_mapping[column] = "Processor"

    elif column_lower in [
        "rear_camera"
    ]:
        column_mapping[column] = "Rear_Camera"

    elif column_lower in [
        "front_camera"
    ]:
        column_mapping[column] = "Front_Camera"

    elif column_lower in [
        "mobile_height"
    ]:
        column_mapping[column] = "Mobile_Height"

    elif column_lower in [
        "ai_lens"
    ]:
        column_mapping[column] = "AI_Lens"


df.rename(
    columns=column_mapping,
    inplace=True
)


# ============================================================
# 8. CHECK TARGET COLUMN
# ============================================================

if "Price" not in df.columns:

    print(
        "\nERROR: Price/Prize column not found."
    )

    print(
        "Available columns:"
    )

    print(
        df.columns.tolist()
    )

    raise SystemExit


# ============================================================
# 9. DATA EXPLORATION
# ============================================================

print("\n[3/15] Exploring dataset...")

print("\nFirst 5 records:")
print(
    df.head()
)

print("\nDataset information:")
print(
    df.info()
)

print("\nMissing values:")
print(
    df.isnull().sum()
)

print("\nDuplicate records:")
print(
    df.duplicated().sum()
)


# ============================================================
# 10. REMOVE DUPLICATES
# ============================================================

print("\n[4/15] Removing duplicate records...")

before = len(df)

df = (
    df
    .drop_duplicates()
    .reset_index(drop=True)
)

after = len(df)

print(
    f"Removed duplicates: {before - after}"
)

print(
    f"Remaining rows: {after}"
)


# ============================================================
# 11. FEATURE EXTRACTION FROM RAW VALUES
# ============================================================

print(
    "\n[5/15] Extracting numerical features..."
)


numeric_columns = [
    "Memory",
    "RAM",
    "Battery",
    "Rear_Camera",
    "Front_Camera",
    "AI_Lens",
    "Mobile_Height",
    "Price"
]


for column in numeric_columns:

    if column in df.columns:

        df[column] = (
            df[column]
            .apply(extract_number)
        )


# ============================================================
# 12. REMOVE INVALID TARGET VALUES
# ============================================================

print(
    "\n[6/15] Cleaning price values..."
)

df["Price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
)

df = df.dropna(
    subset=["Price"]
)

df = df[
    df["Price"] > 0
]

print(
    f"Rows after price cleaning: {len(df)}"
)


# ============================================================
# 13. SAVE CLEANED DATA
# ============================================================

cleaned_data_path = os.path.join(
    OUTPUT_DIR,
    "cleaned_mobile_phone_data.csv"
)

df.to_csv(
    cleaned_data_path,
    index=False
)

print(
    f"\nCleaned dataset saved:"
)

print(
    cleaned_data_path
)


# ============================================================
# 14. SUMMARY STATISTICS
# ============================================================

print(
    "\n[7/15] Generating summary statistics..."
)

numeric_data = (
    df.select_dtypes(
        include=np.number
    )
)

print(
    "\nNumerical summary:"
)

print(
    numeric_data.describe()
)


# ============================================================
# 15. PRICE DISTRIBUTION
# ============================================================

print(
    "\n[8/15] Creating visualizations..."
)

plt.figure(
    figsize=(10, 6)
)

sns.histplot(
    data=df,
    x="Price",
    kde=True
)

plt.title(
    "Mobile Phone Price Distribution"
)

plt.xlabel(
    "Price"
)

plt.ylabel(
    "Number of Phones"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "price_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 16. NUMERICAL FEATURE VISUALIZATIONS
# ============================================================

features_for_plot = [
    "Memory",
    "RAM",
    "Battery",
    "Rear_Camera",
    "Front_Camera",
    "Mobile_Height"
]


for feature in features_for_plot:

    if feature not in df.columns:
        continue

    plt.figure(
        figsize=(10, 6)
    )

    sns.scatterplot(
        data=df,
        x=feature,
        y="Price"
    )

    plt.title(
        f"{feature} vs Mobile Phone Price"
    )

    plt.xlabel(
        feature
    )

    plt.ylabel(
        "Price"
    )

    plt.tight_layout()

    safe_name = (
        feature
        .lower()
        .replace(
            "_",
            "_"
        )
    )

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            f"{safe_name}_vs_price.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 17. CORRELATION ANALYSIS
# ============================================================

print(
    "\n[9/15] Performing correlation analysis..."
)

correlation = (
    numeric_data
    .corr()
)


if "Price" in correlation.columns:

    print(
        "\nCorrelation with Price:"
    )

    print(
        correlation["Price"]
        .sort_values(
            ascending=False
        )
    )


plt.figure(
    figsize=(12, 8)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "correlation_heatmap.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 18. PREPARE X AND Y
# ============================================================

print(
    "\n[10/15] Preparing machine learning data..."
)

X = df.drop(
    columns=["Price"]
)

y = df["Price"]


# ============================================================
# 19. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = (
    X.select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)

categorical_features = (
    X.select_dtypes(
        exclude=np.number
    )
    .columns
    .tolist()
)


print(
    "\nNumerical features:"
)

print(
    numeric_features
)

print(
    "\nCategorical features:"
)

print(
    categorical_features
)


# ============================================================
# 20. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 21. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(
    f"\nTraining records: {len(X_train)}"
)

print(
    f"Testing records : {len(X_test)}"
)


# ============================================================
# 22. LINEAR REGRESSION
# ============================================================

print(
    "\n[11/15] Training Linear Regression..."
)

linear_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = (
    linear_model.predict(
        X_test
    )
)


linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


# ============================================================
# 23. RANDOM FOREST
# ============================================================

print(
    "\n[12/15] Training Random Forest..."
)

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


random_forest_model.fit(
    X_train,
    y_train
)


rf_predictions = (
    random_forest_model.predict(
        X_test
    )
)


rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


# ============================================================
# 24. MODEL COMPARISON
# ============================================================

print(
    "\n[13/15] Comparing models..."
)

results = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Random Forest"
        ],
        "MAE": [
            linear_mae,
            rf_mae
        ],
        "RMSE": [
            linear_rmse,
            rf_rmse
        ],
        "R2 Score": [
            linear_r2,
            rf_r2
        ]
    }
)


print(
    "\n"
    + "=" * 75
)

print(
    "MODEL PERFORMANCE"
)

print(
    "=" * 75
)

print(
    results.to_string(
        index=False
    )
)


results_path = os.path.join(
    OUTPUT_DIR,
    "model_results.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ============================================================
# 25. FEATURE IMPORTANCE
# ============================================================

print(
    "\n[14/15] Extracting feature importance..."
)

rf_preprocessor = (
    random_forest_model
    .named_steps[
        "preprocessor"
    ]
)

rf_model = (
    random_forest_model
    .named_steps[
        "model"
    ]
)


feature_names = (
    rf_preprocessor
    .get_feature_names_out()
)


feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }
)


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print(
    "\nTop 20 important features:"
)

print(
    feature_importance
    .head(20)
    .to_string(
        index=False
    )
)


feature_importance_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

feature_importance.to_csv(
    feature_importance_path,
    index=False
)


# ============================================================
# 26. FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = (
    feature_importance
    .head(15)
    .sort_values(
        by="Importance"
    )
)


plt.figure(
    figsize=(12, 8)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title(
    "Top 15 Features Influencing Mobile Phone Price"
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "feature_importance.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 27. ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.7
)

minimum = min(
    y_test.min(),
    rf_predictions.min()
)

maximum = max(
    y_test.max(),
    rf_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title(
    "Actual vs Predicted Mobile Phone Price"
)

plt.xlabel(
    "Actual Price"
)

plt.ylabel(
    "Predicted Price"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "actual_vs_predicted.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 28. RESIDUAL ANALYSIS
# ============================================================

residuals = (
    y_test -
    rf_predictions
)


plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    rf_predictions,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title(
    "Residual Analysis"
)

plt.xlabel(
    "Predicted Price"
)

plt.ylabel(
    "Residual"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "residual_analysis.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 29. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODELS_DIR,
    "mobile_price_random_forest.pkl"
)

joblib.dump(
    random_forest_model,
    model_path
)


# ============================================================
# 30. FINAL RESULTS
# ============================================================

print(
    "\n[15/15] Saving final results..."
)


print(
    "\n"
    + "=" * 75
)

print(
    "FINAL PROJECT RESULTS"
)

print(
    "=" * 75
)

print(
    f"\nRandom Forest MAE  : ₹{rf_mae:,.2f}"
)

print(
    f"Random Forest RMSE : ₹{rf_rmse:,.2f}"
)

print(
    f"Random Forest R²   : {rf_r2:.4f}"
)


# ============================================================
# 31. BUSINESS RECOMMENDATIONS
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "BUSINESS RECOMMENDATIONS"
)

print(
    "=" * 75
)


for _, row in (
    feature_importance
    .head(10)
    .iterrows()
):

    print(
        f"{row['Feature']} "
        f"-> {row['Importance']:.4f}"
    )


print(
    "\nKey recommendation:"
)

print(
    "The organization should focus pricing strategy "
    "on the mobile phone specifications with the highest "
    "feature importance. Hardware specifications such as "
    "memory, RAM, processor characteristics, camera "
    "capabilities and battery capacity can be considered "
    "when positioning products and determining competitive prices."
)


# ============================================================
# 32. OUTPUT FILES
# ============================================================

print(
    "\n"
    + "=" * 75
)

print(
    "PROJECT COMPLETED SUCCESSFULLY"
)

print(
    "=" * 75
)

print(
    "\nGenerated files:"
)

print(
    "1. outputs/cleaned_mobile_phone_data.csv"
)

print(
    "2. outputs/model_results.csv"
)

print(
    "3. outputs/feature_importance.csv"
)

print(
    "4. outputs/models/mobile_price_random_forest.pkl"
)

print(
    "5. outputs/plots/price_distribution.png"
)

print(
    "6. outputs/plots/correlation_heatmap.png"
)

print(
    "7. outputs/plots/feature_importance.png"
)

print(
    "8. outputs/plots/actual_vs_predicted.png"
)

print(
    "9. outputs/plots/residual_analysis.png"
)

print(
    "\nAll processing completed successfully."
)