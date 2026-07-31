from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def show_shape(df):
    """Display the number of rows and columns."""
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")


def show_info(df):
    """Display dataset information."""
    print(df.info())


def show_statistics(df):
    """Display summary statistics."""
    print(df.describe())


def check_missing_values(df):
    """Display missing values in each column."""
    print(df.isnull().sum())



def check_duplicates(df):
    """Display the number of duplicate rows."""
    print(f"Duplicate Rows: {df.duplicated().sum()}")


def show_columns(df):
    """Display all column names."""
    print("Columns:")
    for column in df.columns:
        print(column)


def show_data_types(df):
    """Display the data type of each column."""
    print(df.dtypes)


def show_unique_values(df):
    """
    Display unique values for categorical columns.
    """
    categorical_columns = df.select_dtypes(include="str").columns

    for column in categorical_columns:
        print(f"\n{column}")
        print("-" * 40)
        print(df[column].unique())
        print(f"Total Unique Values: {df[column].nunique()}")

def show_class_distribution(df):
    print(df["Churn"].value_counts())

def show_class_percentage(df):
    print(df["Churn"].value_counts(normalize=True) * 100)


def clean_data(df):
    """
    Clean the dataset by removing unnecessary columns,
    converting data types, and handling missing values.
    """

    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Remove customerID (it has no predictive value)
    df = df.drop(columns="customerID")

    # Convert TotalCharges from string to numeric
    # Invalid values (empty strings) become NaN
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Check missing values created after conversion
    missing_values = df["TotalCharges"].isnull().sum()

    if missing_values > 0:
        print(f"Missing values in TotalCharges: {missing_values}")

    # Replace missing TotalCharges with the median
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    return df


def split_features_target(df):
    """
    Separate features (X) and target variable (y).
    """

    X = df.drop(columns="Churn")
    y = df["Churn"]

    return X, y


def create_preprocessor(X):
    """
    Create a preprocessing pipeline for numerical
    and categorical features.
    """

    # Numerical columns
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # Categorical columns
    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns

    # Numerical preprocessing
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical preprocessing
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def prepare_data(file_path):
    """
    Complete data preparation pipeline.
    """

    # Load data
    df = load_data(file_path)

    # Clean data
    df = clean_data(df)

    # Split features and target
    X, y = split_features_target(df)

    # Create preprocessing pipeline
    preprocessor = create_preprocessor(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


def main():
    df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    show_shape(df)
    print()

    show_info(df)
    print()

    show_statistics(df)
    print()

    check_missing_values(df)
    print()

    check_duplicates(df)
    print()

    show_columns(df)
    print()

    show_data_types(df)
    print()

    show_unique_values(df)
    print()

    show_class_distribution(df)
    print()

    show_class_percentage(df)
    print()

    df = clean_data(df)

    show_data_types(df)
    print()

    check_missing_values(df)

    X, y = split_features_target(df)

    print("Features Shape:", X.shape)
    print("Target Shape:", y.shape)

    print("\nTarget Classes:")
    print(y.value_counts())

    preprocessor = create_preprocessor(X)
    print(preprocessor)

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    print(f"Training Features: {X_train.shape}")
    print(f"Testing Features : {X_test.shape}")
    print(f"Training Labels  : {y_train.shape}")
    print(f"Testing Labels   : {y_test.shape}")

    print("\nPreprocessor Created Successfully!")

if __name__ == "__main__":
    main()



# Add clean_data(df).
# Add split_features_target(df).
# Add create_preprocessor(X).
# Add prepare_data(file_path).