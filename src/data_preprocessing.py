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

if __name__ == "__main__":
    main()