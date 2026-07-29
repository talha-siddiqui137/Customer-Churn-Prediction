import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load dataset from CSV file.
    """
    return pd.read_csv(file_path)

def churn_distribution(df):
    """
    Visualize customer churn distribution.
    """

    churn_counts = df["Churn"].value_counts()

    plt.figure(figsize=(6,4))

    plt.bar(
        churn_counts.index,
        churn_counts.values,
        color=["orange", "crimson"],
        edgecolor="black"
    )

    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.title("Customer Churn Distribution")

    plt.show()


def gender_vs_churn(df):
    """
    Visualize relationship between gender and churn.
    """

    gender_churn = pd.crosstab(
        df["gender"],
        df["Churn"]
    )

    gender_churn.plot(
        kind="bar",
        figsize=(6,4),
        color=["skyblue", "steelblue"],
        edgecolor="black"
    )

    plt.xlabel("Gender")
    plt.ylabel("Number of Customers")
    plt.title("Gender vs Customer Churn")

    plt.xticks(rotation=0)

    plt.show()

def contract_vs_churn(df):
    """
    Visualize relationship between contract type and churn.
    """

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    contract_churn.plot(
        kind="bar",
        figsize=(7,4),
        color=["mediumpurple", "purple"],
        edgecolor="black"
    )

    plt.xlabel("Contract Type")
    plt.ylabel("Number of Customers")
    plt.title("Contract Type vs Customer Churn")

    plt.xticks(rotation=0)

    plt.show()


def internet_service_vs_churn(df):
    """
    Visualize relationship between internet service and churn.
    """

    internet_churn = pd.crosstab(
        df["InternetService"],
        df["Churn"]
    )

    internet_churn.plot(
        kind="bar",
        figsize=(7,4),
        color=["deepskyblue", "dodgerblue"],
        edgecolor="black"
    )

    plt.xlabel("Internet Service")
    plt.ylabel("Number of Customers")
    plt.title("Internet Service vs Customer Churn")

    plt.xticks(rotation=0)

    plt.show()


def payment_method_vs_churn(df):
    """
    Visualize relationship between payment method and churn.
    """

    payment_churn = pd.crosstab(
        df["PaymentMethod"],
        df["Churn"]
    )

    payment_churn.plot(
        kind="bar",
        figsize=(8,5),
        color=["seagreen", "tomato"],
        edgecolor="black"
    )

    plt.xlabel("Payment Method")
    plt.ylabel("Number of Customers")
    plt.title("Payment Method vs Customer Churn")

    plt.xticks(rotation=10)
    plt.tight_layout()

    plt.show()


def tenure_distribution(df):
    """
    Visualize customer tenure distribution.
    """

    plt.figure(figsize=(8,4))

    plt.hist(
        df["tenure"],
        bins=30,
        color="teal",
        edgecolor="black"
    )

    plt.xlabel("Tenure (Months)")
    plt.ylabel("Number of Customers")
    plt.title("Customer Tenure Distribution")

    plt.show()


def tenure_vs_churn(df):
    """
    Visualize relationship between tenure and churn.
    """

    churn_yes = df[df["Churn"] == "Yes"]
    churn_no = df[df["Churn"] == "No"]

    plt.figure(figsize=(8,4))

    plt.hist(
        churn_no["tenure"],
        bins=30,
        alpha=0.7,
        color="green",
        edgecolor="black",
        label="No Churn"
    )

    plt.hist(
        churn_yes["tenure"],
        bins=30,
        alpha=0.7,
        color="red",
        edgecolor="black",
        label="Churn"
    )

    plt.xlabel("Tenure (Months)")
    plt.ylabel("Number of Customers")
    plt.title("Tenure Distribution by Churn")

    plt.legend()

    plt.show()


def monthly_charges_distribution(df):
    """
    Visualize distribution of monthly charges.
    """

    plt.figure(figsize=(8,4))

    plt.hist(
        df["MonthlyCharges"],
        bins=30,
        color="orange",
        edgecolor="black"
    )

    plt.xlabel("Monthly Charges")
    plt.ylabel("Number of Customers")
    plt.title("Monthly Charges Distribution")

    plt.show()

def monthly_charges_vs_churn(df):
    """
    Visualize relationship between monthly charges and churn.
    """

    churn_yes = df[df["Churn"] == "Yes"]
    churn_no = df[df["Churn"] == "No"]

    plt.figure(figsize=(8,4))

    plt.hist(
        churn_no["MonthlyCharges"],
        bins=30,
        alpha=0.7,
        color="green",
        edgecolor="black",
        label="No Churn"
    )

    plt.hist(
        churn_yes["MonthlyCharges"],
        bins=30,
        alpha=0.7,
        color="red",
        edgecolor="black",
        label="Churn"
    )

    plt.xlabel("Monthly Charges")
    plt.ylabel("Number of Customers")
    plt.title("Monthly Charges Distribution by Churn")

    plt.legend()

    plt.show()


def total_charges_vs_churn(df):
    """
    Visualize relationship between total charges and churn.
    """

    # Convert TotalCharges from string to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    churn_yes = df[df["Churn"] == "Yes"]
    churn_no = df[df["Churn"] == "No"]

    plt.figure(figsize=(8,4))

    plt.hist(
        churn_no["TotalCharges"].dropna(),
        bins=30,
        alpha=0.7,
        color="green",
        edgecolor="black",
        label="No Churn"
    )

    plt.hist(
        churn_yes["TotalCharges"].dropna(),
        bins=30,
        alpha=0.7,
        color="red",
        edgecolor="black",
        label="Churn"
    )

    plt.xlabel("Total Charges")
    plt.ylabel("Number of Customers")
    plt.title("Total Charges Distribution by Churn")

    plt.legend()

    plt.show()

def correlation_heatmap(df):
    """
    Display correlation heatmap of numerical features.
    """

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Convert Churn into numerical values
    df["Churn"] = df["Churn"].map(
        {
            "Yes": 1,
            "No": 0
        }
    )

    correlation = df[
        [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn"
        ]
    ].corr()


    plt.figure(figsize=(8,6))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")

    plt.show()
    
def main():

    df = load_data(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    churn_distribution(df)

    gender_vs_churn(df)

    contract_vs_churn(df)

    internet_service_vs_churn(df)

    payment_method_vs_churn(df)

    tenure_distribution(df)

    tenure_vs_churn(df)

    monthly_charges_distribution(df)

    monthly_charges_vs_churn(df)

    total_charges_vs_churn(df)

    correlation_heatmap(df)

if __name__ == "__main__":
    main()