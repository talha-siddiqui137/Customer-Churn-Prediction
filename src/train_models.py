from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from data_preprocessing import prepare_data


def get_models():
    """
    Create a dictionary of machine learning models.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "Naive Bayes": GaussianNB(),

        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        )
    }

    return models


def train_models(
    X_train,
    y_train,
    preprocessor
):
    """
    Train all machine learning models.
    """

    trained_models = {}

    models = get_models()

    for model_name, model in models.items():

        print(f"Training {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        trained_models[model_name] = pipeline

        print(f"{model_name} trained successfully.\n")

    return trained_models


def main():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    trained_models = train_models(
        X_train,
        y_train,
        preprocessor
    )

    print("=" * 50)
    print("All models trained successfully!")
    print("=" * 50)

    print("\nModels Trained:")

    for model_name in trained_models.keys():
        print(f"- {model_name}")


if __name__ == "__main__":
    main()