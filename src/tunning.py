import os
import joblib
import pandas as pd

from scipy.stats import randint, uniform

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC

from data_preprocessing import prepare_data


# ------------------------------
# Custom F1 scorer
# ------------------------------

f1_yes = make_scorer(
    f1_score,
    pos_label="Yes"
)


def tune_models(
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor
):
    """
    Tune multiple ML models using RandomizedSearchCV.
    """

    models = {

        "Logistic Regression": (

            LogisticRegression(max_iter=1000),

            {
                "classifier__C": uniform(0.01, 10),
                "classifier__solver": [
                    "liblinear",
                    "lbfgs"
                ]
            }
        ),

        "Random Forest": (

            RandomForestClassifier(
                random_state=42
            ),

            {
                "classifier__n_estimators":
                    randint(100, 400),

                "classifier__max_depth":
                    randint(3, 20),

                "classifier__min_samples_split":
                    randint(2, 10),

                "classifier__min_samples_leaf":
                    randint(1, 5)
            }
        ),

        "Gradient Boosting": (

            GradientBoostingClassifier(
                random_state=42
            ),

            {
                "classifier__n_estimators":
                    randint(50, 300),

                "classifier__learning_rate":
                    uniform(0.01, 0.2),

                "classifier__max_depth":
                    randint(2, 6)
            }
        ),

        "Support Vector Machine": (

            SVC(),

            {
                "classifier__C":
                    uniform(0.1, 10),

                "classifier__kernel":
                    [
                        "linear",
                        "rbf"
                    ],

                "classifier__gamma":
                    [
                        "scale",
                        "auto"
                    ]
            }
        )
    }

    results = []

    best_model = None
    best_model_name = None
    best_f1 = -1

    for model_name, (model, params) in models.items():

        print("=" * 60)
        print(f"Tuning {model_name}...")

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "classifier",
                    model
                )
            ]
        )

        search = RandomizedSearchCV(

            estimator=pipeline,

            param_distributions=params,

            n_iter=20,

            cv=5,

            scoring=f1_yes,

            random_state=42,

            n_jobs=-1
        )

        search.fit(
            X_train,
            y_train
        )

        tuned_model = search.best_estimator_

        y_pred = tuned_model.predict(X_test)

        # -------------------------
        # ROC AUC
        # -------------------------

        if hasattr(
            tuned_model,
            "predict_proba"
        ):

            y_score = tuned_model.predict_proba(
                X_test
            )[:, 1]

        else:

            y_score = tuned_model.decision_function(
                X_test
            )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            pos_label="Yes"
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label="Yes"
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label="Yes"
        )

        roc_auc = roc_auc_score(
            (y_test == "Yes").astype(int),
            y_score
        )

        results.append({

            "Model": model_name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1,

            "ROC-AUC": roc_auc

        })

        print("\nBest Parameters:")
        print(search.best_params_)

        print(
            f"Best CV F1 Score: "
            f"{search.best_score_:.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1

            best_model = tuned_model

            best_model_name = model_name

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(

        by=[
            "F1 Score",
            "ROC-AUC",
            "Accuracy"
        ],

        ascending=False

    ).reset_index(drop=True)

    return (
        results_df,
        best_model,
        best_model_name
    )


def save_best_model(model):

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        "models/best_model.pkl"
    )


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

    (
        results,
        best_model,
        best_model_name
    ) = tune_models(

        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor

    )

    print("\n")
    print("=" * 70)
    print("TUNED MODEL COMPARISON")
    print("=" * 70)

    print(results.round(4))

    print("\nFinal Best Model:")
    print(best_model_name)

    save_best_model(
        best_model
    )

    print("\nBest tuned model saved successfully!")
    print("Location: models/best_model.pkl")


if __name__ == "__main__":
    main()