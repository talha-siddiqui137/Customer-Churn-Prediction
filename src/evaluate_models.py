
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from train_models import train_models
from data_preprocessing import prepare_data


def evaluate_models(
    trained_models,
    X_test,
    y_test
):
    """
    Evaluate all trained models and return a comparison table.
    """

    results = []

    best_model = None
    best_model_name = None
    best_f1 = -1

    for model_name, model in trained_models.items():

        print(f"Evaluating {model_name}...")

        y_pred = model.predict(X_test)

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

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(
                (y_test == "Yes").astype(int),
                y_prob
            )
        else:
            roc_auc = None

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = model_name

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["F1 Score", "ROC-AUC", "Accuracy"],
        ascending=False
    ).reset_index(drop=True)

    return (
        results_df,
        best_model,
        best_model_name
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

    trained_models = train_models(
        X_train,
        y_train,
        preprocessor
    )

    (
        results_df,
        best_model,
        best_model_name
    ) = evaluate_models(
        trained_models,
        X_test,
        y_test
    )

    print("\nModel Comparison")
    print("=" * 70)
    print(results_df.round(4))

    print("\nBest Model:")
    print(best_model_name)

    print("\nClassification Report")
    print("=" * 70)

    y_pred = best_model.predict(X_test)

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print("Confusion Matrix")
    print("=" * 70)

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


if __name__ == "__main__":
    main()