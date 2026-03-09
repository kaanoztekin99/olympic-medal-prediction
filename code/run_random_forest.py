from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline

from utils import (
    load_dataset,
    split_data,
    apply_smote,
    compute_cv_roc_auc,
    evaluate_model,
    save_confusion_matrix,
    save_metrics,
)


def main():
    model_name = "random_forest"

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    cv_pipeline = Pipeline([
        ("smote", apply_smote(return_object=True)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ))
    ])

    cv_mean, cv_std = compute_cv_roc_auc(cv_pipeline, X_train, y_train)

    X_train_res, y_train_res = apply_smote(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    results = evaluate_model(model, X_test, y_test)
    results["cv_roc_auc_mean"] = float(cv_mean)
    results["cv_roc_auc_std"] = float(cv_std)

    save_confusion_matrix(results["confusion_matrix"], model_name)
    save_metrics(results, model_name)

    print(f"{model_name} finished.")
    print(results)


if __name__ == "__main__":
    main()