from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline

from utils import (
    load_dataset_for_sport,
    split_data,
    apply_smote,
    compute_cv_roc_auc,
    evaluate_model,
    save_confusion_matrix,
    save_metrics,
    save_roc_curve,
)


def main():
    sports = [
        "Swimming",
        "Basketball",
        "Gymnastics",
        "Athletics",
        "Fencing"
    ]

    for sport in sports:
        print(f"Training for {sport}")

        model_name = f"xgboost_{sport.lower()}"

        X, y = load_dataset_for_sport(sport)
        X_train, X_test, y_train, y_test = split_data(X, y)

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )

        cv_pipeline = Pipeline([
            ("smote", apply_smote(return_object=True)),
            ("model", model)
        ])

        cv_mean, cv_std = compute_cv_roc_auc(cv_pipeline, X_train, y_train)

        X_train_res, y_train_res = apply_smote(X_train, y_train)
        model.fit(X_train_res, y_train_res)

        results = evaluate_model(model, X_test, y_test)
        results["cv_roc_auc_mean"] = float(cv_mean)
        results["cv_roc_auc_std"] = float(cv_std)

        save_confusion_matrix(results["confusion_matrix"], model_name)
        save_roc_curve(model, X_test, y_test, model_name)
        save_metrics(results, model_name)

        print(f"{model_name} finished.")
        print(results)


if __name__ == "__main__":
    main()