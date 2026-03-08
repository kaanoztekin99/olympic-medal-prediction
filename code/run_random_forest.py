from sklearn.ensemble import RandomForestClassifier

from utils import *

def main():
    model_name = "random_forest"

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_res, y_train_res = apply_smote(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    cv_mean, cv_std = compute_cv_roc_auc(model, X_train_res, y_train_res)

    model.fit(X_train_res, y_train_res)

    results = evaluate_model(model, X_test, y_test)
    results["cv_roc_auc_mean"] = cv_mean
    results["cv_roc_auc_std"] = cv_std

    save_confusion_matrix(results["confusion_matrix"], model_name)
    save_metrics(results, model_name)

    print(results)


if __name__ == "__main__":
    main()