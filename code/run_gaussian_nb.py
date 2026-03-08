from sklearn.naive_bayes import GaussianNB

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
    model_name = "gaussian_nb"

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_res, y_train_res = apply_smote(X_train, y_train)

    model = GaussianNB()

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