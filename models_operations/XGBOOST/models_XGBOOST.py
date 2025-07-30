from xgboost import XGBClassifier


def get_xgBoost_model(X_train_combined, y_train):

    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    y_train = y_train.ravel()
    xgb_model.fit(X_train_combined, y_train)

    return xgb_model


def get_best_xgBoost_model(X_train_combined, y_train):

    xgb_model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.687,
        colsample_bytree=0.677,
        random_state=42
    )

    y_train = y_train.ravel()
    xgb_model.fit(X_train_combined, y_train)

    return xgb_model