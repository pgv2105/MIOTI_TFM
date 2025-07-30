import optuna, numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


feature_cols = [
    'trade_signal',
    'kf_long_trend', 'kf_long_slope', 'kf_short_trend', 'kf_short_slope',
    'res_kf_long', 'res_kf_short',
    'rsi10', 'rsi24',
    'vol15', 'vol50',
    'ema12_26', 'macd_hist', 'dmi14',
    'sg15_trend', 'sg15_slope', 'res_sg15',
    'ss10_price', 'ss10_slope',
    'ss50_price', 'ss50_slope'
]

def get_features_optuna(df, min_feats=6,
                        n_random=400, n_tpe=9600, seed=42):

    X = df[feature_cols]
    y = df['signal']

    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('model', xgb.XGBClassifier(
            n_estimators=400, max_depth=3,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0,
            objective='binary:logistic', eval_metric='logloss',
            n_jobs=-1, random_state=seed))
    ])

    tscv = TimeSeriesSplit(n_splits=5)

    # ------------------------------------------------------------------
    def cv_macro_precision(Xsub):
        scores = []
        for tr, te in tscv.split(Xsub):
            pipe.fit(Xsub.iloc[tr], y.iloc[tr])
            pred = (pipe.predict_proba(Xsub.iloc[te])[:, 1] > 0.5).astype(int)
            scores.append(precision_score(y.iloc[te], pred, average='macro'))
        # jitter anti-draw result
        return np.mean(scores) + np.random.uniform(0, 1e-6)

    # ------------------------------------------------------------------
    tried_masks = set()         # avoid duplicates

    def objective(trial):
        mask = tuple(trial.suggest_int(f'f{i}', 0, 1) for i in range(len(feature_cols)))
        if mask.count(1) < min_feats or mask in tried_masks:
            return 0.0
        tried_masks.add(mask)
        chosen = [f for f, m in zip(feature_cols, mask) if m]
        return cv_macro_precision(X[chosen])

    # -------- 1ª FASE: 100-200 random trials ----------------------
    random_study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
        storage="sqlite:///feature_optuna.db",
        study_name="feature_search",
        load_if_exists=True
    )

    random_study.optimize(objective, n_trials=400, show_progress_bar=False)

    # -------- 2ª FASE: TPE multivariate over previous sampler ------

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            seed=42, multivariate=True, group=True, n_startup_trials=0),
        pruner=optuna.pruners.NopPruner(),
        storage="sqlite:///feature_optuna.db",
        study_name="feature_search",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=9600, show_progress_bar=True)

    # ------------------------- RESULTADOS -----------------------------
    best_mask = study.best_trial.params
    best_feats = [f for f, m in zip(feature_cols, best_mask.values()) if m]
    print(f"\nMejor subset ({len(best_feats)} features): {best_feats}")
    print("Precision macro (CV) =", round(study.best_value, 6))
    return best_feats, study.best_value
