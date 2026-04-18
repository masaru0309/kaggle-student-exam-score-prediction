import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import marimo as mo

    return mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. ライブラリのインポートとデータの読み込み

    まずは分析に必要なライブラリを読み込み、学習データ・originalデータ・テストデータ・提出用サンプルファイルを読み込む。
    今回は 02_ridge_feature_engineering の流れを発展させ、Optunaで得た XGBoost のパラメータも用いながら、stacking 的な構成で予測性能の向上を目指す。
    """)
    return


@app.cell
def _(pd):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import RidgeCV

    from xgboost import XGBRegressor

    # データの読み込み
    train = pd.read_csv(
        "kaggle_competitions/kaggle_student_test_scores_prediction/data/train.csv"
    )
    original_train = pd.read_csv(
        "kaggle_competitions/kaggle_student_test_scores_prediction/data/Exam_Score_Prediction_OriginalData.csv"
    )
    test = pd.read_csv(
        "kaggle_competitions/kaggle_student_test_scores_prediction/data/test.csv"
    )
    sample_submission = pd.read_csv(
        "kaggle_competitions/kaggle_student_test_scores_prediction/data/sample_submission.csv"
    )
    return (
        ColumnTransformer,
        KFold,
        OneHotEncoder,
        Pipeline,
        RidgeCV,
        StandardScaler,
        XGBRegressor,
        mean_squared_error,
        original_train,
        sample_submission,
        test,
        train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 特徴量エンジニアリング関数の定義

    まずは 02_ridge_feature_engineering と同様に追加特徴量を作成する。
    今回は `study_hours` の変換特徴量と、一部カテゴリ変数の順序特徴量を追加する。

    また、`study_hours` のビン分割は train・test・original で境界がずれないように、3つをまとめて処理する。
    これにより、データごとに意味の異なるビン番号が作られてしまう不自然さを防ぐ。
    """)
    return


@app.cell
def _(np, pd):
    def add_engineered_features(train_df, test_df, original_df):
        train_df = train_df.copy()
        test_df = test_df.copy()
        original_df = original_df.copy()

        # train・test・originalで同じ基準の特徴量を作るため、一度結合してから処理する
        all_df = pd.concat([train_df, test_df, original_df], axis=0, ignore_index=True)

        # study_hours の2乗特徴量
        all_df["study_hours_squared"] = all_df["study_hours"] ** 2

        # study_hours の対数変換
        all_df["log_study_hours"] = np.log1p(all_df["study_hours"].clip(lower=0))

        # study_hours の平方根変換
        all_df["sqrt_study_hours"] = np.sqrt(all_df["study_hours"].clip(lower=0))

        # study_hours を5分割したビン番号
        all_df["study_bin_num"] = pd.cut(all_df["study_hours"], bins=5, labels=False)
        all_df["study_bin_num"] = all_df["study_bin_num"].fillna(0).astype(int)

        # 順序情報を持つカテゴリ変数を数値化
        ordinal_maps = {
            "sleep_quality": {"poor": 0, "average": 1, "good": 2},
            "facility_rating": {"low": 0, "medium": 1, "high": 2},
            "exam_difficulty": {"easy": 0, "moderate": 1, "hard": 2},
        }

        all_df["sleep_quality_ord"] = all_df["sleep_quality"].map(ordinal_maps["sleep_quality"])
        all_df["facility_rating_ord"] = all_df["facility_rating"].map(ordinal_maps["facility_rating"])
        all_df["exam_difficulty_ord"] = all_df["exam_difficulty"].map(ordinal_maps["exam_difficulty"])

        # 元の長さで分割して返す
        train_len = len(train_df)
        test_len = len(test_df)

        train_fe = all_df.iloc[:train_len].copy()
        test_fe = all_df.iloc[train_len : train_len + test_len].copy()
        original_fe = all_df.iloc[train_len + test_len :].copy()

        return train_fe, test_fe, original_fe

    return (add_engineered_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. RidgeCVで使用するカラムと評価指標の準備

    次に1段目のモデルとして RidgeCV を学習させる。
    そのために、数値変数とカテゴリ変数のカラムを定義し、評価指標として RMSE を計算する関数を用意する。

    02_ridge_feature_engineering と同様に、まずは線形モデルで全体の傾向を捉え、その予測値を後段のモデルに渡す流れで進める。
    """)
    return


@app.cell
def _(mean_squared_error, np):
    # RidgeCVで使用する数値変数のカラムを指定
    ridge_numeric_cols = [
        "age",
        "study_hours",
        "class_attendance",
        "sleep_hours",
        "study_hours_squared",
        "log_study_hours",
        "sqrt_study_hours",
        "study_bin_num",
        "sleep_quality_ord",
        "facility_rating_ord",
        "exam_difficulty_ord",
    ]

    # RidgeCVで使用するカテゴリ変数のカラムを指定
    ridge_categorical_cols = [
        "gender",
        "course",
        "internet_access",
        "sleep_quality",
        "study_method",
        "facility_rating",
        "exam_difficulty",
    ]

    # RMSEを計算する関数
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    return ridge_categorical_cols, ridge_numeric_cols, rmse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. RidgeCVによる1段目モデルの学習

    ここでは、追加特徴量を含むデータからRidgeCVによる予測値を作り、その予測値を後段のモデルに渡すための準備を行う。

    このように、あるモデルの予測値を別のモデルの入力として使う考え方は、stacking の基本的な発想に近い。
    02_ridge_feature_engineering では RidgeCV の予測を XGBoost に渡していたが、今回はさらにその先のメタモデルまでつなげていく。
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    KFold,
    OneHotEncoder,
    Pipeline,
    RidgeCV,
    StandardScaler,
    add_engineered_features,
    np,
    original_train,
    ridge_categorical_cols,
    ridge_numeric_cols,
    rmse,
    test,
    train,
):
    # 目的変数を分離し、説明変数を作成
    X_ridgecv_base = train.drop(columns=["id", "exam_score"]).copy()
    X_ridgecv_base_test = test.drop(columns=["id"]).copy()
    y_ridgecv_base = train["exam_score"].copy()

    # originalデータも説明変数と目的変数に分ける
    X_ridgecv_base_original = original_train.drop(columns=["student_id", "exam_score"]).copy()
    y_ridgecv_base_original = original_train["exam_score"].copy()

    # 特徴量エンジニアリングを適用
    X_ridgecv_fe, X_test_ridgecv_fe, X_original_ridgecv_fe = add_engineered_features(
        X_ridgecv_base,
        X_ridgecv_base_test,
        X_ridgecv_base_original,
    )

    # RidgeCVで探索するalphaの候補を設定
    ridgecv_alphas = np.logspace(-3, 3, 25)

    # 5-foldの交差検証を設定
    kf_ridgecv = KFold(n_splits=5, shuffle=True, random_state=42)

    # OOF予測とテスト予測を保存する配列を用意
    ridgecv_oof_pred = np.zeros(len(X_ridgecv_fe))
    ridgecv_test_pred = np.zeros(len(X_test_ridgecv_fe))

    # 各foldで選ばれたbest alphaを保存するリスト
    ridgecv_best_alpha_list = []

    for ridgecv_fold, (ridgecv_train_idx, ridgecv_valid_idx) in enumerate(
        kf_ridgecv.split(X_ridgecv_fe, y_ridgecv_base), start=1
    ):
        # foldごとに学習用データと検証用データを分割
        X_ridgecv_train_fold = X_ridgecv_fe.iloc[ridgecv_train_idx].copy()
        X_ridgecv_valid_fold = X_ridgecv_fe.iloc[ridgecv_valid_idx].copy()
        y_ridgecv_train_fold = y_ridgecv_base.iloc[ridgecv_train_idx].copy()
        y_ridgecv_valid_fold = y_ridgecv_base.iloc[ridgecv_valid_idx].copy()

        # RidgeCV用の前処理を定義
        # 数値変数は標準化し、カテゴリ変数はOneHotEncoderで数値化する
        ridgecv_preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ridge_numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ridge_categorical_cols),
            ]
        )

        # 前処理 + RidgeCVをまとめたパイプラインを作成
        ridgecv_pipeline = Pipeline([
            ("preprocessor", ridgecv_preprocessor),
            ("model", RidgeCV(alphas=ridgecv_alphas, cv=5, scoring="neg_root_mean_squared_error")),
        ])

        ridgecv_pipeline.fit(X_ridgecv_train_fold, y_ridgecv_train_fold)

        # 検証データとテストデータを予測
        ridgecv_valid_pred = ridgecv_pipeline.predict(X_ridgecv_valid_fold)
        ridgecv_fold_test_pred = ridgecv_pipeline.predict(X_test_ridgecv_fe)

        ridgecv_oof_pred[ridgecv_valid_idx] = ridgecv_valid_pred
        ridgecv_test_pred += ridgecv_fold_test_pred / kf_ridgecv.n_splits

        # foldごとのスコアとalphaを記録
        ridgecv_fold_rmse = rmse(y_ridgecv_valid_fold, ridgecv_valid_pred)
        ridgecv_best_alpha = ridgecv_pipeline.named_steps["model"].alpha_
        ridgecv_best_alpha_list.append(ridgecv_best_alpha)

        print(
            f"RidgeCV Fold {ridgecv_fold}: "
            f"RMSE = {ridgecv_fold_rmse:.5f}, alpha = {ridgecv_best_alpha:.6f}"
        )

    # 全体のCVスコア
    ridgecv_cv_rmse = rmse(y_ridgecv_base, ridgecv_oof_pred)
    print(f"\nRidgeCV CV RMSE: {ridgecv_cv_rmse:.5f}")
    print(f"Mean best alpha: {np.mean(ridgecv_best_alpha_list):.6f}")
    return (
        X_original_ridgecv_fe,
        X_ridgecv_fe,
        X_test_ridgecv_fe,
        ridgecv_alphas,
        ridgecv_oof_pred,
        ridgecv_test_pred,
        y_ridgecv_base,
        y_ridgecv_base_original,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 学習データ全体で学習した RidgeCV により original データの予測値を作成する

    次に、学習データ全体を使って RidgeCV を学習し、そのモデルで original データに対する予測値を作成する。
    学習データ側には OOF 予測を使い、original データ側には学習データ全体で学習したモデルの予測を使うことで、後段のモデルに自然な形で `ridge_pred` を渡せるようにする。
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    RidgeCV,
    StandardScaler,
    X_original_ridgecv_fe,
    X_ridgecv_fe,
    ridge_categorical_cols,
    ridge_numeric_cols,
    ridgecv_alphas,
    y_ridgecv_base,
):
    # 学習データ全体でRidgeCVを学習
    ridgecv_full_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ridge_numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ridge_categorical_cols),
        ]
    )

    ridgecv_full_pipeline = Pipeline([
        ("preprocessor", ridgecv_full_preprocessor),
        ("model", RidgeCV(alphas=ridgecv_alphas, cv=5, scoring="neg_root_mean_squared_error")),
    ])

    ridgecv_full_pipeline.fit(X_ridgecv_fe, y_ridgecv_base)

    # originalデータに対する予測値を作成
    ridgecv_original_pred = ridgecv_full_pipeline.predict(X_original_ridgecv_fe)

    print("Full-data RidgeCV alpha:", ridgecv_full_pipeline.named_steps["model"].alpha_)
    return (ridgecv_original_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. RidgeCVの予測値を新しい特徴量として追加

    次に、RidgeCV で得られた予測値を `ridge_pred` として説明変数に追加する。
    この `ridge_pred` は、1段目モデルが捉えた線形的な傾向を要約した特徴量とみなすことができる。

    そのうえで、XGBoost に元の特徴量と追加特徴量の両方を入力し、さらに original データも学習に加えることで、より柔軟な非線形関係を学習させる。
    """)
    return


@app.cell
def _(
    X_original_ridgecv_fe,
    X_ridgecv_fe,
    X_test_ridgecv_fe,
    ridgecv_oof_pred,
    ridgecv_original_pred,
    ridgecv_test_pred,
    y_ridgecv_base,
):
    # RidgeCVの予測値を新しい特徴量として追加
    X_xgb_ridge_base = X_ridgecv_fe.copy()
    X_xgb_ridge_base_test = X_test_ridgecv_fe.copy()
    X_xgb_ridge_base_original = X_original_ridgecv_fe.copy()
    y_xgb_ridge = y_ridgecv_base.copy()

    X_xgb_ridge_base["ridge_pred"] = ridgecv_oof_pred
    X_xgb_ridge_base_test["ridge_pred"] = ridgecv_test_pred
    X_xgb_ridge_base_original["ridge_pred"] = ridgecv_original_pred
    return (
        X_xgb_ridge_base,
        X_xgb_ridge_base_original,
        X_xgb_ridge_base_test,
        y_xgb_ridge,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. RidgeCV予測を加えたXGBoostモデルの学習

    ここでは、元の特徴量に加えて `ridge_pred` と追加特徴量を使い、XGBoost を学習させる。
    さらに、各 fold の学習時には original データも加えることで、学習データ量を増やしながら予測性能の向上を狙う。

    今回は Optuna によって探索したハイパーパラメータを使っており、02_ridge_feature_engineering よりもチューニングを進めた構成になっている。
    つまり、1段目の RidgeCV が捉えた傾向を XGBoost に引き継ぎつつ、original データと tuned parameter も活用することで、より強い1段目モデルを作っている。
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    KFold,
    OneHotEncoder,
    Pipeline,
    XGBRegressor,
    X_xgb_ridge_base,
    X_xgb_ridge_base_original,
    X_xgb_ridge_base_test,
    np,
    pd,
    rmse,
    y_ridgecv_base_original,
    y_xgb_ridge,
):
    # XGBoostで使用する数値変数のカラムを指定
    xgb_ridge_numeric_cols = [
        "age",
        "study_hours",
        "class_attendance",
        "sleep_hours",
        "ridge_pred",
        "study_hours_squared",
        "log_study_hours",
        "sqrt_study_hours",
        "study_bin_num",
        "sleep_quality_ord",
        "facility_rating_ord",
        "exam_difficulty_ord",
    ]

    # XGBoostで使用するカテゴリ変数のカラムを指定
    xgb_ridge_categorical_cols = [
        "gender",
        "course",
        "internet_access",
        "sleep_quality",
        "study_method",
        "facility_rating",
        "exam_difficulty",
    ]

    # Optunaで得たXGBoostのベストパラメータ
    best_xgb_params = {
        "n_estimators": 1704,
        "learning_rate": 0.04492665346350148,
        "max_depth": 6,
        "min_child_weight": 8,
        "subsample": 0.9285967141489522,
        "colsample_bytree": 0.6648724233687988,
        "reg_alpha": 0.009003692107504988,
        "reg_lambda": 0.20396348568086267,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }

    print(best_xgb_params)

    # 前処理の定義
    # 数値変数はそのまま使用し、カテゴリ変数はOneHotEncoderで数値化する
    xgb_ridge_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", xgb_ridge_numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), xgb_ridge_categorical_cols),
        ]
    )

    # 5-foldの交差検証を設定
    kf_xgb_ridge = KFold(n_splits=5, shuffle=True, random_state=42)

    # OOF予測とテスト予測を保存する配列を用意
    xgb_ridge_oof_pred = np.zeros(len(X_xgb_ridge_base))
    xgb_ridge_test_pred = np.zeros(len(X_xgb_ridge_base_test))

    for xgb_ridge_fold, (xgb_ridge_train_idx, xgb_ridge_valid_idx) in enumerate(
        kf_xgb_ridge.split(X_xgb_ridge_base, y_xgb_ridge), start=1
    ):
        # foldごとに学習用データと検証用データを分割
        X_xgb_ridge_train_fold = X_xgb_ridge_base.iloc[xgb_ridge_train_idx].copy()
        X_xgb_ridge_valid_fold = X_xgb_ridge_base.iloc[xgb_ridge_valid_idx].copy()
        y_xgb_ridge_train_fold = y_xgb_ridge.iloc[xgb_ridge_train_idx].copy()
        y_xgb_ridge_valid_fold = y_xgb_ridge.iloc[xgb_ridge_valid_idx].copy()

        # originalデータを学習用データに追加
        X_xgb_ridge_train_all = pd.concat(
            [X_xgb_ridge_train_fold, X_xgb_ridge_base_original],
            axis=0,
            ignore_index=True,
        )
        y_xgb_ridge_train_all = pd.concat(
            [y_xgb_ridge_train_fold, y_ridgecv_base_original],
            axis=0,
            ignore_index=True,
        )

        # 前処理 + XGBoostをまとめたパイプラインを作成
        xgb_ridge_pipeline = Pipeline([
            ("preprocessor", xgb_ridge_preprocessor),
            ("model", XGBRegressor(**best_xgb_params)),
        ])

        xgb_ridge_pipeline.fit(X_xgb_ridge_train_all, y_xgb_ridge_train_all)

        # 検証データとテストデータを予測
        xgb_ridge_valid_pred = xgb_ridge_pipeline.predict(X_xgb_ridge_valid_fold)
        xgb_ridge_fold_test_pred = xgb_ridge_pipeline.predict(X_xgb_ridge_base_test)

        xgb_ridge_oof_pred[xgb_ridge_valid_idx] = xgb_ridge_valid_pred
        xgb_ridge_test_pred += xgb_ridge_fold_test_pred / kf_xgb_ridge.n_splits

        # foldごとのスコアを表示
        xgb_ridge_fold_rmse = rmse(y_xgb_ridge_valid_fold, xgb_ridge_valid_pred)
        print(f"Tuned XGBoost Fold {xgb_ridge_fold}: RMSE = {xgb_ridge_fold_rmse:.5f}")

    # 全体のCVスコア
    xgb_ridge_cv_rmse = rmse(y_xgb_ridge, xgb_ridge_oof_pred)
    print(f"\nTuned XGBoost CV RMSE: {xgb_ridge_cv_rmse:.5f}")
    return xgb_ridge_oof_pred, xgb_ridge_test_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. メタモデル用の特徴量を作成する

    次に、1段目モデルの予測値を使って 2段目のメタモデル用特徴量を作成する。
    今回は、

    - RidgeCV の予測値
    - tuned XGBoost の予測値

    を中心に使い、加えて一部の元特徴量も残しておく。

    これにより、1段目モデル同士の予測を組み合わせるだけでなく、元の特徴量情報も少し補助的に使いながら、最終予測を安定してまとめることを狙う。
    """)
    return


@app.cell
def _(
    X_ridgecv_fe,
    X_test_ridgecv_fe,
    pd,
    ridgecv_oof_pred,
    ridgecv_test_pred,
    xgb_ridge_oof_pred,
    xgb_ridge_test_pred,
    y_ridgecv_base,
):
    # メタモデル用の学習データを作成
    X_meta_base = pd.DataFrame({
        "xgb_ridge_pred": xgb_ridge_oof_pred,
        "ridge_pred": ridgecv_oof_pred,
        "study_hours": X_ridgecv_fe["study_hours"].values,
        "study_hours_squared": X_ridgecv_fe["study_hours_squared"].values,
        "class_attendance": X_ridgecv_fe["class_attendance"].values,
    })

    # メタモデル用のテストデータを作成
    X_meta_base_test = pd.DataFrame({
        "xgb_ridge_pred": xgb_ridge_test_pred,
        "ridge_pred": ridgecv_test_pred,
        "study_hours": X_test_ridgecv_fe["study_hours"].values,
        "study_hours_squared": X_test_ridgecv_fe["study_hours_squared"].values,
        "class_attendance": X_test_ridgecv_fe["class_attendance"].values,
    })

    y_meta_base = y_ridgecv_base.copy()
    return X_meta_base, X_meta_base_test, y_meta_base


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. RidgeCVによる2段目メタモデルの学習

    最後に、1段目モデルの予測値をまとめる 2 段目のメタモデルとして RidgeCV を学習させる。
    1段目では線形モデルと木モデルの両方を使っていたが、最後は RidgeCV で安定的に統合する。

    このように、

    - 1段目: RidgeCV + tuned XGBoost
    - 2段目: RidgeCV

    という形にすることで、stacking っぽい構成を比較的シンプルに実装している。
    """)
    return


@app.cell
def _(
    KFold,
    Pipeline,
    RidgeCV,
    StandardScaler,
    X_meta_base,
    X_meta_base_test,
    np,
    rmse,
    y_meta_base,
):
    # メタモデルで探索するalphaの候補を設定
    meta_ridgecv_alphas = np.logspace(-4, 2, 20)

    # 5-foldの交差検証を設定
    kf_meta_ridgecv = KFold(n_splits=5, shuffle=True, random_state=42)

    # OOF予測とテスト予測を保存する配列を用意
    meta_ridgecv_oof_pred = np.zeros(len(X_meta_base))
    meta_ridgecv_test_pred = np.zeros(len(X_meta_base_test))

    for meta_ridgecv_fold, (meta_ridgecv_train_idx, meta_ridgecv_valid_idx) in enumerate(
        kf_meta_ridgecv.split(X_meta_base, y_meta_base), start=1
    ):
        # foldごとに学習用データと検証用データを分割
        X_meta_ridgecv_train_fold = X_meta_base.iloc[meta_ridgecv_train_idx].copy()
        X_meta_ridgecv_valid_fold = X_meta_base.iloc[meta_ridgecv_valid_idx].copy()
        y_meta_ridgecv_train_fold = y_meta_base.iloc[meta_ridgecv_train_idx].copy()
        y_meta_ridgecv_valid_fold = y_meta_base.iloc[meta_ridgecv_valid_idx].copy()

        # 標準化 + RidgeCVをまとめたパイプラインを作成
        meta_ridgecv_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=meta_ridgecv_alphas, cv=5, scoring="neg_root_mean_squared_error")),
        ])

        meta_ridgecv_pipeline.fit(X_meta_ridgecv_train_fold, y_meta_ridgecv_train_fold)

        # 検証データとテストデータを予測
        meta_ridgecv_valid_pred = meta_ridgecv_pipeline.predict(X_meta_ridgecv_valid_fold)
        meta_ridgecv_fold_test_pred = meta_ridgecv_pipeline.predict(X_meta_base_test)

        meta_ridgecv_oof_pred[meta_ridgecv_valid_idx] = meta_ridgecv_valid_pred
        meta_ridgecv_test_pred += meta_ridgecv_fold_test_pred / kf_meta_ridgecv.n_splits

        # foldごとのスコアを表示
        meta_ridgecv_fold_rmse = rmse(y_meta_ridgecv_valid_fold, meta_ridgecv_valid_pred)
        print(f"Meta RidgeCV Fold {meta_ridgecv_fold}: RMSE = {meta_ridgecv_fold_rmse:.5f}")

    # 全体のCVスコア
    meta_ridgecv_cv_rmse = rmse(y_meta_base, meta_ridgecv_oof_pred)
    print(f"\nMeta RidgeCV CV RMSE: {meta_ridgecv_cv_rmse:.5f}")
    return (meta_ridgecv_test_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. 提出ファイルの作成

    最後に、2段目メタモデルで得られたテスト予測を提出形式に整え、CSVファイルとして保存する。
    02_ridge_feature_engineering と同様に、最終的には提出ファイルを作成して完了とする。
    """)
    return


@app.cell
def _(meta_ridgecv_test_pred, sample_submission):
    # 提出ファイルの作成
    submission = sample_submission.copy()
    submission["exam_score"] = meta_ridgecv_test_pred

    # CSVとして保存
    submission.to_csv(
        "kaggle_competitions/kaggle_student_test_scores_prediction/output/submission_final_original_aug_meta.csv",
        index=False,
    )

    print("\nSaved: submission_final_original_aug_meta.csv")
    return


if __name__ == "__main__":
    app.run()
