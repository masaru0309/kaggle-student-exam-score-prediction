# Kaggle - Playground Series: Student Exam Score Prediction

![Python](https://img.shields.io/badge/Python-3.12.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![RidgeCV](https://img.shields.io/badge/Meta%20Feature-RidgeCV-orange)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-purple)

## 1. Project Overview (プロジェクト概要)
本リポジトリは、Kaggle「Playground Series - Season 6 Episode 1」における解法コードです。  
生徒の学習・生活データから **exam_score** を予測する回帰タスク（評価指標：RMSE）に取り組みました。

本コンペはすでに終了していたため、**Late Submission** として取り組みました。  
そのため、手元の CV と Public / Private Leaderboard の両方を見ながら、仮説検証ベースで改善を進めています。

本プロジェクトでは、

- XGBoost による回帰ベースラインの構築
- RidgeCV の OOF prediction を用いた `ridge_pred` の導入
- `study_hours` 周辺を中心とした特徴量エンジニアリング
- original dataset を活用した data augmentation
- Meta RidgeCV による最終モデルの構築
- Optuna を用いた XGBoost のハイパーパラメータ探索

までを一通り検証しました。

また、AI は単なるコード生成ツールとしてではなく、仮説を整理し、検証案を比較するための対話的な補助として活用しています。  
一方で、特徴量の採否や最終モデルの選定は、CV と Leaderboard の挙動を見ながら自分で判断しました。

* Competition URL: [Playground Series - Season 6 Episode 1](https://www.kaggle.com/competitions/playground-series-s6e1)

---

## 2. What this project demonstrates
このプロジェクトでは、以下の点を示すことを意識しました。

- 回帰タスクにおけるベースライン構築能力
- OOF prediction を用いたメタ特徴量設計
- 特徴量エンジニアリングの仮説検証
- original dataset を活用した data-centric な改善
- 軽量スタッキングによる最終予測の補正
- Optuna を用いたハイパーパラメータ探索
- CV と Public / Private LB のズレを踏まえたモデル選定

---

## 3. Repository Structure (ファイル構成)
本リポジトリは、スコア改善の流れが分かるように、以下の 3 つの notebook で構成しています。

* [01_baseline_xgb.ipynb](./01_baseline_xgb.ipynb)  
  * XGBoost を用いた回帰ベースラインを構築。  
  * 前処理、KFold による検証、submission 作成までの基本フローを整理。

* [02_ridge_feature_engineering.ipynb](./02_ridge_feature_engineering.ipynb)  
  * RidgeCV による OOF 予測 `ridge_pred` を XGBoost の補助特徴量として導入。  
  * `study_hours_squared`、`log_study_hours`、`sqrt_study_hours`、`study_bin_num`、ordinal encoding など、実際に有効だった特徴量を追加して改善を検証。

* [03_original_aug_meta_model.ipynb](./03_original_aug_meta_model.ipynb)  
  * original dataset を train fold 側に追加する augmentation を導入。  
  * OriginalAug XGBoost を主軸とし、Meta RidgeCV による最終モデルを構築。  
  * XGBoost のハイパーパラメータ探索には Optuna を用いた。  
  * GitHub 掲載版では可読性のため、Optuna 探索そのものではなく、採用した最終パラメータを直接記載している。

---

## 4. Solution Approach by Notebook

### Notebook 1: Baseline XGBoost
まずは、XGBoost を用いたシンプルな回帰ベースラインを構築しました。  
カテゴリ変数は One-Hot Encoding を行い、`KFold (n_splits=5)` によって手元 CV を確認しながら、最小限の前処理で基準スコアを作成しています。

この notebook の目的は、以後の改善を評価するための**比較基準**を明確に作ることです。

---

### Notebook 2: RidgeCV + Feature Engineering
次の段階では、RidgeCV による OOF prediction を `ridge_pred` として作成し、XGBoost の特徴量として追加しました。  
これにより、決定木系モデルだけでは拾いにくい線形的な傾向を補助的に取り込むことを狙いました。

また、特徴量エンジニアリングとして以下を採用しました。

- `study_hours_squared`
- `log_study_hours`
- `sqrt_study_hours`
- `study_bin_num`
- `sleep_quality_ord`
- `facility_rating_ord`
- `exam_difficulty_ord`

特に `study_hours` 周辺の派生特徴量は比較的安定して有効であり、  
勉強時間の効果が単純な線形ではないことを示唆していました。

この notebook の目的は、**OOF prediction を用いた補助特徴量**と、  
**有効な FE の切り分け**を行うことです。

---

### Notebook 3: Original Data Augmentation + Meta RidgeCV
細かい FE の改善幅が小さくなってきたため、最終段階では **original dataset の活用**に進みました。

この段階では、

- original dataset を train fold 側にのみ追加する augmentation
- OriginalAug XGBoost
- RidgeCV
- Meta RidgeCV

を組み合わせた構成を採用しています。

最終的なメタ特徴量には、

- `originalaug_xgb_pred`
- `ridgecv_pred`
- `study_hours`
- `study_hours_squared`
- `class_attendance`

を用いました。

また、OriginalAug XGBoost のハイパーパラメータは、実験段階では **Optuna** を用いて探索しました。  
ただし、公開版 notebook では可読性を優先し、最終的に採用したパラメータを固定値で記載しています。

この notebook の目的は、**特徴量追加より一段抽象度の高い改善**、すなわち

- データソースの拡張
- メタモデルによる補正
- ハイパーパラメータ最適化

を通じて最終提出モデルを構築することです。

---

## 5. Original Dataset
本コンペの train / test データは、original dataset を学習した deep learning model により生成された synthetic data です。  
Kaggle 公式には、以下のように説明されています。

> The dataset for this competition (both train and test) was generated from a deep learning model trained on the Exam score prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

この説明に基づき、本プロジェクトでは original dataset を train fold にのみ追加する augmentation を検証しました。  
validation fold 側には original data を混ぜず、CV の妥当性を保つように設計しています。

* Original dataset URL: [Exam Score Prediction Dataset](https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset?select=Exam_Score_Prediction.csv)

---

## 6. Validation Strategy (検証設計)
本プロジェクトでは、主に **KFold (k=5)** により CV を確認しながら改善を進めました。  
また、実験の過程で

- CV は改善しているのに Private LB では悪化する
- Public LB だけを見ると過大評価される

といったケースも経験しました。

そのため、最終モデルの選定では

- 手元 CV
- Public LB
- Private LB

の 3 つを総合的に見て判断しています。

---

## 7. Results (結果)
* **Best CV RMSE:** 8.72405  
* **Best Public LB:** 8.67547  
* **Best Private LB:** 8.71041  
* **Private leaderboard rank:** 4319人中948位（上位約21.9%）

---

## 8. Learnings & Future Work

### 1. Feature engineering だけでは改善に限界がある
本コンペでは、`study_hours` 周辺の派生特徴量は有効でしたが、  
細かい FE を追加し続けても改善幅は徐々に小さくなりました。  
そのため、後半では FE の追加よりも、データやモデル構造の変更の方が重要だと学びました。

### 2. OOF prediction は強力な情報源になる
`ridge_pred` の導入により、別モデルの予測を特徴量として再利用する有効性を実感しました。  
単一モデルの改善だけでなく、予測そのものを再利用する発想の重要性を学びました。

### 3. Data-centric な改善は非常に有効だった
original dataset augmentation は、本プロジェクトにおいて最も大きな改善源の一つでした。  
細かい FE よりも、**学習データそのものの情報量を増やす** 方が効果的な場面があると学びました。

### 4. CV 改善がそのまま Private LB 改善につながるとは限らない
pseudo-labeling や一部 stacking では、CV が改善しても Private LB では悪化するケースがありました。  
この経験を通じて、過学習や評価設計の難しさを実践的に学ぶことができました。

### Future Work
今後は Playground Series だけでなく、メダル対象のより実戦的な Kaggle コンペティションにも取り組み、

- より多様なモデルの活用
- より堅牢な CV 設計
- pseudo-labeling や stacking の高度化
- feature engineering と data-centric improvement の使い分け

をさらに深め、銅メダル以上の獲得を目指したいと考えています。

---

## 9. References
This project was developed through my own experiments and validation, while also referring to public Kaggle notebooks for ideas on feature engineering, Ridge-based meta features, stacking, and data augmentation.

- [s6e1-8-56-xgboost-with-ridge-regression](https://www.kaggle.com/code/gourabr0y555/s6e1-8-56-xgboost-with-ridge-regression#2.-Feature-Engineering)
- [score-8-66-ridgecv-xgboost-modified-stacking](https://www.kaggle.com/code/vaibhavdlights/score-8-66-ridgecv-xgboost-modified-stacking)
- [score-8-69-plain-stacking-ensemble](https://www.kaggle.com/code/vaibhavdlights/score-8-69-plain-stacking-ensemble#Model-Training)
