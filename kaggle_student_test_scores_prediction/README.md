# Kaggle - Playground Series: Student Exam Score Prediction

![Python](https://img.shields.io/badge/Python-3.12.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![RidgeCV](https://img.shields.io/badge/Meta%20Feature-RidgeCV-orange)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-purple)

## 1. Project Overview (プロジェクト概要)
本リポジトリは、Kaggle「Playground Series - Season 6 Episode 1」における解法コードです。  
生徒の学習データや生活データ等から**exam_score**を予測する回帰タスク（評価指標：RMSE）に取り組みました。  

### Results (結果)
* **Best CV RMSE:** 8.72405  
* **Best Public LB:** 8.67547  
* **Best Private LB:** 8.71041  
* **Private leaderboard rank:** 4319人中948位（上位約21.9%）

本コンペはすでに終了していたため、**Late Submission** として取り組みました。  
そのため、CVスコアとPublic/Private Leaderboardの両方を見ながら、スコアの向上に取り組みました。

また、AI は単なるコード生成ツールとしてではなく、仮説を整理し、検証案を比較するための対話的な補助役として活用しています。  
一方で、特徴量の採否や最終モデルの選定は、CVとLeaderboardを見ながら自分で判断しました。

* Competition URL: [Playground Series - Season 6 Episode 1](https://www.kaggle.com/competitions/playground-series-s6e1)

---

## 2. What this project demonstrates
このプロジェクトでは、以下の点を示すことを意識しました。

- 回帰タスクにおけるベースライン構築能力
- RidgeCV の OOF prediction を用いた `ridge_pred` の導入
- `study_hours` 周辺を中心とした特徴量エンジニアリングの仮説検証
- original dataset を活用した学習データの拡張（5. Original Dataset にて後述する）
- スタッキングの構造を模したモデルによる予測の補正
- Optuna を用いたハイパーパラメータ探索

---

## 3. Repository Structure (ファイル構成)
本リポジトリは以下の3つのnotebookの順でスコア改善に取り組みました。

* [01_baseline_xgb.ipynb](./01_baseline_xgb.ipynb)  
  * XGBoostを用いた回帰ベースラインを構築。  
  * 前処理、KFoldによる検証、submission作成までの基本手順を整理。

* [02_ridge_feature_engineering.ipynb](./02_ridge_feature_engineering.ipynb)  
  * RidgeCVによるOOFの予測値 `ridge_pred` をXGBoostの特徴量の１つとして導入。  
  * `study_hours_squared`、`log_study_hours`、`sqrt_study_hours`、`study_bin_num`、ordinal encoding など、実際に有効だった特徴量を追加して改善を検証。

* [03_original_aug_meta_model.ipynb](./03_original_aug_meta_model.ipynb)  
  * original dataset を train fold 側に追加し、学習データを拡張する。  
  * OriginalAug XGBoost を主軸とし、Meta RidgeCV による最終モデルを構築。  
  * XGBoost のハイパーパラメータ探索には Optuna を用いた。  
  * GitHub 掲載版では Optuna 探索そのものではなく、採用した最終パラメータを直接記載している。

---

## 4. Solution Approach by Notebook

### Notebook 1: Baseline XGBoost
まずは、XGBoost を用いたシンプルな回帰ベースラインを構築しました。  
カテゴリ変数は One-Hot Encoding を行い、`KFold (n_splits=5)` によってCVスコア を確認しながら、最小限の前処理で基準スコアを作成しています。

この notebook の目的は、以後の改善を評価するための**比較基準**を明確に作ることです。

---

### Notebook 2: RidgeCV + Feature Engineering
次の段階では、RidgeCV による OOF prediction を `ridge_pred` として作成し、XGBoost の特徴量として追加しました。この手法はスタッキングの構造にかなり近いです。実際は RidgeCV の予測結果以外にも、もとからあった `study_hour`のような特徴量や `study_hours_squared` のような新規で作った特徴量を併せて XGBoost に学習させているため、スタッキングとは正式に呼べません。

これにより、XGBoost のような決定木系モデルだけでは拾いにくい線形的な傾向を補助的に取り込むことを狙いました。

また、特徴量エンジニアリングとして以下を採用しました。

- `study_hours_squared`
- `log_study_hours`
- `sqrt_study_hours`
- `study_bin_num`
- `sleep_quality_ord`
- `facility_rating_ord`
- `exam_difficulty_ord`

特に `study_hours` 周辺の派生特徴量は比較的安定して有効であり、勉強時間の効果が単純な線形ではないことを示唆していました。

この notebook の目的は、**OOF prediction を用いた補助特徴量**と、  
**有効な FE の切り分け**を行うことです。

---

### Notebook 3: Original Dataset Augmentation + Two-Stage Meta Model
最終 notebook では、細かい特徴量追加だけでは改善幅が小さくなってきたため、  
**学習データの使い方とモデル構造そのものを見直す方針**に切り替えました。

本コンペの train / test データは、もともとの **Exam Score Prediction dataset** を学習した deep learning model から生成された合成データです。  
つまり、今回のコンペには「元になったデータセット」つまり original dataset が別途存在しており、Kaggle 公式でもその original dataset を学習に活用してよいことが明記されています。

本プロジェクトでは、この original dataset を追加の学習データとして利用し、予測性能が改善するかを検証しました。  
ただし、validation fold 側には original dataset を混ぜず、

- 学習: competition train fold + original dataset
- 検証: competition validation fold

という形にすることで、CV の妥当性を保つように設計しています。

この notebook の構成は、大きく3段階です。

#### 1. RidgeCV による補助予測の作成
まず、Notebook 2 と同様に RidgeCV を学習し、OOF prediction を `ridge_pred` として作成しました。ただし今回はもとからある特徴量だけでなく、新規で作成した特徴量を追加して学習させました。  
この `ridge_pred` は、決定木系モデルだけでは拾いにくい線形的な傾向を補助的に取り込むための特徴量です。

#### 2. OriginalAug XGBoost
次に、XGBoost の学習時に original dataset を train fold 側へ追加し、学習データを拡張しました。  
さらに、入力特徴量には元の特徴量群に加えて、RidgeCV で作成した `ridge_pred` も含めています。

つまり、この段階の XGBoost は、

- competition train fold
- original dataset
- `ridge_pred` を含む特徴量群

を使って学習する構成になっています。

#### 3. 二段階メタモデルによる最終予測
最後に、1段目で得られた予測結果を使って、2段目の RidgeCV による最終予測を行いました。  
2段目の入力には、

- `originalaug_xgb_pred`
- `ridgecv_pred`

という1段目モデルの予測に加えて、

- `study_hours`
- `study_hours_squared`
- `class_attendance`

という一部の元特徴量も加えています。

つまり、この notebook の最終モデルは、  
**1段目モデルの予測結果と、一部の強い元特徴量を組み合わせて補正する二段階メタモデル**です。

また、OriginalAug XGBoost のハイパーパラメータは、実験段階では **Optuna** を用いて探索しました。  
ただし、公開版 notebook では可読性を優先し、最終的に採用したパラメータを固定値として記載しています。

* Original dataset URL: [Exam Score Prediction Dataset](https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset?select=Exam_Score_Prediction.csv)

---

## 6. Validation Strategy (検証設計)
本プロジェクトでは、主に **KFold (k=5)** により CV を確認しながら改善を進めました。  
また、実験の過程で

- CV は改善しているのに Private LB では悪化する（いわゆる過学習が起きている）
- Public LB だけを見ると過大評価される

といったケースも経験しました。

そのため、最終モデルの選定では

- CVスコア
- Public LB
- Private LB

の 3 つを総合的に見て判断しています。

---

## 7. Learnings & Future Work

### 1. Feature engineering だけでは改善に限界がある
本コンペでは、`study_hours` 周辺の派生特徴量は有効でしたが、細かい FE を追加し続けても改善幅は徐々に小さくなりました。  
そのため、後半では FE の追加よりも、データやモデル構造の変更の方が重要だと学びました。

### 2. OOF prediction は強力な情報源になる
`ridge_pred` の導入により、別モデルの予測を特徴量として利用する有効性を実感しました。  
特に決定木系のような非線形モデルと、Ridgeなどのような線形モデルはそれぞれ弱点をスタッキングを押して補完しあうことができることを学びました。

### 3. Data-centric な改善は非常に有効だった
original datasetによる学習データの拡張は、本プロジェクトにおいて最も大きな改善源の一つでした。  
細かいFEよりも、**学習データそのものの情報量を増やす** 方が効果的な場面があると学びました。


### Future Work
今後は Playground Seriesのような表形式コンペだけでなく、メダル対象のより実戦的な Kaggle コンペにも取り組みたいと考えています。  
上位者のNotebookやDiscussionを柔軟に取り入れながら、スキルの習得をさらに深め、銅メダル以上の獲得を目指したいと考えています。

---

## 8. References
このプロジェクトは、私自身の実験と検証を通じて開発されたものであり、特徴量エンジニアリング、スタッキング、データ拡張に関するアイデアについては、Kaggleの公開Notebookなども参考にしました。

- [s6e1-8-56-xgboost-with-ridge-regression](https://www.kaggle.com/code/gourabr0y555/s6e1-8-56-xgboost-with-ridge-regression#2.-Feature-Engineering)
- [score-8-66-ridgecv-xgboost-modified-stacking](https://www.kaggle.com/code/vaibhavdlights/score-8-66-ridgecv-xgboost-modified-stacking)
- [score-8-69-plain-stacking-ensemble](https://www.kaggle.com/code/vaibhavdlights/score-8-69-plain-stacking-ensemble#Model-Training)
