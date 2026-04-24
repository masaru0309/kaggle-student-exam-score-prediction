# Kaggle - Playground Series: Student Exam Score Prediction

![Python](https://img.shields.io/badge/Python-3.12.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![RidgeCV](https://img.shields.io/badge/Meta%20Feature-RidgeCV-orange)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-purple)

## 1. Project Overview (プロジェクト概要)
本リポジトリは、Kaggle「Playground Series - Season 6 Episode 1」における解法コードです。  
生徒の学習記録や生活データ等から**exam_score**を予測する回帰タスク（評価指標：RMSE）に取り組みました。  

### Results (結果)
* **Best CV RMSE:** 8.72405  
* **Best Public LB:** 8.67547  
* **Best Private LB:** 8.71041  
* **Private leaderboard rank:** 4319人中948位（上位約21.9%）

本コンペはすでに終了していたため、**Late Submission** として取り組みました。  
そのため、CVスコアとPublic/Private Leaderboardの両方を見ながら、スコアの向上に取り組みました。

また、AIは単なるコード生成ツールとしてではなく、仮説を整理し、検証案を比較するための対話的な補助役として活用しています。  
一方で、特徴量の採否や最終モデルの選定は、CVとLeaderboardを見ながら自分で判断しました。

* Competition URL: [Playground Series - Season 6 Episode 1](https://www.kaggle.com/competitions/playground-series-s6e1)

---

## 2. Repository Structure (ファイル構成)
本リポジトリは以下の3つのnotebookの順でスコア改善に取り組みました。

* [01_baseline_xgb.ipynb](./01_baseline_xgb.ipynb)  
  * XGBoostを用いたベースラインを構築。  
  * 前処理、KFoldによる検証、submission作成までの基本手順を整理。

* [02_ridge_feature_engineering.ipynb](./02_ridge_feature_engineering.ipynb)  
  * RidgeCVによるOOFの予測値`ridge_pred`をXGBoostの特徴量の１つとして導入。  
  * `study_hours_squared`、`log_study_hours`、`sqrt_study_hours`、`study_bin_num`、ordinal encoding など、実際に有効だった特徴量を追加して改善を検証。

* [03_original_aug_meta_model.ipynb](./03_original_aug_meta_model.ipynb)  
  * original datasetをtrain fold側に追加し、train foldを拡張する。  
  * OriginalAug XGBoostとRidgeCVの予測値を主軸とした、Meta RidgeCVによる最終モデルを構築。  
  * XGBoostのハイパーパラメータ探索にはOptunaを用いた。（GitHub掲載版ではOptuna探索そのものではなく、採用した最終パラメータを直接記載している。）

---

## 3. Solution Approach by Notebook

### Notebook 1: Baseline XGBoost
まずは、XGBoost を用いたシンプルな回帰ベースラインを構築しました。  
カテゴリ変数はOne-Hot Encodingを行い、`KFold (n_splits=5)` によってCVスコアを確認しながら、最小限の前処理で作成しています。

この notebook の目的は、以後の改善を評価するための**比較基準**を明確に作ることです。

---

### Notebook 2: RidgeCV + Feature Engineering
次の段階では、RidgeCVによる予測値を`ridge_pred`として作成し、XGBoostの特徴量の１つとして追加しました。この手法はスタッキングの構造にかなり近いです。実際はRidgeCVの予測結果以外にも、もとからあった`study_hour`のような特徴量や`study_hours_squared`のような新規で作った特徴量を併せて XGBoost に学習させているため、スタッキングとは正式に呼べません。

これにより、XGBoostのような決定木系モデルだけでは拾いにくい線形的な傾向や特徴も補助的に取り込むことを狙いとしました。

また、特徴量エンジニアリングとして以下を採用しました。

- `study_hours_squared`
- `log_study_hours`
- `sqrt_study_hours`
- `study_bin_num`
- `sleep_quality_ord`
- `facility_rating_ord`
- `exam_difficulty_ord`

特に`study_hours`周辺の派生特徴量は比較的安定して有効であり、勉強時間の効果が非線形的で、はたまた離散的であるように感じました。

このnotebookの目的は、**別モデルの予測値を用いた補助特徴量の追加**と、**有効なFEの選定**を行うことです。

---

### Notebook 3: Original Dataset Augmentation + Two-Stage Meta Model
最終notebookでは細かい特徴量追加だけでは改善幅が小さくなってきたため、**train foldの使い方**と**モデル構造そのものを見直す方針**に切り替えました。

本コンペのtrain/testデータは、もともとの**Exam Score Prediction dataset**を学習したdeep learning modelから生成された合成データです。  
つまり、今回のコンペには「元になったデータセット」すなわちoriginal dataset が別途存在しており、Kaggle公式でもそのoriginal datasetを学習に活用してよいことが明記されています。

本プロジェクトでは、このoriginal datasetを追加のtrain foldとして利用し、予測性能が改善するかを検証しました。  

- 学習: competition train fold + original dataset
- 検証: competition validation fold

という形にしました。こうした理由としては、**original datasetが入っていない純粋なvalidation foldでCVスコアを測るため**というのが一番大きいです。  
もしvalidation foldにoriginal datasetを入れてしまえば、本番であるtestデータに近い形でCVスコアを測ることができなくなってしまいます。十分なtrain foldで学習させたあと、CVスコアを確認する際、testデータと構造が近いであろうoriginal datasetが入っていない純粋なvalidation foldで検証、評価するという方式を取りたかったためです。

このnotebookの構成は、大きく3段階です。

#### 1. RidgeCV による補助予測の作成
まず、Notebook2と同様にRidgeCVを学習し、予測値を`ridge_pred`として作成しました。ただし今回はもとからある特徴量だけでなく、新規で作成した特徴量を追加して学習させました。  
この`ridge_pred`は前回同様、線形的な傾向や情報を取り込むための特徴量です。

#### 2. OriginalAug XGBoost
次に、XGBoost の学習時にoriginal datasetをtrain fold側へ追加し、train foldの量をを拡張しました。  
さらに、特徴量には元の特徴量群に加えて、RidgeCV で作成した`ridge_pred`も含めています。

#### 3. 二段階メタモデルによる最終予測
最後に、1段目で得られた予測結果を使って、2段目のRidgeCVによる最終予測を行いました。  
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

## 4. Validation Strategy (検証設計)
本プロジェクトでは、主に **KFold (k=5)** により CV を確認しながら改善を進めました。  
また、実験の過程で

- CV は改善しているのにPrivate LBでは悪化する（いわゆる過学習が起きている）
- Public LB だけを見ると過大評価される

といったケースも経験しました。

そのため、最終モデルの選定では

- CVスコア
- Public LB
- Private LB

の3つを総合的に見て判断しています。

---

## 5. Learnings & Future Work

### 1. Feature engineering だけでは改善に限界がある
本コンペでは、`study_hours`  に関連する特徴量を新しく作成することは有効でしたが、細かいFEを追加し続けても改善幅は小さくなり天井があるようでした。  
そのため、後半では FE の追加よりも、モデル構造の変更のやハイパーパラメータのチューニングの方が重要だと学びました。

### 2. 別モデルの予測値は強力な特徴量になる
`ridge_pred`の導入により、別モデルの予測を特徴量として利用する有効性をかなり実感しました。  
特に決定木系のような非線形モデルと、Ridgeなどのような線形モデルはそれぞれ弱点をスタッキングを通して補完しあうことができることを学びました。

### 3. 学習データの拡張も非常に有効であった
original datasetによるtrain foldの拡張は、本プロジェクトにおいて最も大きな改善源の一つでした。  
細かいFEよりも、**train foldそのものの情報量を増やす**方が効果的な場面があると学びました。ただし、今回のコンペのようなoriginal dataasetがある場合には有効であったため、ほかのコンペで再現性があるかどうかは正直微妙です。


### Future Work

今回のコンペを通じて、**上位者のNotebooksやDiscussionを早い段階で確認する**ことの重要性を強く実感しました。  
実際には、新規特徴量を何度も作成・調整し、そのたびにCVスコアを確認する作業を繰り返していましたが、途中でスコア改善が頭打ちになりました。そこで上位者のNotebooksやDiscussionを確認したところ、スタッキングを取り入れたモデル設計やoriginal datasetを活用したtrain foldの拡張など、自力の試行錯誤だけでは到達しづらい発想が数多く存在していることに気づきました。  
これまでは、上位解法を参考にすることは自分の力にならないのではないかと考え、意図的に避けていました。  
しかし今回の経験を通じて、**重要なのは答えをそのまま写すことではなく、優れた発想を理解し、それを自分の実験に落とし込み、自分なりに改善すること**であると学びました。  
今後はPlayground Seriesのような表形式コンペだけでなく、上位者の考え方も積極的に学びながら、メダル対象のより実戦的なKaggleコンペティションにも取り組みたいと考えています。


---

## 6. References
このプロジェクトは、私自身の実験と検証を通じて開発されたものであり、特徴量エンジニアリング、スタッキング、データ拡張に関するアイデアについては、Kaggleの公開Notebookなども参考にしました。

- [s6e1-8-56-xgboost-with-ridge-regression](https://www.kaggle.com/code/gourabr0y555/s6e1-8-56-xgboost-with-ridge-regression#2.-Feature-Engineering)
- [score-8-66-ridgecv-xgboost-modified-stacking](https://www.kaggle.com/code/vaibhavdlights/score-8-66-ridgecv-xgboost-modified-stacking)
- [score-8-69-plain-stacking-ensemble](https://www.kaggle.com/code/vaibhavdlights/score-8-69-plain-stacking-ensemble#Model-Training)
