# Kaggle - Playground Series: Student Exam Score Prediction

![Python](https://img.shields.io/badge/Python-3.12.12-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![RidgeCV](https://img.shields.io/badge/Meta%20Model-RidgeCV-orange)

## 1. Project Overview (プロジェクト概要)
本リポジトリは、Kaggle「Playground Series - Season 6 Episode 1」における解法コードです。  
生徒の学習・生活データから **exam_score** を予測する回帰タスク（評価指標：RMSE）に取り組みました。

本プロジェクトでは、単なるベースライン構築にとどまらず、

- XGBoost を用いた回帰ベースラインの作成
- Ridge の OOF 予測を用いたメタ特徴量 (`ridge_pred`) の導入
- study_hours 周辺の特徴量エンジニアリング
- original dataset を用いた data augmentation
- Meta RidgeCV による軽量スタッキング

までを一通り検証し、最終モデルを構築しました。

また開発過程では、AI を単なるコード生成ツールとしてではなく、仮説検証を加速するための対話的パートナーとして活用しました。  
一方で、特徴量の採否やモデル構成の選定は、CV・Public LB・Private LB の挙動を見ながら自分で判断しています。

* 当該 Kaggle competition URL : https://www.kaggle.com/competitions/playground-series-s6e1

---

## 2. Repository Structure (ファイル構成)
本リポジトリは、スコア改善の流れが分かるように、以下の3つの notebook で構成しています。

* [01_baseline_xgb.ipynb](./01_baseline_xgb.ipynb)  
  * XGBoost を用いた回帰ベースラインを構築。  
  * 前処理、KFold による検証、submission 作成までの基本フローを整理。

* [02_ridge_feature_engineering.ipynb](./02_ridge_feature_engineering.ipynb)  
  * RidgeCV による OOF 予測 `ridge_pred` を作成し、XGBoost の補助特徴量として導入。  
  * `study_hours_squared` など、実際に有効だった特徴量エンジニアリングを検証。

* [03_final_original_aug_meta.ipynb](./03_final_original_aug_meta.ipynb)  
  * original dataset を train fold 側に追加する augmentation を導入。  
  * OriginalAug XGBoost と RidgeCV を組み合わせ、Meta RidgeCV による最終モデルを構築。

---

## 3. Solution Approach (解法アプローチ)

### Baseline Modeling
まずは XGBoost による回帰ベースラインを構築し、CV と LB の基準値を作成しました。  
カテゴリ変数は One-Hot Encoding を行い、`KFold (n_splits=5)` により汎化性能を確認しました。

### Feature Engineering
本コンペでは、特に **study_hours** 周辺の特徴量が有効でした。  
最終的に有効と判断した特徴量群は以下です。

- `study_hours_squared`
- `log_study_hours`
- `sqrt_study_hours`
- `study_bin_num`
- `sleep_quality_ord`
- `facility_rating_ord`
- `exam_difficulty_ord`

特に `study_hours_squared` は比較的安定して改善に寄与し、  
「勉強時間の効果は単純な線形ではなく、非線形な側面を持つ」という仮説を支持する結果になりました。

### Meta Feature: Ridge Prediction
線形的な傾向を捉えるために RidgeCV を別途学習し、OOF prediction を `ridge_pred` として XGBoost に追加しました。  
これにより、決定木系モデルだけでは拾いにくい線形的な構造を補助的に取り込むことを狙いました。

### Data Augmentation with Original Dataset
細かい特徴量追加だけでは改善幅が頭打ちになったため、  
competition train とは別に用意された **original dataset** を train fold 側にのみ追加する augmentation を試しました。

- 学習: competition train fold + original dataset
- 検証: competition valid fold のみ

という設計にすることで、CV の妥当性を保ちながら学習データの情報量を増やしました。  
この original augmentation は、本プロジェクトの中でも特に有効だった改善要素の一つです。

### Final Meta Model
最終モデルでは、

- OriginalAug XGBoost
- RidgeCV

の予測を組み合わせ、  
さらに `study_hours`, `study_hours_squared`, `class_attendance` を加えた **Meta RidgeCV** により最終予測を行いました。

単純平均ではなく、各予測と強い特徴量を踏まえた補正を行うことで、最終スコアの改善を狙いました。

---

## 4. Validation Strategy (検証設計)
* **KFold (k=5)** を採用。  
  各 fold ごとの RMSE を確認しながら、CV と Public / Private LB の挙動を比較しました。

* 実験の過程で、CV が改善しても Private LB で再現しないケース（例: pseudo-labeling）も確認できました。  
  そのため、最終モデルの選定では「CV が良いこと」だけでなく、**Private LB での再現性** も重視しました。

---

## 5. Learnings & Future Work

### 1. Feature engineering だけでは限界がある
当初は特徴量エンジニアリングを中心に改善を試みましたが、  
一定以上は微小な改善にとどまりました。  
その結果、モデルや学習データの構造自体を変える発想が必要であると学びました。

### 2. OOF prediction は有効な情報源になる
`ridge_pred` の導入により、別モデルの予測を特徴量として再利用する有効性を体感しました。  
単一モデルの改善だけでなく、複数モデルの役割分担を考えることの重要性を学びました。

### 3. original dataset の活用は強力だった
本コンペでは、追加の original dataset を train fold にのみ加える augmentation が大きな改善源となりました。  
細かい FE よりも、「学習データそのものの情報量を増やす」ことの方が効果的な場面があると学びました。

### 4. CV 改善がそのまま Private LB 改善につながるとは限らない
pseudo-labeling など、一部の手法では手元 CV は改善しても Private LB では悪化しました。  
この経験から、過学習や評価設計の難しさを実践的に学ぶことができました。

### Future Work
今後は Playground Series に加えて、より実戦的な **メダル対象コンペ** に参加し、

- より多様なモデルの活用
- pseudo-labeling や stacking の高度化
- より堅牢な CV 設計
- feature engineering と data-centric improvement の使い分け

を経験しながら、銅メダル以上の獲得を目指したいと考えています。

---

## 6. Results (結果)
* **Best CV RMSE:** 8.72906  
* **Best Public LB:** 8.68461  
* **Best Private LB:** 8.71644  
* **Private leaderboard rank:** 4319人中1010位（上位約23%）

---

## 7. What this project demonstrates
このプロジェクトでは、以下の点を示すことを意識しました。

- 回帰タスクにおけるベースライン構築能力
- 特徴量エンジニアリングの仮説検証
- OOF prediction を用いたメタ特徴量設計
- original dataset を活用した data augmentation
- CV / Public / Private の挙動を踏まえた最終モデル選定
