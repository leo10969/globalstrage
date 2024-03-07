<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
 tex2jax: {
 inlineMath: [['$', '$'] ],
 displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
 }
 });
</script>



## 参考
### How We Swipe: A Large-scale Shape-writing Dataset and Empirical Findings
https://dl.acm.org/doi/10.1145/3447526.3472059

### WordGesture-GAN: Modeling Word-Gesture Movement with Generative Adversarial Network
https://dl.acm.org/doi/10.1145/3544548.3581279

## wordgesture-ganプロジェクトの構成・ファイルの説明
- datasets/：訓練，テストに使用するユーザが描いたジェスチャのデータ
   - datasets_per_word/
      - test_datasets/
         - {word}.csv: 訓練用ジェスチャデータに含まれる単語のジェスチャ（複数ジェスチャある場合は同じファイルに最大5つのジェスチャが含まれる）
      - train_datasets/
         - {word}.csv: 訓練用ジェスチャデータに含まれる単語のジェスチャ（複数ジェスチャある場合は同じファイルに最大5つのジェスチャが含まれる）
- formatting/：訓練，テストに使用するユーザが描いたジェスチャのデータの下処理を行うファイルを格納
   - alphabet_detect.py:
   - countcsvs.py:
   - formatcsv.py:logファイルからcsvファイルに変換
   - filtercsv.py:
   - normalizecsvs.py:
   - remove_empty_files.py:
   - removetoomuch.py:
   - reremove.py:
   - data_split.py:
   - data_split_each_word.py:
   - train_data_words.py:
   - make-wordslist.py:
- gan/
   - word_prototype.py
   - wordgesture-gan-model.py: VAE, Generator, Discriminatorの定義，訓練
   - wordgesture-gan-inference.py: 訓練モデルを用いて推論
- prototype/
   - prototype_csv/
   - prototype_imgs/
- Generator/
- Discriminator/
## 手順1（下処理）
How We Swipeの著者によって提供されているオープンソース（ログファイルなど）をhttps://osf.io/sj67f/
からダウンロードする．
WordGesture-GANの4.1節を参考に，まずはswipelogsのデータセットの下処理を行う．ここで自分はログファイルからCSVファイルに変換した．

### デフォルトとして，swipelogsのデータセットの中身
- 約124000個のジェスチャデータ
- 1338人のユーザによる11318個の異なる英単語
- logファイルとjsonファイル
- logファイルのヘッダーは，sentence, timestamp, keyb_width, keyb_height, event, x_pos, y_pos, 
x_radius, y_radius, angle, word, is_err

### 下処理
1. formatcsv.py: 'is_err'列が1であるジェスチャデータを除外（'is_err'が1の時は無効）
2. filtercsv.py: 'i'（主語のI）などの1文字で構成される英単語に対するジェスチャを除外~~←今ここ~~2/5
3. rm_ran_hash2.py→removetoomuch.py and reremove.py: 'the'や'you'などの他の単語に比べて頻繁に表現される単語を，出現上限を5として，それ以外をランダムに除外（単語に対するバイアスを避けるため）~~←今ここ2/6~~2/15(2回繰り返して処理することで無理やり解決)
4. remove_empty_files.py: 3によって中身がすべて削除されたCSVファイルを除外する
5. normalizecsv.py: データセットは様々な画面サイズ，キーボードサイズで収集された（キー比率は同じ）ため，キーボード全体のサイズを0-1として，ストローク座標を正規化
※x, y座標の比率を正規化前後で等しくするため，y座標の正規化を修正
6. data_split.py: 全ジェスチャを訓練データとテストデータに8:2の比率で分割，それぞれのCSVファイルに保存
   ※ある単語に最大5つのジェスチャデータが存在するので，データの編集が必要？~~←今ここ~~2/2←pythonのsklearnのGroupShuffleSplitで解決←2/16(改めて下処理完了)
   参考：https://upura.hatenablog.com/entry/2018/12/04/224436#GroupShuffleSplit

### 下処理するうえでの注意点
- How We SWipeが提供しているword_freq.txtはおそらく，その単語が少なくとも1つ含まれているファイルの数（例えば，"you"は1339個のファイルのうちの1039個
- removetoomuch.py：1回の処理ではどうしても特に出現回数が多い単語（this, the, youなど）を5個まで絞ることができない．reremove.pyで2回処理を行うとうまく全ての単語が5つ以下の出現数になるようにできる．よって，formatting/word_file_dict2.txtは1回目の処理がランダムな故，毎回（削除）更新する必要がある．
- 新たに，is_alphabetical関数を作成し，正規表現でないアルファベット以外の文字が含まれているかを調べて除く必要がある．(3/1に発覚)
→formatting/にあるword_file_dict2.txtは，正規表現以外も含まれているので注意．一応，formatting/removetoomuch.pyの
delete_word_group_from_file関数内に追記（ただし，試してない）．word_file_dict.txtを作成済みの場合は，alphabet_detect.pyを使用する

### 下処理後のデータセットの中身
- 約38000個のジェスチャデータ（1英単語最大5つのジェスチャデータ）
- 約11000個の異なる2文字以上で構成される英単語
- 全てcsvファイル
- ヘッダーは変更なし
- 約30000個の単語の訓練用ジェスチャデータと約7600個のテスト用ジェスチャデータ（8:2の比率）この時，対応する英単語のジェスチャは，どちらかにしか含まれないように分割

<!-- 画像用![image]
(https://myoctocat.com/assets/images/base-octocat.svg) -->

## 手順2（WordGesture-GANの実装）
memo
- Generatorへの入力：word-prototypeとVAEを通してエンコードされたガウシアン潜在コード

**※以下は基本WordGesture-GANの論文のコピペ**

***
### Auto Variational Encoder

#### AVEの構成
- Linear 384x192, Leaky ReLU
   - Linear 192x96, Leaky ReLU
      - Linear 96x48, Leaky ReLU
         - Linear 48x32, Leaky ReLU
            - Mu
            - Log Variance


***
### Discriminator

#### DisCriminatorの構成
- Linear 384x192, Leaky ReLU
   - Linear 192x96, Leaky ReLU
      - Linear 96x48, Leaky ReLU
         - Linear 48x24, Leaky ReLU
            - Linear 24x1, Leaky ReLU

#### Discriminatorの損失関数

![dics-loss](https://private-user-images.githubusercontent.com/69385853/309174914-6909ae03-599d-4de1-91d3-4870cf5c841d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDkyNzY5MjcsIm5iZiI6MTcwOTI3NjYyNywicGF0aCI6Ii82OTM4NTg1My8zMDkxNzQ5MTQtNjkwOWFlMDMtNTk5ZC00ZGUxLTkxZDMtNDg3MGNmNWM4NDFkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzAxVDA3MDM0N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWNlMzE0Y2ZhY2ZmOWI4NzRmZGJmMTBlMThlNGU3ZDJkZTNiMGE3OTBkY2RmYmM1ZDBjYjBlZjFlNDgwMzljNjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.g3R4CAQUfod4GK_gyb7P8pdwAQ0IOYNXxP3EHSddOtI)

- $ D(G(z, y))$：識別器 'D' からの出力（実数）であり，シミュレートされたジェスチャ $G(z, y)$ がユーザが描いたジェスチャにどれだけ近いか
- $ E_{z \sim P(z)} D(G(z, y))$ ：$P(z)$ の分布上での $D(G(z, y))$ の期待値
- $x$：ユーザが描いたジェスチャを表し，$P(x)$ の分布を持つ
- $D(x)$：xがユーザが描いたジェスチャにどれだけ近いか表すDiscriminatorの出力


***

### Generator
1. word_prototype.py: 入力のための単語プロトタイプの作成
2. wordgesture-gan.py: Generatorの定義

#### Generatorの構成
- BiLSTM 35x32
   - BiLSTM 32x32
      - BiLSTM 32x32
         - BiLSTM 32x32
            - Linear Layer 32x3
               - Tanh Activation

#### Generatorの損失関数
![gen-loss](https://private-user-images.githubusercontent.com/69385853/309184520-4cc491d1-0cd0-4755-9f48-081cc95dc17d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDkyNzkzNTAsIm5iZiI6MTcwOTI3OTA1MCwicGF0aCI6Ii82OTM4NTg1My8zMDkxODQ1MjAtNGNjNDkxZDEtMGNkMC00NzU1LTlmNDgtMDgxY2M5NWRjMTdkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzAxVDA3NDQxMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTVjMzI0MjJmYzcyYzYzZmQyOTEwYTI4MGRmYTk1NDFmYzE2Yjg5YjE4NDcwY2U4NGU1MzZhYmVhMDU2YWE1ZjImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.n9qxoITlb4W1x97JE-mz-w7dJs7iwei-6pJqBxepo-Q)

gen-lossの内容
- $L_{disc}(y)$ : Discriminatorの損失関数の出力
- $L_{feat}(y)$ : 与えられた単語yに対する特徴マッチング損失．識別器のすべての隠れ層におけるユーザが描いた（x）と生成されたジェスチャ（G(z, y)）の統計の差を測定（式3）
- $L_{rec}(y)$ : 単語yに対する再構成損失．ユーザが描いたジェスチャ(x)と生成器 (G(z, y))からのシミュレートされたジェスチャとの間の$L_1$損失（式4）
   - $L_1$ 距離： $p_i$と$q_i$はユーザが描いたジェスチャ $p$ の $i$ 番目の点とシミュレートされたジェスチャ $q$ のi番目の点であり，(x, y)は座標であり，$t$ はタイムスタンプ（式5）
- $L_{lat}(y)$ : 出力の多様性を強化し、モード崩壊を防ぐため．Bicycle GAN [50]を参照．ガウス分布P(z)からランダムにサンプリングされたエンコーディングzを取り，$\tilde{z} = E(G(z, y))$を使用して回復しようとします。元の潜在コード z と回復された潜在コード $\tilde{z}$ を $L_1$ 距離で比較します。この損失は方程式6として定義されます。
方程式5からの再構成損失は、潜在コードから生成されたシミュレートされたジェスチャ $\tilde{z}$ がユーザが描いたジェスチャ x と一致することを強制する一方で、潜在エンコーディング損失は、シミュレートされたジェスチャからのエンコーディング $\tilde{z}$ が初期エンコーディング z と一致することを保証します
- $L_{KLD}$ : 変分エンコーダの出力と正規分布との間のKullback-Leibler Divergence (KLD)．変分エンコーダの出力の分布が正規分布からあまりにも逸脱しないようにし，その結果，サンプルから遠ざかりすぎてしまうことがないようにする（式6）


![L_feat](https://private-user-images.githubusercontent.com/69385853/309204329-d03a398c-9289-4c39-8c1e-be690acdd268.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDkyODM5NTMsIm5iZiI6MTcwOTI4MzY1MywicGF0aCI6Ii82OTM4NTg1My8zMDkyMDQzMjktZDAzYTM5OGMtOTI4OS00YzM5LThjMWUtYmU2OTBhY2RkMjY4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzAxVDA5MDA1M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTU0ZDllYTEyZDZjNWM5YzBmODk1NjVkZjc0NTIxNmM4MzIyY2FhNTE5YjBiNGEwODE5NDI4ZDlkODgwOTIxMDEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.XD6m_9tLx6-62Ihmd2bOXZwLAx3rGCPLKA65OrWy6vY)

![L_rec](https://private-user-images.githubusercontent.com/69385853/309204570-7f09e8a5-58b4-4d5e-956e-c56948621dd8.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDkyODQwMDYsIm5iZiI6MTcwOTI4MzcwNiwicGF0aCI6Ii82OTM4NTg1My8zMDkyMDQ1NzAtN2YwOWU4YTUtNThiNC00ZDVlLTk1NmUtYzU2OTQ4NjIxZGQ4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzAxVDA5MDE0NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTZiNTYwYjVmYWQ1YWZiMDc3NmZjNjJjMDRhNjBkMTBhZGI1ZDA0OGY1ZmZjMDc3N2Q4N2YzYjc5ZjlmYjA3OGUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.l9hdcgky9ah4amJ-mEk3eei8AYfNmyiVKuIGSW8y_nw)

![p-q](https://private-user-images.githubusercontent.com/69385853/309204637-a9941c7d-ffed-4a41-8486-52e8ec19a5af.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDkyODQwMjAsIm5iZiI6MTcwOTI4MzcyMCwicGF0aCI6Ii82OTM4NTg1My8zMDkyMDQ2MzctYTk5NDFjN2QtZmZlZC00YTQxLTg0ODYtNTJlOGVjMTlhNWFmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzAxVDA5MDIwMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWYwOGRmMTQyMjMyZGU3N2RlYjQ4ZmNhMTU1NWU4NmUxMTIxMzRjMWNmMmQ5NGY1MTRiNzg0NTIzY2ZkMGMxM2EmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.2vjyNXL9YWpVDAnY89hhei7gDDHdp-1nUXAKlbCi22k)

![L_lat](https://private-user-images.githubusercontent.com/69385853/309205272-4be08e85-142c-4b61-af40-75adec0a7a32.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDkyODQxNjMsIm5iZiI6MTcwOTI4Mzg2MywicGF0aCI6Ii82OTM4NTg1My8zMDkyMDUyNzItNGJlMDhlODUtMTQyYy00YjYxLWFmNDAtNzVhZGVjMGE3YTMyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzAxVDA5MDQyM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWMwYmFhYjBiYzYxMmY4ZDg1NDY3Mzk2ZGE0M2FkY2Q4OTIxMWI0OGE3MTYzNjgyMzJlMGZiNGI2MTg0YWE4ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.GpeQCGuIcUXkN_QMoa_JIKuADT1xvYZikbhptxvFSrI)

## 手順3（Training）
基本的にwordgesture-gan-model.py

### 各種パラメータ設定
- ハイパパラメータ
   - $λ_{feat}$：1
   - $λ_{rec}$：5
   - $λ_{lat}$：0.5
   - $λ_{KLD}$：0.05
- バッチサイズ：512
- 学習率：0.0002(アダム最適化手法)
- Discとエンコーダにおいて，全層にLeaky ReLUを使用
- Genにおいて，全層にTanh
- Spectral NormalizetionをDiscの全層に適用

### 訓練の手順
- create_generator_input関数を用いてeneratorの入力の用意（word_prototypeとユーザジェスチャの結合）：
   - word_prototype
      - word_prototype.py:各単語のword_prototypeとして，128点の座標をcsvファイルに保存＋プロットの画像を生成・保存
      - word_prototypeは，128 x 2(x_pos, y_pos)なので，128 x 3になるようにz要素を追加
   - ユーザジェスチャ
      - 手順1で下処理したユーザジェスチャを変分エンコーダ（VAE）を通して 128 x 32の潜在コードzになるようにする
      - make-wordslist.py:(ジェスチャごとにcsvファイルを作成するか，1つのファイル(data_eight.csv)にするかはわからんけど)下処理後の訓練用データにある全単語をリストアップ
- epoch分train_step関数を実行
   