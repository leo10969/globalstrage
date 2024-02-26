## 参考
### How We Swipe: A Large-scale Shape-writing Dataset and Empirical Findings
https://dl.acm.org/doi/10.1145/3447526.3472059

### WordGesture-GAN: Modeling Word-Gesture Movement with Generative Adversarial Network
https://dl.acm.org/doi/10.1145/3544548.3581279

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

### 下処理後のデータセットの中身
- 約38000個のジェスチャデータ（1英単語最大5つのジェスチャデータ）
- 約11000個の異なる2文字以上で構成される英単語
- 全てcsvファイル
- ヘッダーは変更なし
- 約30000個の単語の訓練用ジェスチャデータと約7600個のテスト用ジェスチャデータ（8:2の比率）この時，対応する英単語のジェスチャは，どちらかにしか含まれないように分割

<!-- 画像用![image]
(https://myoctocat.com/assets/images/base-octocat.svg) -->

##　手順2（WordGesture-GAN）
memo
- Generatorへの入力：word-prototypeとVAEを通してエンコードされたガウシアン潜在コード
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

### Auto Variational Encoder

#### AVEの構成
- Linear 384x192, Leaky ReLU
   - Linear 192x96, Leaky ReLU
      - Linear 96x48, Leaky ReLU
         - Linear 48x32, Leaky ReLU
            - Mu
            - Log Variance



### Discriminator

#### DisCriminatorの構成
- Linear 384x192, Leaky ReLU
   - Linear 192x96, Leaky ReLU
      - Linear 96x48, Leaky ReLU
         - Linear 48x24, Leaky ReLU
            - Linear 24x1, Leaky ReLU