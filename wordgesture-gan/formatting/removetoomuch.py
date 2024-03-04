import os
import pandas as pd
import random
import json
import sys
import alphabet_detect as ad

# テキストファイルを読み込む関数
def load_word_freq_dict(filepath):
    word_freq_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                freq, word = parts
                word_freq_dict[word] = int(freq)
    return word_freq_dict

def add_group_markers(csv_folder_path, output_folder_path):
    for filename in os.listdir(csv_folder_path):
        if filename.endswith('.csv'):
            input_filepath = os.path.join(csv_folder_path, filename)
            output_filepath = os.path.join(output_folder_path, filename)  # 出力ファイルパス
            df = pd.read_csv(input_filepath, usecols=range(12))
            print('Processing {}'.format(filename))
            # 'word'列と'sentence'列の両方が存在するかチェック
            if 'word' in df.columns and 'sentence' in df.columns:
                print('word and senctence is in')
                # 一時的なグループマーカーを追加
                df['temp_group'] = ((df['word'] != df['word'].shift()) | (df['sentence'] != df['sentence'].shift())).cumsum()
                # 各グループのサイズを計算
                df['group_size'] = df.groupby('temp_group')['temp_group'].transform('size')
                # サイズが1より大きい行のみを保持し、その後temp_groupを再計算
                df = df.loc[df['group_size'] > 1].copy()
                df['group'] = ((df['word'] != df['word'].shift()) | (df['sentence'] != df['sentence'].shift())).cumsum()
                # 不要な列を削除
                df.drop(columns=['temp_group', 'group_size'], inplace=True)
                # 結果を新しいフォルダに保存
                df.to_csv(output_filepath, index=False)
                print('Added group markers to {}'.format(filename))


def search_and_update_word_file_dict(output_folder, word_freq_dict, word_file_dict, word_file_dict_name):
    word_file_dict_path = os.path.join(os.path.dirname(__file__), word_file_dict_name)
    # txtファイルが存在すれば既存のword_file_dictを読み込む
    if os.path.exists(word_file_dict_path):
        with open(word_file_dict_path, 'r') as file:
            print('opening word_file_dict.txt')
            word_file_dict.update(json.load(file))
    #存在しない場合は新規作成，word_file_dictを更新
    else:
        for filename in os.listdir(output_folder):
            print('updating word_file_dict:{}'.format(filename))
            if filename.endswith('.csv'):
                filepath = os.path.join(output_folder, filename)
                df = pd.read_csv(filepath, usecols=range(13))
                if 'word' in df.columns:
                    for word in word_freq_dict.keys():
                        df['is_target_word'] = df['word'] == word
                        df['group'] = (df['is_target_word'] != df['is_target_word'].shift()).cumsum()
                        # 各グループのサイズを計算
                        df['group_size'] = df.groupby('group')['group'].transform('size')
                        # サイズが1のグループを削除
                        df = df[~((df['is_target_word']) & (df['group_size'] == 1))]
                        # 指定された単語を含むグループの数をカウント（行削除後）
                        target_groups_count = df[df['is_target_word']]['group'].nunique()
                        if target_groups_count > 0:
                            # word_file_dictに単語とファイル名を追加
                            if word not in word_file_dict:
                                word_file_dict[word] = []
                            # word_file_dict[word]にfilenameをtarget_groups_count回追加(同じファイルに複数回出現することを考慮)
                            word_file_dict[word].extend([filename] * target_groups_count)
                    # if word == 'you':
                    #     print('file:{}, target_groups_count:{} '.format(filename, target_groups_count))
                    #     print('all:{}'.format(len(word_file_dict[word])))
        # 処理完了後、結果をword_file_dict.txtに保存
        with open(word_file_dict_path, 'w') as file:
            json.dump(word_file_dict, file)

# 指定された単語を含む行グループを削除する関数
def delete_word_group_from_file(output_folder_path, filename, word):
    output_filepath = os.path.join(output_folder_path, filename)  # 出力ファイルパス
    df = pd.read_csv(output_filepath, usecols=range(13))
    if 'group' in df.columns and 'word' in df.columns:
        # 指定された単語を含むグループを特定
        groups_to_remove = df[df['word'] == word]['group'].unique()
        # groups_to_removeが空の場合はエラーメッセージを出力し、プログラムを終了
        if len(groups_to_remove) == 0:
            print(f"Error: No groups found to remove for word '{word}' in file '{filename}'.")
            sys.exit(1)  # プログラムをエラー終了させる
        else:
            # ランダムに1つのグループだけを削除対象として選択
            group_to_remove = random.choice(groups_to_remove)
            # 選択された1つのグループを削除
            df = df[~(df['group'] == group_to_remove)]
            #正規表現でアルファベット以外の単語を削除
            df_filtered = df[~df['word'].apply(ad.is_non_alphabetical)]

            # 変更を新しいフォルダに保存
            df_filtered.to_csv(output_filepath, index=False)
            print('deleted:', word)

# #ランダム→上から順に消す
# def delete_word_group_from_file(output_folder_path, filename, word):
#     output_filepath = os.path.join(output_folder_path, filename)  # 出力ファイルパス
#     df = pd.read_csv(output_filepath)
#     if 'group' in df.columns and 'word' in df.columns:
#         # 指定された単語を含むグループを特定
#         groups_to_remove = df[df['word'] == word]['group'].unique()
#         # groups_to_removeが空ではない場合、最初のグループを削除対象として選択
#         if len(groups_to_remove) > 0:
#             # 最初のグループを選択（上から順）
#             group_to_remove = groups_to_remove[0]
#             # 選択されたグループを削除
#             df = df[~(df['group'] == group_to_remove)]
#             # 変更をファイルに保存
#             df.to_csv(output_filepath, index=False)
#             print('Deleted group containing word "{}" from file "{}"'.format(word, filename))


# word_file_dictの内容を更新するための機能を実装
def update_word_file_dict_after_deletion(word_file_dict, word, file_to_modify):
    # 指定されたファイル名の出現回数を1減らす
    if file_to_modify in word_file_dict[word]:
        word_file_dict[word].remove(file_to_modify)


#処理の最後に，追加していたgroup列を全てのファイルから削除する
def delete_group_column_in_all_files(csv_folder_path):
    for filename in os.listdir(csv_folder_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(csv_folder_path, filename)
            df = pd.read_csv(filepath)
            if 'group' in df.columns:
                df.drop(columns=['group'], inplace=True)
                df.to_csv(filepath, index=False)
    print("group column deleted in all files")

def process_word_groups(word_file_dict, output_folder_path):
    debug_dict = {}
    # word_file_dictの各キー（単語）とそれに対応するファイルリストでループ
    for word, file_list in word_file_dict.items():
        # ファイルリストが5より大きく(＝同じファイルに2回以上出現することを考慮して重複ありのファイルリストの長さを指定)かつ
        # wordが1より大きい(word_file_dictには，「i」などの1文字英単語も含まれるので除外)場合
        if len(file_list) > 5 and len(word) > 1:
            # ファイルリストの長さから5を引いた数だけ削除を行う->最大で5つのグループのみ残す
            num_deletions = len(file_list) - 5
            if word == 'you' or word == 'this' or word == 'the' or word == 'is' or word == 'are' or word == 'to' or word == 'we' or word == 'am':
                debug_dict[word] = num_deletions
            for _ in range(num_deletions):
                # ファイルリストが空でない場合、ランダムにファイルを選択して削除処理を行う
                if file_list:
                    file_to_modify = random.choice(file_list)
                    filepath = os.path.join(output_folder_path, file_to_modify)
                    # 指定されたファイル内の特定のwordを含む行グループを削除
                    # print('{} wo delete simasu'.format(word))
                    delete_word_group_from_file(output_folder_path, filepath, word)
                    # 削除後にword_file_dictを更新
                    update_word_file_dict_after_deletion(word_file_dict, word, file_to_modify)
    print(debug_dict.items())



