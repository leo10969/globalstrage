import os
import pandas as pd
import random
import json
import sys
import removetoomuch as rtm

# --------------------------------実行部分--------------------------------
text_file_path = 'formatting/words_freq.txt'
#元のCSVファイルが保存されているフォルダのパスを指定
csv_folder_path = '/Users/satoreo/python_projects/word-gesture_gan/datasets/swipecsvs_formatted'
# 結果を保存するフォルダのパスを指定
output_folder_path = '/Users/satoreo/python_projects/word-gesture_gan/datasets/debug_datasets'  

# word_file_dictの外部ファイル名（適宜変更）
word_file_dict_name = 'word_file_dict2.txt'
# 出現する単語リストを読み込む
word_freq_dict = rtm.load_word_freq_dict(text_file_path)

#2回目だけを実行する場合は以下の行をコメントアウト

# # 出力フォルダが存在しない場合は作成
# if not os.path.exists(output_folder_path):
#     os.makedirs(output_folder_path)
# else:
#     # 出力フォルダが存在する場合は中身を削除
#     for filename in os.listdir(output_folder_path):
#         file_path = os.path.join(output_folder_path, filename)
#         if os.path.isfile(file_path):
#             os.remove(file_path)


# # print("freq_dict_you: " + str(word_freq_dict["you"]))
# # print("freq_dict_nissan: " + str(word_freq_dict["nissan"]))

# #マーカーを追加したCSVファイルのコピーを新しいフォルダに保存
# rtm.add_group_markers(csv_folder_path, output_folder_path)

# #--------------------------------実行部分（1回目）--------------------------------
# word_file_dict = {}
# # word_file_dictを更新，すでにあれば読み込む，なければ新規作成
# rtm.search_and_update_word_file_dict(output_folder_path, word_freq_dict, word_file_dict, word_file_dict_name)


# rtm.process_word_groups(word_file_dict, output_folder_path)

# # すべてのファイルに含まれる指定された単語の出現回数を表示
# print("you:{}, the:{}, this:{}, is:{}, are:{}, to:{}, we:{}, am:{}".format(word_file_dict['you'], word_file_dict['the'], word_file_dict['this'], word_file_dict['is'], word_file_dict['are'], word_file_dict['to'], word_file_dict['we'], word_file_dict['am']))
# # プログラムの最後にCSVファイルからgroup列を削除する
# # delete_group_column_in_all_files(output_folder_path)

#--------------------------------実行部分（2回目）--------------------------------
word_file_dict2 = {}
print("--------------------------again--------------------------")
rtm.search_and_update_word_file_dict(output_folder_path, word_freq_dict, word_file_dict2, word_file_dict_name)

rtm.process_word_groups(word_file_dict2, output_folder_path)

rtm.delete_group_column_in_all_files(output_folder_path)