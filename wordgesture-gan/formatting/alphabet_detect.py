import os
import ast
import re
import pandas as pd
import shutil

# アルファベットのみで構成されているかをチェックする関数
def is_alphabetical(word):
    return re.match(r'^[a-zA-Z]+$', word) is not None

def is_non_alphabetical(word):
    return re.match(r'^[a-zA-Z]+$', word) is None

def read_only_non_alphabetical(file_path, dict, csv_folder):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()  # ファイルの内容を読み込む
        dict_data = ast.literal_eval(data)  # 文字列を辞書として評価

    # 条件に合致するキー（単語）を返す(アルファベット以外の単語を返す)
    for word, file_list in dict_data.items():
        if len(word) > 1 and not is_alphabetical(word):  # 文字列の長さが1より大きく、アルファベットのみで構成されている場合
            dict[word] = file_list

    all_csv_files = set(os.listdir(csv_folder))
    processed_files = set()

def main():
    file_path = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/formatting/word_file_dict2.txt'
    csv_folder = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/swipecsvs_normalized'
    new_csv_folder = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/swipecsvs_new'
    non_alphabetical_word_file_dict = {}
    read_only_non_alphabetical(file_path, non_alphabetical_word_file_dict, csv_folder)
    
    os.makedirs(new_csv_folder, exist_ok=True)
    
    all_csv_files = set(os.listdir(csv_folder))
    processed_files = set()

    for file_list in non_alphabetical_word_file_dict.values():
        for file_name in file_list:
            processed_files.add(file_name)
            file_path = os.path.join(csv_folder, file_name)
            new_file_path = os.path.join(new_csv_folder, file_name)
            
            df = pd.read_csv(file_path, encoding='utf-8')
            df_filtered = df[~df['word'].apply(is_non_alphabetical)]
            
            df_filtered.to_csv(new_file_path, index=False, encoding='utf-8')

    # `non_alphabetical_word_file_dict`に含まれないファイルを新しいフォルダにコピー
    for file_name in all_csv_files - processed_files:
        original_file_path = os.path.join(csv_folder, file_name)
        new_file_path = os.path.join(new_csv_folder, file_name)
        shutil.copy2(original_file_path, new_file_path)

if __name__ == "__main__":
    main()
