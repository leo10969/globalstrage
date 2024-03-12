import os
import glob
import pandas as pd

# 指定したフォルダパス
folder_path = "/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/prototype/prototype_csv"

# フォルダ内のCSVファイルのパスを取得
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# 各CSVファイルのファイル名が大文字を含んでいる場合、ファイル名を小文字に変換し、'word' 列の文字列も小文字に変換
for file_path in csv_files:
    # ファイル名の取得
    file_name = os.path.basename(file_path)
    
    if any(char.isupper() for char in file_name):
        # ファイル名を小文字に変換
        new_file_name = file_name.lower()
        
        # 新しいファイル名を作成
        new_file_path = os.path.join(folder_path, new_file_name)
        
        # ファイル名の変更
        os.rename(file_path, new_file_path)
        
        print(f"{file_name} を {new_file_name} に変更しました。")
        
        # CSVファイルの読み込み
        df = pd.read_csv(new_file_path)
        
        # 'word' 列の文字列を小文字化
        df['word'] = df['word'].str.lower()
        
        # 変更を保存
        df.to_csv(new_file_path, index=False)
        
        print(f"{file_name} の 'word' 列の文字列を小文字に変換しました。")
