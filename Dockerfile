# 1. Python 3.9の軽量版をベースにする
FROM python:3.9-slim

# 2. コンテナ内の作業ディレクトリを設定
WORKDIR /app

# 3. 依存関係ファイルをコピーしてインストール
# (requirements.txt に pandas, fastapi, uvicorn, joblib, scikit-learn を書いている前提)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 全ファイル（main.py, .joblibファイルなど）をコピー
COPY . .

# 5. Cloud Runのポート(8080)に合わせて起動
# main:app の 'main' は main.py のファイル名に対応
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]