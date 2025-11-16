# import pandas as pd, requests
# # import os
# # print(os.getcwd())
# df = pd.read_parquet("../data/processed/test.parquet")
# payload = df.drop(columns=["Class"]).iloc[0].to_dict()
# r = requests.post("http://localhost:8000/predict", json=payload, timeout=30)
# print(r.status_code, r.json())


from pathlib import Path
import pandas as pd, requests, time

df = pd.read_parquet("../data/processed/test.parquet")
payload = df.drop(columns=["Class"]).iloc[0].to_dict()

for i in range(100):
    r = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
    if i % 10 == 0: print(i, r.status_code)
    time.sleep(0.1)  # 10 req/s; adjust if you like


