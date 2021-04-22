import pandas as pd

df = pd.read_csv('readability_res.csv')
i = 0
cor = 0
for index, row in df.iterrows():
    if row['class'] == 2:
        i += 1
        if row['readability'] == 3:
            cor += 1

print(cor / i)
