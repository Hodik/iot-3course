import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv("bear_attacks.csv")
df.head()

df.info()

df.describe(include="all")

print(df.columns)

f, axes = plt.subplots(2, 2, sharey=True, figsize=(30, 30))
axes = axes.flatten()

info_categorial = ["Bear", "Gender", "Age"]

for i, col in enumerate(info_categorial):
    print(df[col])
    sns.countplot(df[col], ax=axes[i])

plt.show()  # categorial attributes

import time

time.sleep(10)

plt.close()
numeric_infos = ["Latitude", "Longitude"]
f, ax = plt.subplots(1, 2, figsize=(35, 35))
for i, col in enumerate(numeric_infos):
    sns.distplot(df[col], ax=ax[i])

plt.show()  # numerical attributes

time.sleep(10)

plt.close()
