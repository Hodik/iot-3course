import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import minmax_scale
from scipy import stats

df = pd.read_csv("bear_attacks.csv")
df.head()

df.info()
df = df["Age"]
df = minmax_scale(df, feature_range=(-1, 1))
plt.figure(figsize=[20, 10])
plt.plot(df[:600])
plt.show(block=True)
