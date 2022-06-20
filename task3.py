import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv(r"bear_attacks.csv")
print(df.head())

inplace = (True,)
for columns in df.columns:
    if len(df[columns].unique()) == 2:
        df[columns] = df[columns].map(
            {
                "class_1_of_variable_1": 1,
                "class_2_ of_variable_1": 0,
                "class_1_of_variable_2": 1,
                "class_2_of_variable_2": 0,
                1: 1,
                0: 0,
            }
        )

df = pd.concat(
    [
        df["Location"],
        pd.get_dummies(df["Bear"], prefix="Bear"),
        pd.get_dummies(df["Gender"], prefix="Gender"),
        pd.get_dummies(df["Age"], prefix="Age"),
    ],
    axis=1,
)
print(df.head())
df_y = df["Location"]
df.drop(["Location"], axis=1, inplace=True)
df_train, df_holdout, df_y_train, df_y_holdout = train_test_split(
    df.values, df_y, test_size=0.3
)
tree = DecisionTreeClassifier()
tree.fit(df_train, df_y_train)
tree_predict = tree.predict(df_holdout)
result = accuracy_score(df_y_holdout, tree_predict)
print("Accuracy: ", result)
df.describe(include="all")
tree_params = {"max_depth": range(1, 11), "max_features": range(4, 19)}
tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)
tree_grid.fit(df_train, df_y_train)
print("The best model setting on cross-validation: ", tree_grid.best_params_)
result_cv = accuracy_score(df_y_holdout, tree_grid.predict(df_holdout))
print("The best model setting on cross-validation:", result_cv)
print("The model without setting defining:", result)
