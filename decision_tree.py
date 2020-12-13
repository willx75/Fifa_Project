import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree as sk_tree
import sklearn


import pre_processing

base_features = ['age',
                 'overall',
                 'potential',
                 'wage_eur',
                 'preferred_foot',
                 'international_reputation',
                 'weak_foot',
                 'skill_moves',
                 'work_rate',
                 'team_position',
                 # 'loaned_from',
                 'contract_remaining_year',
                 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']

base_averaging_feature = ['gk']

list_averaged_features = ['goalkeeping', 'attacking',
                          'skill', 'movement', 'power', 'mentality', 'defending']


categorical_features = ["preferred_foot", "work_rate", "team_position"]

fifa_df = pd.read_csv("data/players_20.csv")
fifa_df = pre_processing.compute_remaining_contract_years(fifa_df, year=2020)
for feature in base_averaging_feature + list_averaged_features:
    fifa_df = pre_processing.mean_per_category(fifa_df, feature)
fifa_df = pre_processing.group_team_position(fifa_df)

list_features = base_features + base_averaging_feature
# list_features = base_features + base_averaging_feature + list_averaged_features

X = fifa_df[list_features].fillna(0)
# One hot enconding for categorical features
one_hot_encoded = pd.get_dummies(X[categorical_features])
X = X.drop(categorical_features, axis=1)
X = X.assign(**one_hot_encoded)
X = X.to_numpy()

Y = fifa_df["value_eur"]

# Discretize price
bins = np.array([0, 1, 5, 10, 50, 100, 200, 400, 600, 800, 100000]) * 10000
labels = ["F", 'E', "D", "C", "C+", "B", "B+", "A", "A+", "S"]
Y = pd.cut(Y, bins=bins, labels=labels, include_lowest=True).to_numpy()

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, random_state=12)
tree = sk_tree.DecisionTreeClassifier(random_state=0)
tree.fit(X_train, Y_train)
print("Training :", tree.score(X_train, Y_train))
print("Validation :", tree.score(X_test, Y_test))
