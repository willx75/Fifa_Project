import ipdb
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree as sk_tree
from sklearn import pipeline as sk_pipeline
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


list_categorical_features = ["preferred_foot", "work_rate", "team_position"]
list_features = base_features + base_averaging_feature

PredictedVariable = "value_eur"


def read_players(year: int):
    X = pd.read_csv(f"data/players_{year}.csv")
    X["year"] = year
    return X


class MiscellaneousTransform(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, list_features: List[str], list_features_to_average: List[str], list_categorical_features: List[str]):
        self.list_features = list_features
        self.list_features_to_average = list_features_to_average
        self.list_categorical_features = list_categorical_features

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        X = pre_processing.compute_remaining_contract_years(X)
        for feature in self.list_features_to_average:
            X = pre_processing.mean_per_category(X, feature)
        X = pre_processing.group_team_position(X)

        X = X[self.list_features]
        # Encode categorical features
        one_hot_encoded = pd.get_dummies(X[self.list_categorical_features])
        X = X.drop(self.list_categorical_features, axis=1)
        X = X.assign(**one_hot_encoded)
        return X.fillna(0)


def show_feature_importance(model: sk_tree.DecisionTreeClassifier, list_features):
    n_features = len(list_features)
    plt.barh(np.arange(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), list_features)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.show()


def train_on_dataset(year: int):

    X = read_players(year)
    Y = X[PredictedVariable]

    pipeline = sk_pipeline.Pipeline(
        [("pre_processing", MiscellaneousTransform(list_features, base_averaging_feature + list_averaged_features, list_categorical_features)),
         ("tree", sk_tree.DecisionTreeRegressor(random_state=0, max_depth=15))]
    )

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X, Y, random_state=12)
    pipeline.fit(X_train, Y_train)

    print("Training :", pipeline.score(X_train, Y_train))
    print("Validation :", pipeline.score(X_test, Y_test))
    show_feature_importance(pipeline.named_steps["tree"],
                            pipeline.named_steps["pre_processing"].transform(X).keys())
    return pipeline


pipeline = train_on_dataset(20)


def test_on_dataset(year: int):
    X = read_players(year)
    Y = X[PredictedVariable]
    print(f"Results on {year} :", pipeline.score(X, Y))


# def plot_relative_error(pipeline, year):
#    X = read_players(year)
#    X = X.sort_values(PredictedVariable)
#    predict = pipeline.predict(X)
#    Y = X[PredictedVariable]
#    plt.plot(Y, (Y - predict).abs() / predict)
#    plt.show()
#plot_relative_error(pipeline, 19)
