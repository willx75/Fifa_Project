import ipdb
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree as sk_tree
from sklearn import ensemble as sk_ensemble
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


def train_model_on_dataset(year: int, show_importance=True, model="decision_tree", **model_kwargs):

    X = read_players(year)
    Y = X[PredictedVariable]

    model = sk_tree.DecisionTreeRegressor if model == "decision_tree" else sk_ensemble.RandomForestRegressor

    pipeline = sk_pipeline.Pipeline(
        [("pre_processing", MiscellaneousTransform(list_features, base_averaging_feature + list_averaged_features, list_categorical_features)),
         ("tree", model(random_state=0, **model_kwargs))]
    )

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X, Y, random_state=12)
    pipeline.fit(X_train, Y_train)

    train_score = pipeline.score(X_train, Y_train)
    test_score = pipeline.score(X_test, Y_test)
    if show_importance:
        show_feature_importance(pipeline.named_steps["tree"],
                                pipeline.named_steps["pre_processing"].transform(X).keys())
    return pipeline, train_score, test_score, X_test, Y_test


def plot_price_histogram(year):
    X = read_players(year)
    plt.hist(X[PredictedVariable], bins="sqrt")
    plt.title(f"Price distribution on 20{year} dataset")
    plt.show()


def test_on_dataset(pipeline, year: int):
    X = read_players(year)
    Y = X[PredictedVariable]
    print(f"Results on {year} :", pipeline.score(X, Y))


def plot_relative_error_on_test(pipeline, X_test, Y_test):

    predict = pipeline.predict(X_test)
    relative_error = (Y_test - predict).abs() / predict
    frame = pd.DataFrame({"true": Y_test.to_numpy(), "relative": relative_error})
    frame = frame.groupby("true").mean()
    plt.plot(frame.index, frame.relative * 100)
    plt.title("Relative error according to real price on test set")
    plt.xlabel("Price")
    plt.ylabel("Error")
    plt.show()


def plot_relative_error(pipeline, year, without_zero: bool = False):
    X = read_players(year)
    predict = pipeline.predict(X)
    Y = X[PredictedVariable]
    relative_error = (Y - predict).abs() / predict
    frame = pd.DataFrame({"true": Y.to_numpy(), "relative": relative_error})
    frame = frame.groupby("true").mean()
    if without_zero:
        frame = frame.drop(index=0)
    plt.plot(frame.index, frame.relative * 100)
    plt.title(f"Relative error according to real price on 20{year} dataset")
    plt.xlabel("Price")
    plt.ylabel("Error")
    plt.show()


def plot_max_depth_evolution(year, min_depth=1, max_depth=50, model="decision_tree"):
    list_train_score = list()
    list_test_score = list()
    list_depth = list(range(min_depth, max_depth + 1))
    for depth in list_depth:
        _, train_score, test_score, _, _ = train_model_on_dataset(
            year, model=model, max_depth=depth, show_importance=False)
        list_train_score.append(train_score)
        list_test_score.append(test_score)
    plt.plot(list_depth, list_train_score, label="train")
    plt.plot(list_depth, list_test_score, label="test")
    plt.title("Train and test scores with max depth variation")
    plt.xlabel("Max depth")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


# plot_price_histogram(20)
#
# To train decision tree on 2020 dataset
#
# pipeline, train_score, test_score, X_test, Y_test = train_model_on_dataset(
#    20, model="decision_tree", show_importance=True, max_depth=15)
#print("Training :", train_score)
#print("Validation :", test_score)
# plot_relative_error(pipeline, 19)  # Use this on a dataset different from training
#plot_relative_error_on_test(pipeline, X_test, Y_test)
# plot_max_depth_evolution(20)

# To train random forest on 2020 dataset

pipeline, train_score, test_score, X_test, Y_test = train_model_on_dataset(
    20, model="random_forest", show_importance=True, max_depth=15, n_estimators=100)
print("Training :", train_score)
print("Validation :", test_score)
plot_relative_error(pipeline, 19)
plot_relative_error_on_test(pipeline, X_test, Y_test)
