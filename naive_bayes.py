import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics


import pre_processing


# Choose features to use
list_features = ["age", "overall", "potential", "contract"]

fifa_df = pd.read_csv("data/players.csv")
fifa_df = pre_processing.compute_remaining_contract_years(fifa_df)
#fifa_df = pre_processing.remove_free_players(fifa_df)

X = fifa_df[list_features].to_numpy()
Y = fifa_df["value_eur"]
# Discretize price
bins = np.array([0, 1, 5, 10, 50, 100, 200, 400, 600, 800, 10000]) * 100000
labels = ["F", 'E', "D", "C", "C+", "B", "B+", "A", "A+", "S"]
Y = pd.cut(Y, bins=bins, labels=labels, include_lowest=True).to_numpy()


# Test de différents Naive Bayes, inspiré de la fameuse classification de vins
# Tester sur le dataset puis sur des joueurs de grande classe
def data_split(X, Y, test_size, random_state):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size, random_seed)
    return X_train, X_test, Y_train, Y_test


def naive_bayes_fit_and_predict(X_train, X_test, Y_train, Y_test):
    gnb, mnb, cnb, bnb, canb = GaussianNB(), MultinomialNB(), ComplementNB(), BernoulliNB(), CategoricalNB()
    Y_pred_gnb = gnb.fit(X_train, Y_train).predict(X_test)
    Y_pred_mnb = mnb.fit(X_train, Y_train).predict(X_test)
    Y_pred_cnb = cnb.fit(X_train, Y_train).predict(X_test)
    Y_pred_bnb = bnb.fit(X_train, Y_train).predict(X_test)
    Y_pred_canb = canb.fit(X_train, Y_train).predict(X_test)
    return Y_pred_gnb, Y_pred_mnb, Y_pred_cnb, Y_pred_bnb, Y_pred_canb


def naive_bayes_evaluate(Y_test, Y_pred):
    return metrics.accuracy_score(Y_test, Y_pred)


test_prop = 0.3
random_seed = 500
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_prop, random_state=random_seed)
# X_test = X[:20]
# Y_test = Y[:20]
# X_train, X_test, Y_train, Y_test = data_split(X,Y,test_size = test_prop,random_state = random_seed)
Y_pred_gnb, Y_pred_mnb, Y_pred_cnb, Y_pred_bnb, Y_pred_canb = naive_bayes_fit_and_predict(X_train, X_test, Y_train,
                                                                                          Y_test)
print(naive_bayes_evaluate(Y_test, Y_pred_gnb))
print(naive_bayes_evaluate(Y_test, Y_pred_mnb))
print(naive_bayes_evaluate(Y_test, Y_pred_cnb))
print(naive_bayes_evaluate(Y_test, Y_pred_bnb))
print(naive_bayes_evaluate(Y_test, Y_pred_canb))


def plot_value_by(fifa_df: pd.DataFrame, variable: str):

    x_values = fifa_df[variable].to_numpy()
    y_values = fifa_df["value_eur"].to_numpy()
    plt.xlabel(variable)
    plt.ylabel("value")
    plt.plot(x_values, y_values, "ro")
    plt.show()


# Ne pas exécuter les fonctions à la suite -> Overwrite
plot_value_by(fifa_df, "age")
#plot_value_by(fifa_df, "overall")
#plot_value_by(fifa_df, "potential")
#plot_value_by(fifa_df, "contract")
