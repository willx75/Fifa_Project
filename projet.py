# encoding=utf-8

import re
import functools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics


# Field player
# 3 Long Name Key
# 2 Short Name #0
# 4 Age #1
# 6 Height #2
# 7 Weight #3
# 8 Nationality #4
# 9 Club #5
# 10 League #6
# 11 League rank #7
# 16 Positions (list) #8
# 17 Bon pied #9
# 12 Overall #10
# 13 Potential #11
# 18 International reputation #12
# 19 Weak foot #13
# 20 Skills #14
# 21 Work rate (2 infos) #15
# 25 Tags #16
# 45 Traits #17
# 33-38 Simplified Stats #18
# 46-74 Stats #19
# 29 Contract #20
# 15 Wage #21
# 14 Value #22

# Goalkeeper
# 3 Long Name Key
# 2 Short Name #0
# 4 Age #1
# 6 Height #2
# 7 Weight #3
# 8 Nationality #4
# 9 Club #5
# 10 League #6
# 11 League rank #7
# 16 Positions (list) #8
# 17 Bon pied #9
# 12 Overall #10
# 13 Potential #11
# 18 International reputation #12
# 19 Weak foot #13
# 20 Skills #14
# 21 Work rate (2 infos) #15
# 25 Tags #16
# 45 Traits #17
# 39-44 Simplified Stats #18
# 75-79 Stats #19
# 29 Contract remaining (faire l année moins 2000 + année du fichier) #20
# 15 Wage #21
# 14 Value #22

# Créer la liste des caractéristiques à stocker

def fusion_file(l_file, new_file):
    fusion = open(new_file, "w")
    fusion.write("First line \n")
    # Le parser élimine la première ligne, donc il faut la remplir dans la fusion
    for fichier in l_file:
        f = open(fichier, "r")
        next(f)
        for l in f:
            num = re.search(r'\d+', fichier).group()
            l = l.replace("\n", "")
            l = l + "," + num + "," + str(int(
                num) + 1999) + "\n"  # On ajoute num comme repère pour différencier Joueur_X de fifa 20 et Joueur_X de fifa 21 et le str final comme année de début du jeu (cela servira à calculer les durées de contrat)
            fusion.write(l)
        f.close()
    fusion.close()


def append_list_player(l_numbers_to_append, l):
    list_att = []
    list_play = []
    for elem in l_numbers_to_append:
        list_play.append(l[elem])
    if l[16] == "GK":
        for i in range(39, 45):
            list_att.append(l[i])
        list_play.append(list_att)
        list_att = []
        for i in range(75, 80):
            list_att.append(l[i])
        list_play.append(list_att)
        list_att = []
    else:
        for i in range(33, 39):
            list_att.append(l[i])
        list_play.append(list_att)
        list_att = []
        for i in range(46, 75):
            list_att.append(l[i])
        list_play.append(list_att)
        list_att = []
    if l[30] != '':
        year = int((l[-1].replace("\n", "")))
        list_play.append(int(l[30]) - year)  # Calcul temps de contrat restant
    list_play.append(l[15])
    list_play.append(l[14])
    return (list_play)


# Première tentative de parser, avec functools: des défauts
def parse_line(line):
    return (list(functools.reduce(lambda a, b: a + b, [[x] if i % 2 else x.split(",") for (i, x) in
                                                       filter(lambda x: x, enumerate(line.split("\"")))], [])))


# Seconde tentative de parser, avec des regex: ok !
def parse_line2(line):
    pattern = '(\"[^\"]+\"|([^,]+)|(?=(,,)))'
    temp = re.findall(pattern, line)
    res = ['' if s == ",," else s for s in temp]
    i = 0
    a = []
    while i < len(res):
        b = res[i]
        a.append(b[0])
        i += 1
    return a


# Création du dictionnaire de données
def parser_fifa(fichier):
    f = open(fichier, "r")
    next(f)
    c = 0
    dict_player = {}
    list_player = []
    l_numbers_to_append = [2, 4, 6, 7, 8, 9, 10, 11, 16, 17, 12, 13, 18, 19, 20, 21, 25, 45]
    for line in f:
        c += 1
        l = parse_line2(line)
        if l[30] != '':  # On élimine les joueurs sans contrat (moins de 0,3%)
            list_player = append_list_player(l_numbers_to_append, l)
            dict_player[l[3] + " " + l[-2]] = list_player
        list_player = []
    f.close()
    # print(c)
    # print(dict_player)
    return dict_player


# Mise en forme données pour training
def take_x_and_y(dict_player, list_to_add_X):
    X = []
    Y = []
    for elem in dict_player.keys():
        x = []
        l = dict_player[elem]
        for indice in list_to_add_X:
            x.append(int(l[indice]))
        y = int(l[22])
        X.append(x)
        Y.append(y)
    return (X, Y)


# On discrétise y
def classify_value(oldY):
    Y = []
    for oldy in oldY:
        if oldy <= 100000:
            y = "Classe F"
        elif oldy > 100000 and oldy <= 500000:
            y = "Classe E"
        elif oldy > 500000 and oldy <= 1000000:
            y = "Classe D"
        elif oldy > 1000000 and oldy <= 5000000:
            y = "Classe C"
        elif oldy > 5000000 and oldy <= 10000000:
            y = "Classe C+"
        elif oldy > 10000000 and oldy <= 20000000:
            y = "Classe B"
        elif oldy > 20000000 and oldy <= 40000000:
            y = "Classe B+"
        elif oldy > 40000000 and oldy <= 60000000:
            y = "Classe A"
        elif oldy > 60000000 and oldy <= 80000000:
            y = "Classe A+"
        elif oldy > 80000000:
            y = "Classe S"
        Y.append(y)
    return (Y)


# On choisit les caractéristiques à utiliser, exemple : age, overall, potentiel, contrat age
list_to_add_X = [1, 10, 11, 20]

# fusion_file(["players_15.csv","players_16.csv","players_17.csv","players_18.csv","players_19.csv","players_20.csv","players_21.csv"],"players.csv")

X, oldY = take_x_and_y(parser_fifa("players.csv"), list_to_add_X)
Y = classify_value(oldY)


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


# def mean_error_nb():

test_prop = 0.3
random_seed = 500
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_prop, random_state=random_seed)
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


def plot_value_by(list_to_add_X, file, name):
    X, Y = take_x_and_y(parser_fifa(file), list_to_add_X)
    newX = []
    for x in X:
        newX.append(x[0])
    plt.xlabel(name[9:-4])
    plt.ylabel(name[:5])
    plt.plot(newX, Y, "ro")
    plt.show()

# Ne pas exécuter les fonctions à la suite -> Overwrite
plot_value_by([1],"players.csv","value_by_age.png")
# plot_value_by([10],"players.csv","value_by_overall.png")
# plot_value_by([11],"players.csv","value_by_potential.png")
# plot_value_by([20],"players.csv","value_by_contract.png")
