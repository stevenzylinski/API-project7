# API-project7

## Description

Cette API est développée en Flask et permet de faire 2 choses :

- En lui envoyant des données sur un client et sa demande de prêt, l'API renvoie la probabilité de faire défaut du client
- L'API renvoie également un Explainer dans le même format que SHAP, pour pouvoir interpréter localement la prédiction

## Prérequis

- Python 3.10
- Flask
- Json
- Pickle
- Pandas
- Numpy
- Shap

## Installation

1. Clonez ce dépôt sur votre ordinateur
2. Ouvrez un terminal et naviguez jusqu'au dossier de l'API
3. Exécutez la commande `pip install -r requirements.txt` pour installer les dépendances
4. Exécutez la commande `flask run` pour lancer l'API en local

## Utilisation

Pour utiliser l'API :

- Accéder à cette adresse : https://msdocs-python-webapp-quickstart-szm.azurewebsites.net/api/makecalc/ pour obtenir la probabilité de défaut du/des client(s)
- Accéder à cette adresse : https://msdocs-python-webapp-quickstart-szm.azurewebsites.net/api/shap_imp/ pour obtenir les valeurs pour interpéter localement la prédiction du/des client(s)
