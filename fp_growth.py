from fpgrowth_py import fpgrowth # Importation de la bibliothèque fpgrowth pour l'algorithme FP-Growth
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
import time

# Chargement des données depuis le fichier CSV avec l'encodage approprié
data = pd.read_csv('ecommerce-data/data.csv', encoding='ISO-8859-1')

# Calcul de la colonne 'GroupPrice'
data['GroupPrice'] = data['Quantity'] * data['UnitPrice']

# Affichage des dimensions du jeu de données
print('Dimensions du jeu de données : ', data.shape)

# Suppression des lignes contenant des valeurs manquantes
data = data.dropna()

# Affichage des nouvelles dimensions après la suppression des valeurs manquantes
print('Les nouvelles dimensions après la suppression des valeurs manquantes :', data.shape)
print('---------')
print(data.head())

# Liste des codes de stock uniques
liste = data['StockCode'].unique() 

# Création d'une liste de codes de stock à supprimer
stock_to_del = []
for el in liste:
    # Supprimer les produits correspondant à des cadeaux (codes de stock qui ne commencent pas par un chiffre)
    if el[0] not in ['1','2','3','4','5','6','7','8','9','10']:
        stock_to_del.append(el)

# Suppression des produits indésirables du jeu de données
data = data[data['StockCode'].map(lambda x: x not in stock_to_del)]

# Grouper les produits par facture et client
basket = data.groupby(['InvoiceNo', 'CustomerID']).agg({'StockCode': lambda s: list(set(s))})

# Affichage des dimensions du nouveau jeu de données regroupé
print('Dimension du nouveau jeu de données regroupé :', basket.shape)
print('----------')
print(basket.head())

# Mesure du temps d'exécution
a = time.time()

# Application de l'algorithme FP-Growth pour extraire les fréquent items sets et les règles d'association
freqItemSet, rules = fpgrowth(basket['StockCode'].values, minSupRatio=0.005, minConf=0.5)

# Mesure du temps d'exécution
b = time.time()

# Affichage du temps d'exécution en secondes
print("Temps d'exécution en secondes : ", b - a, 's.')

# Affichage du nombre de règles générées
print('Nombre de règles générées : ', len(rules))

# Création d'un DataFrame à partir des règles d'association
association = pd.DataFrame(rules, columns=['basket', 'next_product', 'proba'])

# Tri du DataFrame en fonction de la probabilité (confidence) de manière décroissante
association = association.sort_values(by='proba', ascending=False)

# Affichage des dimensions du tableau d'association
print("Les dimensions du tableau d'association : ", association.shape)

# Affichage des 10 premières lignes du tableau d'association
print(association.head(10))

def compute_next_best_product(basket_el):
    """
    paramètre : basket_el = liste des éléments du panier du consommateur
    retour : next_pdt, proba = prochain produit à recommander, probabilité d'achat. Ou (0,0) si aucun produit n'est trouvé.
    
    Description : à partir du panier d'un utilisateur, renvoie le produit à recommander s'il n'est pas trouvé 
    dans la liste des associations de la table associée au modèle FP-Growth. 
    Pour ce faire, nous recherchons dans la table des associations le produit à recommander à partir de chaque 
    produit individuel dans le panier du consommateur.
    """
    
    for k in basket_el:  # Pour chaque élément dans le panier du consommateur
        k = {k}
        if len(association[association['basket'] == k].values) != 0:  # Si nous trouvons une association correspondante dans la table FP Growth
            next_pdt = list(association[association['basket'] == k]['next_product'].values[0])[0]  # Nous prenons le produit conséquent
            if next_pdt not in basket_el:  # Nous vérifions si le client n'a pas déjà acheté le produit
                proba = association[association['basket'] == k]['proba'].values[0]  # Trouver la probabilité associée
                return next_pdt, proba
    
    return 0, 0  # Retourne (0,0) si aucun produit n'a été trouvé.


def find_next_product(basket):
    """
    Paramètre : basket = dataframe du panier du consommateur
    Retour : list_next_pdt, list_proba = liste des prochains éléments à recommander et les probabilités d'achat associées.
    
    Description : Fonction principale qui utilise la fonction ci-dessus. Pour chaque client dans le jeu de données, 
    nous recherchons une association correspondante dans la table du modèle FP Growth. Si aucune association n'est trouvée, 
    nous appelons la fonction compute_next_best_product qui recherche des associations de produits individuels.
    Si aucune association individuelle n'est trouvée, la fonction retourne (0,0).
    """
    n = basket.shape[0]
    list_next_pdt = []
    list_proba = []
    for i in range(n):  # Pour chaque client
        el = set(basket['StockCode'][i])  # Panier du client
        if len(association[association['basket'] == el].values) != 0:  # Si nous trouvons une association correspondante dans la table FP Growth
            next_pdt = list(association[association['basket'] == el]['next_product'].values[0])[0]  # Nous prenons le produit conséquent
            proba = association[association['basket'] == el]['proba'].values[0]  # Probabilité associée dans la table
            list_next_pdt.append(next_pdt)
            list_proba.append(proba)

        elif len(association[association['basket'] == el].values) == 0:  # Si aucun antécédent correspondant à tout le panier n'est trouvé dans la table
            next_pdt, proba = compute_next_best_product(basket['StockCode'][i])  # Appel à la fonction précédente
            list_next_pdt.append(next_pdt)
            list_proba.append(proba)

    return list_next_pdt, list_proba


a = time.time()

# Appel de la fonction find_next_product pour obtenir les produits recommandés et les probabilités associées
list_next_pdt, list_proba = find_next_product(basket)

b = time.time()
print("Temps d'exécution de la fonction", b - a)  # Affichage du temps d'exécution de la fonction

# Ajout des colonnes 'Recommended Product' et 'Probability' au DataFrame basket
basket['Recommended Product'] = list_next_pdt  # Ensemble des produits recommandés
basket['Probability'] = list_proba  # Ensemble des probabilités associées

# Affichage des premières lignes du DataFrame mis à jour
basket.head()


# Renommer la colonne 'StockCode' en 'Customer basket' dans le DataFrame basket
basket = basket.rename(columns={'StockCode': 'Customer basket'})

# Créer un DataFrame data_stock contenant les données uniques basées sur 'StockCode' dans le DataFrame data
data_stock = data.drop_duplicates(subset="StockCode", inplace=False)

# Initialiser des listes vides pour stocker les informations sur les prix et les descriptions
prices = []
description_list = []

# Parcourir chaque ligne du DataFrame basket
for i in range(basket.shape[0]):
    stockcode = basket['Recommended Product'][i]
    probability = basket['Probability'][i]
    
    # Si le code de stock n'est pas 0 (c'est-à-dire, s'il y a un produit recommandé)
    if stockcode != 0:
        # Obtenir le prix unitaire et la description du produit recommandé
        unitprice = data_stock[data_stock['StockCode'] == stockcode]['UnitPrice'].values[0]
        description = data_stock[data_stock['StockCode'] == stockcode]['Description'].values[0]
        
        # Estimer le prix en multipliant le prix unitaire par la probabilité
        estim_price = unitprice * probability
        
        # Ajouter le prix estimé et la description à leurs listes respectives
        prices.append(estim_price)
        description_list.append(description)
    else:
        # Si le code de stock est 0, ajouter 0 au prix et 'Null' à la description
        prices.append(0)
        description_list.append('Null')

# Ajouter les listes de prix et de descriptions au DataFrame basket
basket['Price estimation'] = prices
basket['Product description'] = description_list

# Réorganiser les colonnes du DataFrame basket
basket = basket.reindex(columns=['Customer basket', 'Recommended Product', 'Product description', 'Probability', 'Price estimation'])

# Afficher les premières lignes du DataFrame basket mis à jour
print(basket.head())


print('En moyenne, le système de recommandation peut prédire en ', basket['Probability'].mean() * 100, '% des cas le prochain produit que le client achètera.')


# Affichage des 10 premières règles d'association
print("Top 10 premières règles d'association:")
print(association.head(10))


# Boîte à moustaches des probabilités
plt.figure(figsize=(10, 6))
sns.boxplot(x=association['proba'], color='skyblue')
plt.title('Boîte à moustaches des probabilités')
plt.xlabel('Probability')
plt.show()


# Statistiques descriptives des probabilités
print('Statistiques descriptives des probabilités:')
print(association['proba'].describe())


# Calcul de la moyenne des probabilités par facture
average_probabilities = basket.groupby(['InvoiceNo', 'CustomerID'])['Probability'].mean()

# Histogramme des moyennes de probabilités par facture
plt.figure(figsize=(12, 6))
plt.hist(average_probabilities, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution des probabilités moyennes par facture')
plt.xlabel('Average Probability')
plt.ylabel('Count')
plt.show()


# Graphique en boîte des moyennes de probabilités par facture

sns.boxplot(x=average_probabilities, color='skyblue')
plt.title('Boîte des moyennes de probabilités par facture')
plt.xlabel('Average Probability')
plt.show()


# Nuage de points des prix estimés par probabilité
plt.scatter(basket['Probability'], basket['Price estimation'], color='skyblue')
plt.title('Nuage de points des prix estimés par probabilité')
plt.xlabel('Probability')
plt.ylabel('Price Estimation')
plt.show()


# Statistiques descriptives des prix estimés
print('Statistiques descriptives des prix estimés:')
print(basket['Price estimation'].describe())


# Histogramme des prix estimés
plt.figure(figsize=(10, 6))
plt.hist(basket['Price estimation'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogramme des prix estimés')
plt.xlabel('Price Estimation')
plt.ylabel('Count')
plt.show()


# Produits recommandés les plus fréquents
top_recommended_products = basket['Recommended Product'].value_counts().head(10)

# Graphique des produits recommandés les plus fréquents
plt.figure(figsize=(12, 8))
top_recommended_products.plot(kind='bar', color='skyblue')
plt.title('Top 10 produits recommandés les plus fréquents')
plt.xlabel('Product Code')
plt.ylabel('Count')
plt.show()


# Filtrer les règles en fonction du support et de la confiance
min_support_threshold = 0.05
min_confidence_threshold = 0.7


# Convertir les règles en DataFrame
rules_df = pd.DataFrame(rules, columns=['antecedent', 'consequent', 'confidence'])

# Filtrer les règles en fonction de la confiance
filtered_rules = rules_df[rules_df['confidence'] >= min_confidence_threshold]

# Visualisation des règles filtrées
print("Filtered Rules:")
print(filtered_rules)

# Visualisation de la confiance des règles d'association (après filtrage)
plt.figure(figsize=(12, 6))
sns.histplot(filtered_rules['confidence'], bins=20, kde=True, color='skyblue')
plt.title('Distribution de la confiance des règles d\'association (après filtrage)')
plt.xlabel('Confiance')
plt.ylabel('Count')
plt.show()