import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
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
basket = data.groupby(['InvoiceNo', 'CustomerID'])['StockCode'].apply(list).reset_index(name='StockCodeList')

# Affichage des dimensions du nouveau jeu de données regroupé
print('Dimension du nouveau jeu de données regroupé :', basket.shape)
print('----------')
print(basket.head())


# Convertir le format pour l'Apriori
te = TransactionEncoder()
basket_encoded = te.fit_transform(basket['StockCodeList'].apply(lambda x: [str(i) for i in x]))


# Convertir en DataFrame
basket_df = pd.DataFrame(basket_encoded, columns=te.columns_)


# Appliquer l'algorithme Apriori avec le support minimal et la confiance minimale spécifiés
min_support_threshold = 0.005
min_confidence_threshold = 0.5


# Mesure du temps d'exécution
a = time.time()

# Application de l'algorithme apriori pour extraire les fréquent items sets et les règles d'association
frequent_itemsets_apriori = apriori(basket_df, min_support=min_support_threshold, use_colnames=True)

# Mesure du temps d'exécution
b = time.time()

# Affichage du temps d'exécution en secondes
print("Temps d'exécution en secondes : ", b - a, 's.')


# Générer les règles d'association avec la confiance minimale
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence_threshold)


# Afficher les ensembles fréquents de l'Apriori
print('Les ensembles fréquents de trouvé par Apriori:')
print(frequent_itemsets_apriori)



# Appliquer l'algorithme Apriori avec des paramètres de support minimal = 0.01 et confiance minimal = 0.8
min_support_threshold = 0.01
min_confidence_threshold = 0.8


# Mesure du temps d'exécution
a = time.time()

frequent_itemsets_apriori = apriori(basket_df, min_support=min_support_threshold, use_colnames=True)


# Mesure du temps d'exécution
b = time.time()

# Affichage du temps d'exécution en secondes
print("Temps d'exécution en secondes : ", b - a, 's.')


# Générer les règles d'association avec la confiance minimale
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence_threshold)


# Afficher les règles d'association avec le support 0.01 et la confiance 0.8
rules_apriori_with_metrics = rules_apriori[['antecedents', 'consequents', 'support', 'confidence']]
print("Les ", len(rules_apriori_with_metrics), " premières règles d'association:")
print(rules_apriori_with_metrics.sort_values(by='confidence', ascending=False))