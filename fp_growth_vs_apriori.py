import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import time


# Charger les données depuis le fichier CSV avec l'encodage approprié
data = pd.read_csv('ecommerce-data/data.csv', encoding='ISO-8859-1')

# Suppression des lignes contenant des valeurs manquantes
data = data.dropna()

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



# Convertir le format pour l'Apriori
te = TransactionEncoder()
basket_encoded = te.fit_transform(basket['StockCodeList'].apply(lambda x: [str(i) for i in x]))

# Convertir en DataFrame
basket_df = pd.DataFrame(basket_encoded, columns=te.columns_)


# Paramètres pour Apriori et FP-Growth
min_support_threshold = 0.02
min_confidence_threshold = 0.3


# Apriori
start_time_apriori = time.time()
frequent_itemsets_apriori = apriori(basket_df, min_support=min_support_threshold, use_colnames=True)
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence_threshold)
end_time_apriori = time.time()


# FP-Growth (mlxtend)
start_time_fpgrowth_mlxtend = time.time()
frequent_itemsets_fpgrowth = fpgrowth(basket_df, min_support=min_support_threshold, use_colnames=True)
rules_fpgrowth_mlxtend = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=min_confidence_threshold)
end_time_fpgrowth_mlxtend = time.time()


# Afficher les résultats
print("Apriori:")
print("frequent_itemsets_apriori")
print(frequent_itemsets_apriori)
print("rules_apriori")
print(rules_apriori)

print("\n-------------------------------------------------------------------------------------")
print("\nFP-Growth (mlxtend):")
print("frequent_itemsets_fpgrowth")
print(frequent_itemsets_fpgrowth)
print("rules_fpgrowth_mlxtend")
print(rules_fpgrowth_mlxtend)


# Afficher les performances
print("\nTemps d'exécution Apriori:", end_time_apriori - start_time_apriori, "secondes")
print("Temps d'exécution FP-Growth:", end_time_fpgrowth_mlxtend - start_time_fpgrowth_mlxtend, "secondes")