# Importer les bibliothèques nécessaires
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Charger les données
data_reviews = pd.read_csv('video_game_reviews.csv')

# Prévisualiser les premières lignes des données
print(data_reviews.head())

# Préparer les données pour l'arbre de décision
# Sélectionner des colonnes pertinentes pour le modèle
X = data_reviews[['Price', 'User Rating', 'Multiplayer']] 
y = data_reviews['Multiplayer']  # Cible

# Encodage des variables catégoriques dans X
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Encodage de la cible si nécessaire
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# Créer et entraîner l'arbre de décision
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X, y)

# Afficher l'arbre de décision
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree for Video Game Reviews")
plt.show()
