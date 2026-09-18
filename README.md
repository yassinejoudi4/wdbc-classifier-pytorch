# WDBC Classifier (PyTorch)

Classification binaire (bénin/malin) sur le dataset Wisconsin Diagnostic Breast Cancer (WDBC), 
implémentée avec un MLP en PyTorch. Pipeline complet : téléchargement des données, 
préparation, entraînement avec recherche d'hyperparamètres, et évaluation.

## Dataset

- Source : [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)
- 569 échantillons, 30 features numériques (mesures de noyaux cellulaires)
- Classes déséquilibrées : 357 bénins / 212 malins

## Pipeline

1. **`preprocessing.py`** — télécharge les données brutes et la documentation depuis UCI, 
   parse les noms de colonnes, encode la target (M/B → 1/0), sauvegarde un CSV propre.
2. **`dataset.py`** — split stratifié train/val/test (70/15/15), standardisation des features 
   (fit sur train uniquement pour éviter le data leakage), construction des `DataLoader` PyTorch.
3. **`model.py`** — MLP (couches cachées configurables, ReLU, Dropout), sortie en logits bruts.
4. **`train.py`** — grid search sur l'architecture (`hidden_dims`) et le learning rate. 
   Sélection de la meilleure configuration sur le **F1-score** en validation (le recall seul 
   s'est révélé trompeur : certaines configurations atteignaient un recall de 1.0 en prédisant 
   presque tout comme malin, au prix d'une loss très élevée).
5. **`evaluate.py`** — évaluation finale sur le test set (jamais vu pendant la grid search) : 
   accuracy, precision, recall, F1, ROC-AUC, matrice de confusion.

## Résultats

Meilleure configuration : `hidden_dims=[16, 8]`, `lr=0.01`

| Métrique | Valeur |
|---|---|
| Accuracy | 0.9884 |
| Precision | 1.0000 |
| Recall | 0.9688 |
| F1-score | 0.9841 |
| ROC-AUC | 0.9959 |

Matrice de confusion (test set, 86 échantillons) :
[[54 0]
[ 1 31]]

**1 seul faux négatif** sur 32 cas malins du test set — un patient malin classé à tort comme 
bénin. Dans un contexte médical, c'est l'erreur la plus coûteuse (un faux négatif peut retarder 
un diagnostic), donc le recall a été priorisé sur l'accuracy seule tout au long du projet.

## Limites connues (honnêteté méthodologique)

- Le test set ne contient que 86 exemples : un seul cas raté représente ~3 points de recall. 
  Ces résultats varient d'une exécution à l'autre selon le split aléatoire (`random_state`), 
  comme le montrent les 3 runs de grid search effectués (résultats cohérents mais non identiques).
- Pas de validation croisée (K-fold) : les métriques sont issues d'un seul split fixe, pas 
  d'une moyenne sur plusieurs splits. Une évaluation plus rigoureuse inclurait du K-fold pour 
  quantifier la variance des résultats.
- Grid search simple (recherche exhaustive sur une petite grille), pas d'optimisation bayésienne 
  ni de recherche aléatoire plus poussée.

## Installation

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Utilisation

```bash
python src/preprocessing.py   # télécharge et nettoie les données
python src/train.py           # grid search + entraînement, sauvegarde le meilleur modèle
python src/evaluate.py        # évalue le modèle sauvegardé sur le test set
```

## Structure du projet

wdbc-classifier-pytorch/
├── data/
│ ├── raw/ # données brutes téléchargées (ignoré par git)
│ └── processed/ # CSV nettoyé (ignoré par git)
├── models/ # modèle entraîné (ignoré par git)
├── src/
│ ├── preprocessing.py
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ └── evaluate.py
├── requirements.txt
└── README.md
