

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration de style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10



# %%
# lecture de la base de données
df = pd.read_csv("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv")
df = df.dropna()  # pour supprimer les lignes avec des valeurs manquantes

print("="*80)
print("STATISTIQUES DU DATASET")
print("="*80)
print(f"Nombre total d'échantillons : {len(df)}")
print(f"\nDistribution des espèces :")
print(df['species'].value_counts().to_string())
print(f"\nStatistiques descriptives :")
print(df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].describe())


# %%
# matrice des entrées (longueur du bec, largueur du bec, longueur des nageoires, poids du corps)
entrees = df[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]].to_numpy()
sp = ["Adelie","Gentoo","Chinstrap"]

# matrice des sorties (manchot Adélie, manchot Papou, manchot à jugulaire)
sorties = np.zeros([len(df), len(sp)])
for s in range(len(sp)):
    sorties[:, s] = (df.species == sp[s]).to_numpy()

# Visualisation des distributions
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Distribution des caractéristiques morphologiques par espèce", 
             fontsize=18, fontweight='bold', y=0.995)

features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
labels = ["Longueur du bec (mm)", "Profondeur du bec (mm)", 
          "Longueur des nageoires (mm)", "Masse corporelle (g)"]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, (feat, label) in enumerate(zip(features, labels)):
    ax = axes[idx // 2, idx % 2]
    for i, species in enumerate(sp):
        data = df[df['species'] == species][feat]
        ax.hist(data, bins=20, alpha=0.65, label=species, color=colors[i], 
                edgecolor='black', linewidth=1.2)
    ax.set_xlabel(label, fontsize=12, fontweight='bold')
    ax.set_ylabel("Fréquence", fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle='--')

plt.tight_layout()
plt.show()


# %%
# Standardisation
mu = entrees.mean(axis=0)
sigma = entrees.std(axis=0)
entrees_norm = (entrees - mu) / sigma

print("="*80)
print("NORMALISATION DES DONNÉES")
print("="*80)
print(f"Moyennes originales :\n{mu}")
print(f"\nÉcarts-types originaux :\n{sigma}")
print(f"\nAprès normalisation :")
print(f"  Nouvelles moyennes : {entrees_norm.mean(axis=0)}")
print(f"  Nouveaux écarts-types : {entrees_norm.std(axis=0)}")

# Application de la permutation fixe (reproductibilité)
perm = np.loadtxt("randperm0_332.txt", dtype=float).astype(int)
entrees_perm = entrees_norm[perm, :]
sorties_perm = sorties[perm, :]

# Séparation train/valid/test
# Train: 50%, Valid: 26%, Test: 24%
X_train = entrees_perm[:166, :]
Y_train = sorties_perm[:166, :]
X_valid = entrees_perm[166:254, :]
Y_valid = sorties_perm[166:254, :]
X_test = entrees_perm[254:, :]
Y_test = sorties_perm[254:, :]

print("\n" + "="*80)
print("RÉPARTITION DES DONNÉES")
print("="*80)
print(f"Ensemble d'entraînement : {X_train.shape[0]} échantillons ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Ensemble de validation  : {X_valid.shape[0]} échantillons ({X_valid.shape[0]/len(df)*100:.1f}%)")
print(f"Ensemble de test        : {X_test.shape[0]} échantillons ({X_test.shape[0]/len(df)*100:.1f}%)")

# Vérification de la répartition par classe
print("\nRépartition par espèce dans chaque ensemble :")
for i, species in enumerate(sp):
    n_train = Y_train[:, i].sum()
    n_valid = Y_valid[:, i].sum()
    n_test = Y_test[:, i].sum()
    print(f"  {species:10s} - Train: {n_train:.0f}, Valid: {n_valid:.0f}, Test: {n_test:.0f}")


# %%
def widrow_hoff(entrees, sorties, taux_apprentissage=0.008, epochs=20):
    """
    Algorithme de Widrow-Hoff pour l'entraînement d'un perceptron monocouche.
    Permet d'entrainer 3 perceptrons en parallele. 
    
    Paramètres:
    -----------
    entrees : array (n_samples, n_features)
        Matrice des données d'entrée normalisées
    sorties : array (n_samples, n_classes)
        Matrice des sorties
    taux_apprentissage : float
        Pas de la descente de gradient
    epochs : int
        Nombre d'itérations sur l'ensemble d'entraînement
        
    Retourne:
    ---------
    matrice_poids : array (n_features, n_classes)
        Poids optimisés pour chaque perceptron
    tab_erreur_moyenne : list
        Historique de l'erreur quadratique moyenne
    """
    nb_exemples, nb_attributs = entrees.shape
    nb_classes = sorties.shape[1]
    
    # Initialisation Xavier : variance = 1/√(nb_attributs)
    facteur_normalisation = 1 / np.sqrt(nb_attributs)
    
    
    matrice_poids = np.random.randn(nb_attributs, nb_classes) * facteur_normalisation
    
    tab_erreur_moyenne = []
    
    for epoch in range(epochs):
        # Mélange des exemples à chaque époque
        perm = np.random.permutation(nb_exemples)
        somme_erreur = 0.0
        
        for exemple in perm:
            x = entrees[exemple, :]
            y_true = sorties[exemple, :]
            
            # Sortie du perceptron (combinaison linéaire)
            y_pred = np.dot(x, matrice_poids)
            
            # Calcul de l'erreur
            erreur = y_true - y_pred
            
            # Mise à jour des poids par descente de gradient
            matrice_poids += taux_apprentissage * np.outer(x, erreur)
            
            somme_erreur += np.sum(erreur**2)
        
        # Erreur quadratique moyenne pour cette époque
        erreur_moyenne = somme_erreur / nb_exemples
        tab_erreur_moyenne.append(erreur_moyenne)
    
    return matrice_poids, tab_erreur_moyenne

# Entraînement
print("="*80)
print("ENTRAÎNEMENT DES PERCEPTRONS")
print("="*80)

W, err_train = widrow_hoff(X_train, Y_train, taux_apprentissage=0.008, epochs=20)

print(f"Entraînement terminé après 100 époques")
print(f"Erreur finale : {err_train[-1]:.6f}")
print(f"\nDimensions de la matrice des poids : {W.shape}")
print(f"  → {W.shape[0]} attributs × {W.shape[1]} classes")

# Visualisation de la convergence
plt.figure(figsize=(12, 5))
plt.plot(err_train, linewidth=2.5, color='#2E86AB', label='Erreur quadratique moyenne')
plt.xlabel("Époques", fontsize=13, fontweight='bold')
plt.ylabel("Erreur quadratique moyenne", fontsize=13, fontweight='bold')
plt.title("Convergence de l'algorithme Widrow-Hoff sur l'ensemble d'entraînement", 
          fontsize=15, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4, linestyle='--')
plt.tight_layout()
plt.show()


# %%
def compute_scores(X, W):
    """Calcule les scores (sorties linéaires) pour chaque classe."""
    return np.dot(X, W)

def ROC(y_true, scores):
    """
    Calcule la courbe ROC et l'AUC.
    
    Paramètres:
    -----------
    y_true : array (n_samples,)
        0 ou 1
    scores : array (n_samples,)
        Scores de prédiction (plus le score est élevé, plus la prédiction est positive)
        
    Retourne:
    ---------
    fp_r : array
        Taux de faux positifs
    tp_r : array
        Taux de vrais positifs
    auc : float
        Aire sous la courbe ROC
    """
    # Tri par scores décroissants
    sorted_indices = np.argsort(-scores)
    y_sorted = y_true[sorted_indices]
    
    # Nombre de positifs et négatifs
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    
    # Calcul cumulé des TP et FP
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    
    # Calcul des taux
    tp_r = np.concatenate([[0], tp / n_pos])
    fp_r = np.concatenate([[0], fp / n_neg])
    
    # Calcul de l'AUC par la méthode des trapèzes
    auc = np.trapezoid(tp_r, fp_r)
    
    return fp_r, tp_r, auc

def PR(y_true, scores):
    """
    Calcule la courbe Précision-Rappel et l'Average Precision.
    
    Paramètres:
    -----------
    y_true : array (n_samples,)
        0 ou 1
    scores : array (n_samples,)
        Scores de prédiction
        
    Retourne:
    ---------
     Rappel : array
        Valeurs de rappel
    precision : array
        Valeurs de précision
    ap : float
        Average Precision (aire sous la courbe PR)
    """
    sorted_indices = np.argsort(-scores)
    y_sorted = y_true[sorted_indices]
    
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return np.array([0, 1]), np.array([1, 0]), 0.0
    
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / n_pos
    
    # Calcul de l'Average Precision (méthode d'intégration correcte)
    # On ajoute les points (0, precision[0]) et on calcule l'aire
    recall_with_zero = np.concatenate([[0],  recall])
    precision_with_first = np.concatenate([[precision[0] if len(precision) > 0 else 1], precision])
    
    # L'AP est l'aire sous la courbe PR (intégration par trapèzes)
    # diff(recall) donne les largeurs des rectangles
    # on multiplie par les précisions correspondantes
    ap = np.sum(np.diff(recall_with_zero) * precision_with_first[:-1])
    
    return  recall, precision, ap

# Calcul des scores sur validation et test
scores_valid = compute_scores(X_valid, W)
scores_test = compute_scores(X_test, W)

# Calcul des métriques pour chaque classe
auc_valid = []
auc_test = []
ap_valid = []
ap_test = []

roc_curves_valid = []
roc_curves_test = []
pr_curves_valid = []
pr_curves_test = []

print("\n" + "="*80)
print("CALCUL DES AUC ET AVERAGE PRECISION PAR CLASSE")
print("="*80)

for i, species in enumerate(sp):
    # Courbes ROC et calcul de l'AUC
    fpr_v, tpr_v, auc_v = ROC(Y_valid[:, i], scores_valid[:, i])
    fpr_t, tpr_t, auc_t = ROC(Y_test[:, i], scores_test[:, i])
    
    auc_valid.append(auc_v)
    auc_test.append(auc_t)
    roc_curves_valid.append((fpr_v, tpr_v))
    roc_curves_test.append((fpr_t, tpr_t))
    
    # Courbes PR et calcul de l'AP
    recall_v, prec_v, ap_v = PR(Y_valid[:, i], scores_valid[:, i])
    recall_t, prec_t, ap_t = PR(Y_test[:, i], scores_test[:, i])
    
    ap_valid.append(ap_v)
    ap_test.append(ap_t)
    pr_curves_valid.append((recall_v, prec_v))
    pr_curves_test.append((recall_t, prec_t))
    
    print(f"\n{species:12s}:")
    print(f"  AUC (validation) : {auc_v:.4f}   |   AUC (test) : {auc_t:.4f}")
    print(f"  AP  (validation) : {ap_v:.4f}   |   AP  (test) : {ap_t:.4f}")

# Calcul des moyennes
mauc_valid = np.mean(auc_valid)
mauc_test = np.mean(auc_test)
map_valid = np.mean(ap_valid)
map_test = np.mean(ap_test)

print("\n" + "="*80)
print("MOYENNES SUR LES TROIS CLASSES (MAUC et MAP)")
print("="*80)
print(f"MAUC (validation) : {mauc_valid:.4f}   |   MAUC (test) : {mauc_test:.4f}")
print(f"MAP  (validation) : {map_valid:.4f}   |   MAP  (test) : {map_test:.4f}")


# %% [markdown]
# ### <a id="sec3-3"></a> 3.3 Tableau Récapitulatif des Métriques
# 
# Le tableau suivant résume les 6 valeurs d'AUC (3 classes × 2 ensembles) ainsi que les 6 valeurs d'AP.

# %%
results_df = pd.DataFrame({
    'Espèce': sp,
    'AUC (Valid)': [f"{auc:.4f}" for auc in auc_valid],
    'AUC (Test)': [f"{auc:.4f}" for auc in auc_test],
    'AP (Valid)': [f"{ap:.4f}" for ap in ap_valid],
    'AP (Test)': [f"{ap:.4f}" for ap in ap_test]
})

print("\n" + "="*80)
print("TABLEAU RÉCAPITULATIF DES PERFORMANCES")
print("="*80)
print(results_df.to_string(index=False))
print("\n" + "-"*80)
print(f"{'MOYENNES':12s}  {mauc_valid:7.4f}      {mauc_test:7.4f}     {map_valid:7.4f}     {map_test:7.4f}")
print("="*80)


# %%
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("Courbes ROC par espèce", fontsize=18, fontweight='bold', y=0.95)

for i, species in enumerate(sp):
    # Courbes ROC - Validation
    fp_r, tp_r = roc_curves_valid[i]
    ax = axes[i, 0]
    ax.plot(fp_r, tp_r, linewidth=3, label=f'{species} (AUC={auc_valid[i]:.3f})',
            color=colors[i], marker='o', markersize=4, markevery=max(1, len(fp_r)//10))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, label='Aléatoire (AUC=0.5)')
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12, fontweight='bold')
    ax.set_title(f'{species} - Validation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Courbes ROC - Test
    fp_r, tp_r = roc_curves_test[i]
    ax = axes[i, 1]
    ax.plot(fp_r, tp_r, linewidth=3, label=f'{species} (AUC={auc_test[i]:.3f})',
            color=colors[i], marker='o', markersize=4, markevery=max(1, len(fp_r)//10))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, label='Aléatoire (AUC=0.5)')
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12, fontweight='bold')
    ax.set_title(f'{species} - Test', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("Courbes Précision-Rappel par espèce", fontsize=18, fontweight='bold', y=0.95)

for i, species in enumerate(sp):
    # Validation
    recall, precision = pr_curves_valid[i]
    ax = axes[i, 0]
    ax.plot(recall, precision, linewidth=3, label=f'{species} (AP={ap_valid[i]:.3f})',
            color=colors[i], marker='s', markersize=4, markevery=max(1, len(recall)//10))
    ax.set_xlabel('Rappel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(f'{species} - Validation', fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    # Test
    recall, precision = pr_curves_test[i]
    ax = axes[i, 1]
    ax.plot(recall, precision, linewidth=3, label=f'{species} (AP={ap_test[i]:.3f})',
            color=colors[i], marker='s', markersize=4, markevery=max(1, len(recall)//10))
    ax.set_xlabel('Rappel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(f'{species} - Test', fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %%
print("="*80)
print("ANALYSE COMPARATIVE : VALIDATION vs TEST")
print("="*80)

print("\n1. COMPARAISON PAR CLASSE")
print("-" * 80)

for i, species in enumerate(sp):
    diff_auc = auc_valid[i] - auc_test[i]
    diff_ap = ap_valid[i] - ap_test[i]
    print(f"\n{species}:")
    print(f"  Δ AUC (valid - test) : {diff_auc:+.4f}") 
    print(f"  Δ AP  (valid - test) : {diff_ap:+.4f}")

print("\n" + "-" * 80)
print("\n2. MOYENNES GLOBALES")
print("-" * 80)

diff_mauc = mauc_valid - mauc_test
diff_map = map_valid - map_test

print(f"\nΔ MAUC (valid - test) : {diff_mauc:+.4f}")
print(f"Δ MAP  (valid - test) : {diff_map:+.4f}")
    

# %%
print("\n" + "="*80)
print("ANALYSE COMPARATIVE : MAP vs MAUC")
print("="*80)

print("\n1. COMPARAISON DES MÉTRIQUES")
print("-" * 80)

ecart_valid = map_valid - mauc_valid
ecart_test = map_test - mauc_test

print(f"\nSur l'ensemble de VALIDATION :")
print(f"  MAP  = {map_valid:.4f}")
print(f"  MAUC = {mauc_valid:.4f}")
print(f"  Écart (MAP - MAUC) = {ecart_valid:+.4f}")

print(f"\nSur l'ensemble de TEST :")
print(f"  MAP  = {map_test:.4f}")
print(f"  MAUC = {mauc_test:.4f}")
print(f"  Écart (MAP - MAUC) = {ecart_test:+.4f}")

# Visualisation comparative
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

x = np.arange(2)
width = 0.35

bars1 = ax.bar(x - width/2, [mauc_valid, mauc_test], width, label='MAUC', 
               color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, [map_valid, map_test], width, label='MAP', 
               color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Ensemble de données', fontsize=13, fontweight='bold')
ax.set_ylabel('Score de performance', fontsize=13, fontweight='bold')
ax.set_title('Comparaison MAP vs MAUC sur Validation et Test', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Validation', 'Test'], fontsize=12)
ax.legend(fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim([0, 1.05])

# Ajout des valeurs sur les barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()



# %% [markdown]
# ## <a id="sec7"></a>7. Ajout de bruit dans les données et impact sur les performances

# %%
import copy
import numpy as np
import matplotlib.pyplot as plt

# --- Fonctions de bruit ---
def add_label_noise(Y, taux_bruit=0.1, seed=None):
    """
    Ajoute du bruit sur les labels (changement aléatoire pour un pourcentage donné d'échantillons).
    """
    if seed is not None:
        np.random.seed(seed)
    Y_noisy = copy.deepcopy(Y)
    n_samples, n_classes = Y.shape
    n_bruit = int(n_samples * taux_bruit)
    indices = np.random.choice(n_samples, n_bruit, replace=False)
    for idx in indices:
        true_class = np.argmax(Y_noisy[idx])
        new_class = np.random.choice([c for c in range(n_classes) if c != true_class])
        Y_noisy[idx] = np.zeros(n_classes)
        Y_noisy[idx, new_class] = 1
    return Y_noisy

def add_feature_noise(X, amplitude=0.2, seed=None, per_feature=True):
    """
    Ajoute un bruit multiplicatif uniforme aux attributs.
    Chaque élément est modifié ainsi : X_noisy = X * (1 + r) avec r ~ Uniform(-amplitude, +amplitude).

    Paramètres :
      - X : array (n_samples, n_features)
      - amplitude : float, amplitude relative maximale (ex: 0.2 pour ±20%)
      - seed : int ou None pour reproductibilité
      - per_feature : bool, si True on tire un bruit par attribut (même bruit pour tous les échantillons)

    Retour :
      - X_noisy : array de même forme que X
    """
    if seed is not None:
        np.random.seed(seed)

    if per_feature:
        # bruit par attribut, identique pour tous les exemples
        noise = np.random.uniform(-amplitude, amplitude, size=(1, X.shape[1]))
        noise = np.repeat(noise, X.shape[0], axis=0)
    else:
        # bruit par élément (échantillon × attribut)
        noise = np.random.uniform(-amplitude, amplitude, size=X.shape)

    return X * (1.0 + noise)

# %% [markdown]
# ## <a id="sec7-1"></a>7.1 Test : bruit uniquement sur Test set

# %%
# --- Ajout de bruit sur le Test ---
Y_test_noisy = add_label_noise(Y_test, taux_bruit=0, seed=42)
X_test_noisy = add_feature_noise(X_test, amplitude=0.8, seed=42, per_feature=True)

# --- Calcul des scores avec le modèle entraîné sur les données propres ---
scores_test_noisy = compute_scores(X_test_noisy, W)

# --- Calcul des métriques ---
auc_test_noisy = []
ap_test_noisy = []
roc_curves_test_noisy = []
pr_curves_test_noisy = []

for i, species in enumerate(sp):
    fpr, tpr, auc_val = ROC(Y_test_noisy[:, i], scores_test_noisy[:, i])
    recall, precision, ap_val = PR(Y_test_noisy[:, i], scores_test_noisy[:, i])
    auc_test_noisy.append(auc_val)
    ap_test_noisy.append(ap_val)
    roc_curves_test_noisy.append((fpr, tpr))
    pr_curves_test_noisy.append((recall, precision))
    
print("\n" + "="*80)
print("ENTRAÎNEMENT ET TEST AVEC BRUIT")
print("="*80)
for i, species in enumerate(sp):
    print(f"{species:12s} : AUC = {auc_test_noisy[i]:.3f} | AP = {ap_test_noisy[i]:.3f}")
print(f"\nMAUC(test bruité train+test) : {np.mean(auc_test_noisy):.3f}")
print(f"MAP(test bruité train+test)  : {np.mean(ap_test_noisy):.3f}")

# --- Visualisation ROC par classe ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, species in enumerate(sp):
    fpr, tpr = roc_curves_test_noisy[i]
    axes[i].plot(fpr, tpr, linewidth=2.5, label=f'AUC={auc_test_noisy[i]:.3f}', color=colors[i])
    axes[i].plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC=0.5)')
    axes[i].set_xlabel('FPR', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('TPR', fontsize=12, fontweight='bold')
    axes[i].set_title(f'ROC - {species}', fontsize=14, fontweight='bold')
    axes[i].legend(loc='lower right')
    axes[i].grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# --- Visualisation PR par classe ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, species in enumerate(sp):
    recall, precision = pr_curves_test_noisy[i]
    axes[i].plot(recall, precision, linewidth=2.5, label=f'AP={ap_test_noisy[i]:.3f}', color=colors[i])
    axes[i].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[i].set_title(f'PR - {species}', fontsize=14, fontweight='bold')
    axes[i].legend(loc='lower left')
    axes[i].grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## <a id="sec7-2"></a>7.2 Train + Test bruités

# %%
# Bruit d'amplitude à 1.2

Y_train_noisy = add_feature_noise(Y_test, amplitude=1.2, seed=123, per_feature=True)
X_train_noisy = add_feature_noise(X_test, amplitude=1.2, seed=123, per_feature=True)

# Ré-entraînement sur train bruité
W_noisy, err_train_noisy = widrow_hoff(X_train_noisy, Y_train_noisy, taux_apprentissage=0.008, epochs=20)

# Scores sur test bruité après train bruité
scores_test_noisy2 = compute_scores(X_test_noisy, W_noisy)

auc_test_noisy2 = []
ap_test_noisy2 = []
roc_curves_test_noisy2 = []
pr_curves_test_noisy2 = []

for i, species in enumerate(sp):
    fp_r, tp_r, auc_t = ROC(Y_test_noisy[:, i], scores_test_noisy2[:, i])
    recall, precision, ap_t = PR(Y_test_noisy[:, i], scores_test_noisy2[:, i])
    auc_test_noisy2.append(auc_t)
    ap_test_noisy2.append(ap_t)
    roc_curves_test_noisy2.append((fp_r, tp_r))
    pr_curves_test_noisy2.append((recall, precision))

print("\n" + "="*80)
print("ENTRAÎNEMENT ET TEST AVEC BRUIT")
print("="*80)
for i, species in enumerate(sp):
    print(f"{species:12s} : AUC = {auc_test_noisy2[i]:.3f} | AP = {ap_test_noisy2[i]:.3f}")
print(f"\nMAUC(test bruité train+test) : {np.mean(auc_test_noisy2):.3f}")
print(f"MAP(test bruité train+test)  : {np.mean(ap_test_noisy2):.3f}")

# --- Visualisation ROC par classe après train bruité + test bruité ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, species in enumerate(sp):
    fp_r, tp_r = roc_curves_test_noisy2[i]
    axes[i].plot(fp_r, tp_r, linewidth=2.5, label=f'AUC={auc_test_noisy2[i]:.3f}', color=colors[i])
    axes[i].plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC=0.5)')
    axes[i].set_xlabel('FPR', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('TPR', fontsize=12, fontweight='bold')
    axes[i].set_title(f'ROC - {species}', fontsize=14, fontweight='bold')
    axes[i].legend(loc='lower right')
    axes[i].grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# --- Visualisation PR par classe après train bruité + test bruité ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, species in enumerate(sp):
    recall, precision = pr_curves_test_noisy2[i]
    axes[i].plot(recall, precision, linewidth=2.5, label=f'AP={ap_test_noisy2[i]:.3f}', color=colors[i])
    axes[i].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[i].set_title(f'PR - {species}', fontsize=14, fontweight='bold')
    axes[i].legend(loc='lower left')
    axes[i].grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Comparaison des Courbes ROC : Modèle Normal vs Modèle Bruité", 
             fontsize=16, fontweight='bold', y=1.02)

color_normal = '#2E86AB'  
color_noisy = '#D9534F'   

for i, species in enumerate(sp):
    ax = axes[i]
    
    # Courbe ROC "Normale"
    fpr_clean, tpr_clean = roc_curves_test[i]
    auc_clean = auc_test[i]
    ax.plot(fpr_clean, tpr_clean, color=color_normal, linewidth=2.5, 
            label=f'Normal (AUC={auc_clean:.3f})')
    
    # Courbe ROC "Bruitée" (Train & Test bruités)
    fpr_noisy, tpr_noisy = roc_curves_test_noisy2[i]
    auc_noisy = auc_test_noisy2[i]
    ax.plot(fpr_noisy, tpr_noisy, color=color_noisy, linewidth=2.5, linestyle='--', 
            label=f'Bruité (AUC={auc_noisy:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, alpha=0.5, label='Aléatoire')
    ax.set_title(f"Espèce : {species}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def mAUC(X, Y, W):
    scores = np.dot(X, W)
    aucs = []
    for i in range(Y.shape[1]):
        _, _, auc = ROC(Y[:, i], scores[:, i])
        aucs.append(auc)
    return np.mean(aucs)

def add_feature_noise(X, amplitude=0.2, seed=None, per_feature=True):
    if seed is not None:
        np.random.seed(seed)
    if per_feature:
        noise = np.random.uniform(-amplitude, amplitude, size=(1, X.shape[1]))
        X_noisy = X * (1.0 + noise)
    else:
        noise = np.random.uniform(-amplitude, amplitude, size=X.shape)
        X_noisy = X * (1.0 + noise)
    return X_noisy

def k_fold_test_vs_valid(X_train, Y_train, X_test, Y_test, k, lr=0.008, epochs=20):
    N = len(X_train)
    fold_size = N // k
    n_features = X_train.shape[1]
    n_classes = Y_train.shape[1]
    
    sum_valid_auc = np.zeros(epochs)
    sum_test_auc = np.zeros(epochs)

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else N
        
        val_idx = np.arange(start, end)
        train_idx = np.concatenate([np.arange(0, start), np.arange(end, N)])
        
        Xt, Xv = X_train[train_idx], X_train[val_idx]
        Yt, Yv = Y_train[train_idx], Y_train[val_idx]
        
        W = np.random.randn(n_features, n_classes) * (1 / np.sqrt(n_features))
        
        for e in range(epochs):
            sum_valid_auc[e] += mAUC(Xv, Yv, W)
            sum_test_auc[e] += mAUC(X_test, Y_test, W)
            
            perm = np.random.permutation(len(Xt))
            for idx in perm:
                x_s = Xt[idx]
                y_s = Yt[idx]
                pred = np.dot(x_s, W)
                err = y_s - pred
                W += lr * np.outer(x_s, err)
    
    return sum_valid_auc / k, sum_test_auc / k

X_noisy = add_feature_noise(entrees_perm, amplitude=1.2, seed=999, per_feature=True)
Y_noisy = sorties_perm

EPOCHS_B = 150
mean_valid_B, mean_test_B = k_fold_test_vs_valid(
    X_noisy, Y_noisy, X_test_noisy, Y_test_noisy, k=5, epochs=EPOCHS_B
)

k_values = range(2, 30)
deltas_C_abs = []
for k in k_values:
    m_valid, m_test = k_fold_test_vs_valid(X_noisy, Y_noisy, X_test_noisy, Y_test_noisy, k=k, epochs=20)
    best_epoch = np.argmax(m_valid)
    deltas_C_abs.append(abs(m_valid[best_epoch] - m_test[best_epoch]))


plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
t = np.arange(EPOCHS_B)
plt.plot(t, mean_test_B, label='Test', color='#2E86AB', linewidth=2)
plt.plot(t, mean_valid_B, label='Validation', color='#D9534F', linewidth=2)

opt_idx = np.argmax(mean_valid_B)
plt.axvline(opt_idx, color='green', linestyle='--', alpha=0.6, label='Optimal')
y_min_zoom = min(np.min(mean_test_B[10:]), np.min(mean_valid_B[10:])) - 0.05
plt.ylim(max(0.4, y_min_zoom), 1.01)
plt.title("B : Test vs Validation (Zoom)", fontsize=12, fontweight='bold')
plt.xlabel("Époques")
plt.ylabel("mAUC")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(k_values, deltas_C_abs, marker='o', linestyle='-', color='purple', linewidth=2)
plt.fill_between(k_values, 0, deltas_C_abs, color='purple', alpha=0.1)
plt.ylim(bottom=0)
plt.title("C : |Delta(mAUC)| vs k", fontsize=12, fontweight='bold')
plt.xlabel("k (Folds)")
plt.ylabel("|Delta(mAUC)|")
plt.xticks(k_values)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
new_df = df.drop(columns=['island', 'sex', 'year'])


variance_df = new_df.groupby('species').std()
# print(variance_df)


moyenne_df = new_df.groupby('species').mean()
# print(moyenne_df)


from scipy.stats import norm

df_features = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
variance_per_species = df.groupby('species')[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].std()
mean_per_species = df.groupby('species')[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].mean()

N_AUGMENTATION = 2
GAIN_MIN = 0.1
GAIN_MAX = 1.0

def augment_data(X_orig, Y_orig, species_labels, n_copies=2):
    X_augmented = [X_orig]
    Y_augmented = [Y_orig]
    
    for copy_idx in range(n_copies):
        X_noisy = np.zeros_like(X_orig)
        
        for sample_idx in range(len(X_orig)):
            species_idx = species_labels[sample_idx]
            species_name = sp[species_idx]
            
            for feat_idx, feat_name in enumerate(["bill_length_mm", "bill_depth_mm", 
                                                    "flipper_length_mm", "body_mass_g"]):
                sigma = variance_per_species.loc[species_name, feat_name]
                gain = np.random.uniform(GAIN_MIN, GAIN_MAX)
                sigma_scaled = sigma * gain
                
                noise = norm.rvs(loc=0, scale=sigma_scaled)
                X_noisy[sample_idx, feat_idx] = X_orig[sample_idx, feat_idx] + noise
        
        X_augmented.append(X_noisy)
        Y_augmented.append(Y_orig.copy())
    
    return np.vstack(X_augmented), np.vstack(Y_augmented)

species_indices_perm = np.argmax(sorties_perm, axis=1)
species_train = species_indices_perm[:166]
species_valid = species_indices_perm[166:254]
species_test = species_indices_perm[254:]

entrees_train_orig = entrees[perm[:166], :]
entrees_valid_orig = entrees[perm[166:254], :]

X_train_aug_orig, Y_train_aug = augment_data(entrees_train_orig, Y_train, species_train, N_AUGMENTATION)
X_valid_aug_orig, Y_valid_aug = augment_data(entrees_valid_orig, Y_valid, species_valid, N_AUGMENTATION)

X_train_aug = (X_train_aug_orig - mu) / sigma
X_valid_aug = (X_valid_aug_orig - mu) / sigma

W_aug, err_train_aug = widrow_hoff(X_train_aug, Y_train_aug, taux_apprentissage=0.008, epochs=20)
scores_valid_aug = compute_scores(X_valid_aug, W_aug)
scores_test_aug = compute_scores(X_test, W_aug)

auc_valid_aug = []
auc_test_aug = []
ap_valid_aug = []
ap_test_aug = []

for i, species in enumerate(sp):
    fpr_v, tpr_v, auc_v = ROC(Y_valid_aug[:, i], scores_valid_aug[:, i])
    fpr_t, tpr_t, auc_t = ROC(Y_test[:, i], scores_test_aug[:, i])
    
    recall_v, prec_v, ap_v = PR(Y_valid_aug[:, i], scores_valid_aug[:, i])
    recall_t, prec_t, ap_t = PR(Y_test[:, i], scores_test_aug[:, i])
    
    auc_valid_aug.append(auc_v)
    auc_test_aug.append(auc_t)
    ap_valid_aug.append(ap_v)
    ap_test_aug.append(ap_t)
    
    print(f"\n{species:12s}:")
    print(f"  AUC (validation) : {auc_v:.4f}   |   AUC (test) : {auc_t:.4f}")
    print(f"  AP  (validation) : {ap_v:.4f}   |   AP  (test) : {ap_t:.4f}")

mauc_valid_aug = np.mean(auc_valid_aug)
mauc_test_aug = np.mean(auc_test_aug)
map_valid_aug = np.mean(ap_valid_aug)
map_test_aug = np.mean(ap_test_aug)

print("\n" + "="*80)
print("MOYENNES")
print("="*80)
print(f"MAUC (validation) : {mauc_valid_aug:.4f}   |   MAUC (test) : {mauc_test_aug:.4f}")
print(f"MAP  (validation) : {map_valid_aug:.4f}   |   MAP  (test) : {map_test_aug:.4f}")

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("Courbes ROC par espèce - Données augmentées", fontsize=18, fontweight='bold', y=0.95)

roc_curves_valid_aug = []
roc_curves_test_aug = []

for i, species in enumerate(sp):
    fpr_v, tpr_v, _ = ROC(Y_valid_aug[:, i], scores_valid_aug[:, i])
    fpr_t, tpr_t, _ = ROC(Y_test[:, i], scores_test_aug[:, i])
    
    roc_curves_valid_aug.append((fpr_v, tpr_v))
    roc_curves_test_aug.append((fpr_t, tpr_t))
    
    ax = axes[i, 0]
    ax.plot(fpr_v, tpr_v, linewidth=3, label=f'{species} (AUC={auc_valid_aug[i]:.3f})',
            color=colors[i], marker='o', markersize=4, markevery=max(1, len(fpr_v)//10))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, label='Aléatoire (AUC=0.5)')
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12, fontweight='bold')
    ax.set_title(f'{species} - Validation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    ax = axes[i, 1]
    ax.plot(fpr_t, tpr_t, linewidth=3, label=f'{species} (AUC={auc_test_aug[i]:.3f})',
            color=colors[i], marker='o', markersize=4, markevery=max(1, len(fpr_t)//10))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, label='Aléatoire (AUC=0.5)')
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12, fontweight='bold')
    ax.set_title(f'{species} - Test', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()




    
    



# %%



