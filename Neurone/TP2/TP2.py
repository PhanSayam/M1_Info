  # Normalisation (important : évite dominances d'échelle)
mu = entrees.mean(axis=0) # moyenne par colonne
sigma = entrees.std(axis=0) # écart-type par colonne
entrees = (entrees - mu) / sigma # standardisation donc on garde shape (N,d) et on modifie les valeurs pour centrer/réduire les données

# fonction pour diviser les données en un ensemble d'entrainement et un ensemble de test
def donnees_entrainement_test(entrees, sorties, proportion_test):
    nb_exemples = entrees.shape[0]
    nb_test = int(nb_exemples * proportion_test)
    indices = np.random.permutation(nb_exemples)
    test_idx, train_idx = indices[:nb_test], indices[nb_test:]
    return entrees[train_idx,:], sorties[train_idx,:], entrees[test_idx,:], sorties[test_idx,:]

'''
Définition

Epoch :
Dans le contexte de l'entraînement d'un modèle, l'"epoch" est un terme utilisé 
pour référer à une itération où le modèle voit tout le training set pour mettre à jour ses coefficients.
'''

def widrow_hoff(entrees, sorties, taux_apprentissage=0.01, epochs=10):
    ligne = entrees.shape[0]
    nb_attributs = entrees.shape[1]
    sortie_reelle = sorties.shape[1]

    # initialisation stable des poids / biais
    facteur_normalisation = 1 / np.sqrt(nb_attributs)
    matrice_poids = np.random.randn(nb_attributs, sortie_reelle) * facteur_normalisation
    vecteur_biais = np.zeros((1, sortie_reelle))

    tab_erreur_moyenne = []

    for epoch in range(epochs):
        # On mélange les exemples pour chaque itérations
        perm = np.random.permutation(ligne)
        somme = 0.0

        # Donc on parcourt les exemples dans un ordre aléatoire, puis pour j chaque sortie on calcule la sortie du perceptron Sj
        # on calcule l'erreur entre la sortie attendue SA et la sortie réelle Sj, puis on met à jour les poids et biais en conséquence
        # somme est la somme des erreurs quadratiques pour cette itération, 
        # on calcule l'erreur moyenne à la fin de l'itération et on l'ajoute à tab_erreur_moyenne
        for exemple in perm:
            entrees_utilisees = entrees[exemple, :]
            SA = sorties[exemple, :]
            for j in range(sortie_reelle):
                Sj = np.dot(entrees_utilisees, matrice_poids[:, j]) + vecteur_biais[0, j]
                erreur = SA[j] - Sj
                vecteur_biais[0, j] += taux_apprentissage * erreur
                for i in range(nb_attributs):
                    matrice_poids[i, j] += taux_apprentissage * erreur * entrees_utilisees[i]
                somme += erreur**2
        erreur_moyenne = somme / (nb_attributs * sortie_reelle) 
        tab_erreur_moyenne.append(erreur_moyenne)

    return matrice_poids, vecteur_biais, tab_erreur_moyenne


donnees_entrainement, sorties_entrainement, donnees_test, sorties_test = donnees_entrainement_test(entrees, sorties, 0.4)
matrice_poids, vecteur_biais, tab_erreur_moyenne = widrow_hoff(donnees_entrainement, sorties_entrainement, 0.008, 100)
matrice_poids_test, vecteur_biais_test, tab_erreur_moyenne_test = widrow_hoff(donnees_test, sorties_test, 0.008, 100)

plt.plot(tab_erreur_moyenne, label="erreur moyenne TRAIN")
plt.plot(tab_erreur_moyenne_test, label="erreur moyenne TEST")
plt.xlabel("Itérations (epochs)")
plt.ylabel("Erreur moyenne")
plt.title("Erreur donnees de test et entrainement")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

plt.plot(vecteur_biais[0, :], marker='o')
plt.xlabel("Index des biais (classe)")
plt.ylabel("Valeurs des biais")
plt.title("Valeurs des biais après entraînement")
plt.xticks(range(len(sp)), sp, rotation=30)
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

plt.plot(matrice_poids[:, 0], label=f"Poids -> {sp[0]}")
plt.plot(matrice_poids[:, 1], label=f"Poids -> {sp[1]}")
plt.plot(matrice_poids[:, 2], label=f"Poids -> {sp[2]}")
plt.legend()
plt.xlabel("Index des poids (attributs)")
plt.ylabel("Valeurs des poids")
plt.title("Valeurs des poids après entraînement")
plt.xticks(range(matrice_poids.shape[0]), ["bill_len","bill_dep","flipper_len","body_mass"])
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()



    
