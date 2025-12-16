from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = Image.open(r"C:\Users\phand\Documents\M1\Neurone\TP1\cachalot.jpeg") 
m3d = np.array(im)  
m2d = np.sum(m3d, axis=2)

plt.subplot(2, 1, 1)
plt.imshow(m3d)
plt.title("Image originale")
plt.subplot(2, 1, 2)
plt.imshow(m2d, cmap="gray")
plt.title("Image en niveaux de gris")
plt.show()


# Représente une cellule ganglionnaire de la rétine
poids = np.array([[-1, -1, -1],
                  [-1,  8, -1],
                  [-1, -1, -1]]) 

poids_verticaux = np.array([[ -1,   2,  -1],
                            [ -1,   2,  -1],
                            [ -1,   2,  -1]])

poids_horizontaux = np.array([[ -1,  -1,  -1],
                              [  2,   2,   2],
                              [ -1,  -1,  -1]])

# Fonctions d'activations
def heavyside(x):
    if x < 0:
        return 0
    else:
        return 1
    
def identite(x):
    return x

# Calcul de la sortie de la cellule ganglionnaire
def somme_poids_entree(m2d, poids, seuil=765):
    # On créer une matrice de sortie de meme taille que l image d entree
    lignes, colonnes = m2d.shape
    sortie = np.zeros((lignes, colonnes))
    
    for i in range(1, lignes - 1):
        for j in range(1, colonnes - 1):
            # Extraction matrice locale 3x3 autour du pixel (i, j)
            matrice_locale = m2d[i-1:i+2, j-1:j+2]
            
            # E w_ij * matrice_locale 
            somme_poids_entree = np.sum(matrice_locale * poids)
            
            # Seuil 
            if abs(somme_poids_entree) > seuil : # Ici on fait une valeur absolue pour detecter les contours clairs et foncés
                # Fonction d'activation
                sortie[i, j] = heavyside(somme_poids_entree)
            else:
                sortie[i, j] = 255  # Pixel blanc
    return sortie


contours = somme_poids_entree(m2d, poids)
plt.subplot(2, 1, 1)
plt.imshow(m2d, cmap="gray")
plt.title("Image en niveaux de gris")
plt.subplot(2, 1, 2)
plt.imshow(contours, cmap="gray")
plt.title("Contours détectés")
plt.show()


# Détection des contours horizontaux et verticaux
horizontales = somme_poids_entree(contours, poids_horizontaux)
plt.subplot(2, 1, 1)
plt.imshow(horizontales, cmap="gray")
plt.title("Contours horizontaux détectés")
plt.show()

verticales = somme_poids_entree(contours, poids_verticaux)
plt.subplot(2, 1, 1)
plt.imshow(verticales, cmap="gray")
plt.title("Contours verticaux détectés")
plt.show()




# Cas seuil trop bas
contours_seuil_bas = somme_poids_entree(m2d, poids, seuil=100)
plt.subplot(2, 1, 1)
plt.imshow(contours_seuil_bas, cmap="gray")
plt.title("Contours détectés avec seuil bas")

# Cas seuil trop haut
contours_seuil_haut = somme_poids_entree(m2d, poids, seuil=1400)
plt.subplot(2, 1, 2)
plt.imshow(contours_seuil_haut, cmap="gray")
plt.title("Contours détectés avec seuil haut")
plt.show()
