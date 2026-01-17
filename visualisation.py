import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lecture de l'archive générée par le MOHS
try:
    data = pd.read_csv('pareto_archive.csv')
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Création du nuage de points 3D (Latence, Énergie, Coût)
    # Nous utilisons les 3 objectifs définis dans votre modélisation [cite: 54, 56, 60]
    img = ax.scatter(data['Latency'], data['Energy'], data['Cost'], 
                     c=data['Latency'], cmap='plasma', s=60, edgecolors='w')

    # Étiquettes des axes selon votre problématique [cite: 31]
    ax.set_xlabel('Latence (f1)')
    ax.set_ylabel('Consommation Énergie (f2)')
    ax.set_zlabel('Coût d\'Infrastructure (f3)')
    ax.set_title('Frontière de Pareto 3D - Groupe 3\n(MOHS avec Réinjection de la Dominance)')

    fig.colorbar(img, ax=ax, label='Niveau de Latence')
    
    print(f"Affichage de {len(data)} solutions non-dominées.")
    plt.show()

except FileNotFoundError:
    print("Erreur : Le fichier 'pareto_archive.csv' n'existe pas encore.")