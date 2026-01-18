import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

try:
    data = pd.read_csv('pareto_archive.csv')
    
    required_columns = ['Latency', 'Energy', 'Cost']
    if not all(col in data.columns for col in required_columns):
        print(f"Erreur : Le fichier CSV doit contenir les colonnes {required_columns}")
        print(f"Colonnes trouvées : {list(data.columns)}")
        exit(1)
    
    n_points = len(data)
    if n_points < 2:
        print(f"Erreur : Pas assez de points ({n_points}) pour visualiser la frontière Pareto.")
        exit(1)
    
    #compter les occurrences de chaque solution unique
    data_unique = data.drop_duplicates()
    n_unique = len(data_unique)
    
    freq = data.groupby(['Latency', 'Energy', 'Cost']).size().reset_index(name='Frequency')
    data_with_freq = data_unique.merge(freq, on=['Latency', 'Energy', 'Cost'])
    
    #affichage des statistiques
    print("=" * 60)
    print("STATISTIQUES DE L'ARCHIVE PARETO")
    print("=" * 60)
    print(f"Nombre total de lignes dans le CSV : {n_points}")
    print(f"Nombre de solutions UNIQUES non-dominées : {n_unique}")
    print(f"\nATTENTION : Le CSV contient {n_points - n_unique} doublons!")
    print(f"   (Certaines solutions apparaissent plusieurs fois)")
    print(f"\nDétail des solutions uniques et leurs fréquences:")
    print("-" * 60)
    for idx, row in data_with_freq.sort_values('Latency').iterrows():
        print(f"Solution {idx+1}: Latency={row['Latency']:.6f}, Energy={row['Energy']:.1f}, Cost={row['Cost']:.1f} -> Frequence: {row['Frequency']}")
    print("-" * 60)
    print(f"\nStatistiques sur les solutions uniques:")
    print(f"Latence (f1):")
    print(f"  Min = {data_unique['Latency'].min():.6f}, Max = {data_unique['Latency'].max():.6f}, Moyenne = {data_unique['Latency'].mean():.6f}")
    print(f"Énergie (f2):")
    print(f"  Min = {data_unique['Energy'].min():.4f}, Max = {data_unique['Energy'].max():.4f}, Moyenne = {data_unique['Energy'].mean():.4f}")
    print(f"Coût (f3):")
    print(f"  Min = {data_unique['Cost'].min():.4f}, Max = {data_unique['Cost'].max():.4f}, Moyenne = {data_unique['Cost'].mean():.4f}")
    print("=" * 60)
    

    x = data_unique['Latency'].values
    y = data_unique['Energy'].values
    z = data_unique['Cost'].values
    frequencies = data_with_freq['Frequency'].values
    
    sizes = 100 + (frequencies - 1) * 50
    
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    z_sorted = z[sort_idx]
    sizes_sorted = sizes[sort_idx]
    

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x, y, z, 
                        c=x, cmap='viridis', s=sizes, 
                        edgecolors='white', linewidths=2,
                        alpha=0.85, label=f'Solutions Pareto ({n_unique} uniques)')

    if n_unique > 1:
        segments = []
        for i in range(len(x_sorted) - 1):
            segments.append([(x_sorted[i], y_sorted[i], z_sorted[i]),
                           (x_sorted[i+1], y_sorted[i+1], z_sorted[i+1])])
        
        line_collection = Line3DCollection(segments, colors='red', 
                                          linestyle='--', linewidth=1.5, 
                                          alpha=0.6, label='Frontière Pareto')
        ax.add_collection3d(line_collection)

    ax.set_xlabel('Latence (f1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Consommation Énergie (f2)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Coût d\'Infrastructure (f3)', fontsize=12, fontweight='bold')
    


    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Niveau de Latence', fontsize=11, fontweight='bold')

    ax.legend(loc='upper left', fontsize=9)
    
    if n_unique < n_points:
        ax.text2D(0.02, 0.98, f'Note: Taille des points ∝ fréquence\n({n_points-n_unique} doublons dans le CSV)', 
                 transform=ax.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    margin = 0.05
    ax.set_xlim([x.min() * (1 - margin), x.max() * (1 + margin)])
    ax.set_ylim([y.min() * (1 - margin), y.max() * (1 + margin)])
    ax.set_zlim([z.min() * (1 - margin), z.max() * (1 + margin)])

    plt.tight_layout()
    
    save_path = 'pareto_frontier_3d.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegardé dans '{save_path}'")
    
    print(f"\nAffichage de {n_unique} solutions uniques dans la visualisation 3D...")
    if n_unique < n_points:
        print(f"   ({n_points} lignes dans le CSV, {n_points - n_unique} doublons)")
    print("(La taille des points reflète leur fréquence dans le CSV)")
    print("(Vous pouvez faire tourner la vue avec la souris)")
    plt.show()

except FileNotFoundError:
    print("Erreur : Le fichier 'pareto_archive.csv' n'existe pas encore.")
    print("Assurez-vous d'avoir exécuté l'algorithme MOHS au préalable.")
except Exception as e:
    print(f"Erreur lors de la visualisation : {e}")
    import traceback
    traceback.print_exc()
