import pandas as pd
import numpy as np

def calculate_metrics():
    try:
        df = pd.read_csv('pareto_archive.csv')
        data = df.values
        n_points = len(data)
        
        if n_points < 2:
            print("Pas assez de points pour calculer les métriques.")
            return


        distances = []
        for i in range(n_points):
            d_i = np.sqrt(np.sum((data - data[i])**2, axis=1))
            distances.append(np.min(d_i[d_i > 0]))
        
        d_mean = np.mean(distances)
        spacing = np.sqrt(np.sum((distances - d_mean)**2) / (n_points - 1))


        ref_point = np.max(data, axis=0) * 1.1
        
        mins = np.min(data, axis=0)
        maxs = ref_point
        
        samples = np.random.uniform(mins, maxs, (100000, 3))
        
        dominated_count = 0
        for sample in samples:
            if np.any(np.all(data <= sample, axis=1)):
                dominated_count += 1
        
        total_volume = np.prod(maxs - mins)
        hypervolume = (dominated_count / 100000) * total_volume

        print("="*40)
        print(f" ANALYSE DE QUALITÉ (PARETO) ")
        print("="*40)
        print(f"Nombre de solutions : {n_points}")
        print(f"Indicateur de Spacing (S)  : {spacing:.6f}")
        print(f"Indicateur Hypervolume (HV): {hypervolume:.2e}")
        print("="*40)
        print("INTERPRÉTATION :")
        print("- Spacing proche de 0 = Excellente répartition des solutions.")
        print("- Plus l'Hypervolume est grand, plus l'algo est performant.")

    except FileNotFoundError:
        print("Fichier 'pareto_archive.csv' introuvable. Lancez main.py d'abord.")

if __name__ == "__main__":
    calculate_metrics()