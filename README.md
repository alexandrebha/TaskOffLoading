PROJET OPTIMISATION METAHEURISTIQUE : TASK OFFLOADING FOG
1. PRESENTATION DU PROJET
Ce projet implémente l'algorithme MOHS (Multi-Objective Harmony Search) pour résoudre le problème du placement de tâches dans une infrastructure Fog Computing. L'objectif est de trouver le meilleur compromis entre trois critères souvent contradictoires : la latence, l'énergie et le coût.

2. FONCTIONNEMENT DE L'ALGORITHME (MOHS)
L'algorithme s'inspire de l'improvisation musicale pour trouver une harmonie (une solution) optimale :

HM (Harmony Memory) : Stocke les solutions actuelles.

Archive Pareto : Stocke les 50 meilleures solutions non-dominées.

Reinjection (Innovation) : 25% des nouvelles décisions sont prises directement depuis l'archive Pareto pour guider la recherche vers le front optimal.

Gestion des Contraintes : Utilisation de pénalités lourdes en cas de dépassement de la capacité des buffers des nœuds.

3. STRUCTURE DES FICHIERS
main.py : Lanceur principal de la simulation.

policies/heuristics/MOHS.py : Contient toute la logique de l'algorithme et de l'archive.

visualisation.py : Génère un graphique 3D de la frontière de Pareto.

quality_metrics.py : Calcule les indicateurs de Spacing et d'Hypervolume.

pareto_archive.csv : Fichier de données généré contenant les solutions optimales.

4. COMMENT LANCER LE PROJET
A. Installation des bibliothèques nécessaires : pip install pandas numpy matplotlib

B. Lancement de la simulation (génère l'archive) : python main.py --config configs/Pakistan/Heuristics/MOHS.yaml

C. Visualisation des résultats (Front de Pareto 3D) : python visualisation.py

D. Calcul des métriques de qualité (HV et Spacing) : python quality_metrics.py

5. RESULTATS OBTENUS (DATASET PAKISTAN)
TaskThrowRate : ~15.21% (Amélioration majeure par rapport aux 95% initiaux).

AvgPower (MOHS) : ~238 711 Watts (Consomme 30% de moins que l'algorithme Greedy).

Archive : 50 solutions de compromis trouvées.

Hypervolume (HV) : 4.07e+08 (Preuve de la convergence de l'algorithme).

6. ANALYSE COMPARATIVE
MOHS vs Greedy : MOHS est plus lent en latence mais beaucoup plus économe en énergie.

MOHS vs Random/RoundRobin : MOHS offre une meilleure stabilité et un meilleur respect des capacités matérielles grâce à sa gestion des contraintes.
