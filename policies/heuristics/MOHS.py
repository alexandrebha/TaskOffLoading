import random
import numpy as np
import csv

class MOHSPolicy:
    def __init__(self, hm_size=15, archive_size=50):
        self.hm_size = hm_size
        self.archive_limit = archive_size
        self.archive = []  # Archive Pareto pour stocker les solutions non-dominées [cite: 64]
        self.hm = []
        self.is_trained = False
        self.best_mapping = {}

    def act(self, env, task, train=False):
        """
        Méthode appelée par main.py. L'optimisation est lancée au premier appel.
        """
        if not self.is_trained:
            # Récupération de l'infrastructure via le scénario
            infra = getattr(env.scenario, 'infrastructure', None)
            nodes = list(infra.get_nodes().values()) if infra else []
            all_tasks = getattr(env, 'tasks', [task])
            
            if nodes:
                print(f"\n[MOHS] Lancement de l'optimisation Multi-Objectif...")
                self._run_mohs_optimization(all_tasks, nodes)
                self.is_trained = True
            else:
                print("[MOHS] Erreur : Infrastructure inaccessible.")

        # Attribution du nœud (Variable de décision xij) [cite: 6, 21, 61]
        task_id = getattr(task, 'task_id', getattr(task, 'id', 0))
        n_nodes = len(env.scenario.infrastructure.get_nodes()) if hasattr(env.scenario, 'infrastructure') else 1
        return self.best_mapping.get(task_id, task_id % n_nodes), None

    def _run_mohs_optimization(self, tasks, nodes):
        """
        Algorithme Multi-Objective Harmony Search (MOHS) [cite: 63, 64]
        """
        n_tasks = len(tasks)
        n_nodes = len(nodes)

        # 1. Initialisation de la HM
        for _ in range(self.hm_size):
            sol = [random.randint(0, n_nodes - 1) for _ in range(n_tasks)]
            objs = self._evaluate_objectives(sol, tasks, nodes)
            harmony = {'mapping': sol, 'objs': objs}
            self.hm.append(harmony)
            self._update_pareto_archive(harmony)

        # 2. Improvisation avec RÉINJECTION (Consigne Prof)
        for _ in range(250): 
            new_sol = [0] * n_tasks
            for i in range(n_tasks):
                rand = random.random()
                # Réinjection de l'archive pour utiliser la dominance dans la recherche
                if rand < 0.25 and len(self.archive) > 0:
                    new_sol[i] = random.choice(self.archive)['mapping'][i]
                elif rand < 0.85: # HMCR
                    new_sol[i] = random.choice(self.hm)['mapping'][i]
                    if random.random() < 0.3: # PAR
                        new_sol[i] = random.randint(0, n_nodes - 1)
                else:
                    new_sol[i] = random.randint(0, n_nodes - 1)

            objs = self._evaluate_objectives(new_sol, tasks, nodes)
            new_harmony = {'mapping': new_sol, 'objs': objs}
            self.hm[random.randint(0, self.hm_size - 1)] = new_harmony
            self._update_pareto_archive(new_harmony)

        # 3. Sauvegarde du mapping et EXPORTATION de l'Archive
        if self.archive:
            # On prend la première solution Pareto comme référence pour l'exécution
            best_sol = self.archive[0]
            mapping_list = best_sol['mapping']
            for idx, t in enumerate(tasks):
                t_id = getattr(t, 'task_id', getattr(t, 'id', idx))
                self.best_mapping[t_id] = mapping_list[idx]

            # ÉCRITURE du fichier CSV (C'est ici qu'on crée le fichier)
            try:
                with open('pareto_archive.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Latency', 'Energy', 'Cost'])
                    for sol in self.archive:
                        writer.writerow(sol['objs'])
                print(f"[MOHS] Archive sauvegardée ({len(self.archive)} solutions) dans 'pareto_archive.csv'.")
                print(f"[MOHS] Scores : f1={best_sol['objs'][0]:.4f}, f2={best_sol['objs'][1]:.2f}, f3={best_sol['objs'][2]:.2f}")
            except Exception as e:
                print(f"[MOHS] Erreur lors de la sauvegarde : {e}")

    def _evaluate_objectives(self, mapping, tasks, nodes):
        """
        Calcul des trois fonctions objectifs contradictoires [cite: 13, 17, 21, 47]
        """
        f1, f2, f3 = 0, 0, 0
        node_usage = [0] * len(nodes)

        for t_idx, n_idx in enumerate(mapping):
            t, n = tasks[t_idx], nodes[n_idx]
            t_size = getattr(t, 'task_size', 100)
            
            f1 += (t_size / (n.max_cpu_freq if n.max_cpu_freq > 0 else 1)) # f1: Latence [cite: 15, 49]
            f2 += (t_size * n.exe_energy_coef)                            # f2: Énergie [cite: 18, 52, 56]
            f3 += (t_size * n.idle_energy_coef * 0.75)                    # f3: Coût
            node_usage[n_idx] += t_size

        # Contraintes de capacité (Capj) [cite: 11, 21, 59]
        penalty = 0
        for i, load in enumerate(node_usage):
            max_cap = nodes[i].task_buffer.max_size
            if load > max_cap:
                penalty += (load - max_cap) * 1000000 
        return [f1 + penalty, f2 + penalty, f3 + penalty]

    def _update_pareto_archive(self, sol):
        """
        Mise à jour de l'archive selon la dominance de Pareto 
        """
        if any(self._is_dominated(sol['objs'], a['objs']) for a in self.archive):
            return
        self.archive = [a for a in self.archive if not self._is_dominated(a['objs'], sol['objs'])]
        self.archive.append(sol)
        if len(self.archive) > self.archive_limit:
            self.archive.pop(0)

    def _is_dominated(self, obj_a, obj_b):
        """
        Vérifie si la solution A est dominée par B [cite: 64]
        """
        return all(b <= a for a, b in zip(obj_a, obj_b)) and any(b < a for a, b in zip(obj_a, obj_b))