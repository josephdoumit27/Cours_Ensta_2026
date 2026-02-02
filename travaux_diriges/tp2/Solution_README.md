# Solutions TP2 - Programmation Parallèle avec MPI

## Installation de l'environnement

### Windows
1. **Microsoft MPI** est déjà installé
2. **Packages Python** sont déjà installés (mpi4py, numpy, Pillow, matplotlib)

### Vérification
```bash
mpiexec
python -c "from mpi4py import MPI; print('MPI OK')"
```

## Fichiers de solution

### 1. Ensemble de Mandelbrot

#### Solution_mandelbrot_bloc.py
Parallélisation avec partition par blocs de lignes équitables.
```bash
mpiexec -n 4 python Solution_mandelbrot_bloc.py
```
- Chaque processus calcule un bloc continu de lignes
- Distribution équitable (gestion du reste de la division)
- Image sauvegardée: `mandelbrot_bloc.png`

#### Solution_mandelbrot_statique.py
Parallélisation avec répartition cyclique (interleaving).
```bash
mpiexec -n 4 python Solution_mandelbrot_statique.py
```
- Chaque processus calcule les lignes y où y % nbp == rank
- Meilleur équilibrage de charge (lignes coûteuses réparties)
- Image sauvegardée: `mandelbrot_statique.png`

#### Solution_mandelbrot_maitre_esclave.py
Stratégie maître-esclave avec équilibrage dynamique.
```bash
mpiexec -n 4 python Solution_mandelbrot_maitre_esclave.py
```
- Processus 0 = maître (distribution des tâches)
- Processus 1-3 = esclaves (calcul des lignes)
- Équilibrage dynamique optimal
- Image sauvegardée: `mandelbrot_maitre_esclave.png`

### 2. Produit Matrice-Vecteur

#### Solution_matvec_colonne.py
Produit matrice-vecteur avec partitionnement par colonnes.
```bash
mpiexec -n 4 python Solution_matvec_colonne.py
```
- Chaque processus génère Nloc colonnes de A
- Calcul: v_local = A_local . u_local
- Réduction avec Allreduce (somme des contributions)
- Vérification automatique du résultat

#### Solution_matvec_ligne.py
Produit matrice-vecteur avec partitionnement par lignes.
```bash
mpiexec -n 4 python Solution_matvec_ligne.py
```
- Chaque processus génère Nloc lignes de A
- Calcul: v_local = A_local . u (vecteur complet)
- Rassemblement avec Allgather
- Vérification automatique du résultat

### 3. Exercice Théorique

#### Solution_exercice_amdhal.md
Solution complète de l'exercice sur les lois d'Amdahl et Gustafson:
- Calcul de l'accélération maximale (Amdahl)
- Nombre optimal de nœuds
- Application de la loi de Gustafson

## Tests de performance

### Exemple de benchmark Mandelbrot
```bash
# Test avec différents nombres de processus
for n in 1 2 4 8; do
    echo "=== Test avec $n processus ==="
    mpiexec -n $n python Solution_mandelbrot_bloc.py
done
```

### Résultats obtenus (exemple)
- 1 processus: ~4.5s
- 4 processus: ~1.15s (speedup ~3.9)
- Méthode bloc: plus efficace pour communication
- Méthode maître-esclave: meilleur équilibrage mais overhead

## Comparaison des approches

### Mandelbrot
| Approche | Avantages | Inconvénients |
|----------|-----------|---------------|
| Bloc | Simple, peu de communication | Déséquilibre de charge possible |
| Statique (interleaving) | Meilleur équilibrage | Plus de communication |
| Maître-esclave | Équilibrage optimal | Overhead de communication, maître peut être goulot |

### Produit Matrice-Vecteur
| Approche | Communication | Mémoire |
|----------|---------------|---------|
| Colonnes | Allreduce (somme) | Nloc colonnes par processus |
| Lignes | Allgather (concat) | Nloc lignes par processus |

## Notes importantes

- **Dimension** doit être divisible par le nombre de processus pour matvec
- **Encodage** des caractères: problèmes résolus (Windows CP1252)
- **Images** générées dans le répertoire courant
- **Vérification** automatique des résultats dans matvec

## Pour aller plus loin

1. Mesurer le speedup avec différents nombres de processus
2. Tester avec des dimensions différentes (1024x1024, 2048x2048)
3. Analyser l'efficacité parallèle (speedup/nbp)
4. Comparer avec la version vectorisée (mandelbrot_vec.py)
