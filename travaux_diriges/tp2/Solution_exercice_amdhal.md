# Solution Exercice 3 - Entraînement pour l'examen écrit

## Données du problème

Alice a parallélisé un code où la partie parallèle représente **90% du temps d'exécution séquentiel**.

## Question 1: Accélération maximale selon la loi d'Amdahl

La loi d'Amdahl est donnée par la formule:

$$S(n) = \frac{1}{(1-p) + \frac{p}{n}}$$

où:
- $p$ est la fraction parallélisable du programme (ici $p = 0.9$)
- $n$ est le nombre de processeurs
- $S(n)$ est l'accélération (speedup)

Pour $n \to \infty$, on obtient l'accélération maximale:

$$S_{\max} = \frac{1}{1-p} = \frac{1}{1-0.9} = \frac{1}{0.1} = 10$$

**Réponse**: L'accélération maximale théorique qu'Alice peut obtenir est **10**.

Cela signifie que même avec un nombre infini de processeurs, le programme ne pourra jamais s'exécuter plus de 10 fois plus vite que la version séquentielle, car 10% du code reste séquentiel.

## Question 2: Nombre raisonnable de nœuds de calcul

Pour déterminer le nombre raisonnable de nœuds, calculons l'accélération pour différentes valeurs de $n$:

| Nombre de nœuds (n) | Accélération S(n) | Efficacité E(n) = S(n)/n |
|---------------------|-------------------|--------------------------|
| 1                   | 1.00              | 100%                     |
| 2                   | 1.82              | 91%                      |
| 4                   | 3.08              | 77%                      |
| 8                   | 4.71              | 59%                      |
| 16                  | 6.40              | 40%                      |
| 32                  | 7.74              | 24%                      |
| 64                  | 8.77              | 14%                      |

L'efficacité diminue rapidement après 8 nœuds. Un bon compromis serait d'utiliser **entre 4 et 8 nœuds**, où:
- Avec 4 nœuds: accélération de ~3, efficacité de 77%
- Avec 8 nœuds: accélération de ~4.7, efficacité de 59%

Au-delà de 16 nœuds, l'efficacité devient inférieure à 50%, ce qui signifie qu'on gaspille plus de la moitié des ressources CPU.

## Question 3: Accélération avec la loi de Gustafson

Alice observe une accélération maximale de **4** (au lieu des 10 théoriques). Cela suggère que la partie séquentielle est plus importante en pratique, ou qu'il y a des surcoûts de communication.

Si on considère que l'accélération réelle est 4, on peut calculer la fraction parallèle effective:
$$4 = \frac{1}{(1-p_{eff}) + \frac{p_{eff}}{n}}$$

Avec $n$ grand, on a: $4 \approx \frac{1}{1-p_{eff}}$, donc $p_{eff} = 0.75$ (75% du code est réellement parallélisable).

### Loi de Gustafson

La loi de Gustafson considère que la taille du problème augmente avec le nombre de processeurs:

$$S(n) = (1-p) + n \cdot p$$

où $p$ est la fraction parallèle du temps d'exécution sur $n$ processeurs.

Si on double la quantité de données avec un algorithme de complexité linéaire:
- Le temps séquentiel double
- Le temps parallèle double également

Avec $p_{eff} = 0.75$ et en supposant le même nombre de nœuds qui donnait une accélération de 4:

$$S_{nouveau} = (1-0.75) + n \cdot 0.75 = 0.25 + 0.75n$$

Si on utilisait $n=5$ nœuds pour obtenir environ l'accélération de 4 avec Amdahl:
$$S_{Gustafson}(5) = 0.25 + 0.75 \times 5 = 4$$

Avec le double de données et le même nombre de processeurs, selon Gustafson, l'accélération reste **environ 4** car la fraction parallèle reste constante.

Cependant, si Alice peut utiliser **plus de nœuds** avec le problème plus grand, l'accélération augmentera linéairement. Par exemple, avec 10 nœuds:
$$S_{Gustafson}(10) = 0.25 + 0.75 \times 10 = 7.75$$

**Conclusion**: 
- Avec le même nombre de nœuds: accélération reste similaire (~4)
- Avec le double de nœuds (scalabilité faible): accélération ~ 7.75
- La loi de Gustafson est plus optimiste car elle considère que la taille du problème augmente avec les ressources disponibles
