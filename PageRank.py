import pandas as pd
from collections import defaultdict

def compute_pagerank_inbound(edges_file, damping=0.85, max_iter=100, tol=1e-8):
    """
    Calcule le PageRank depuis un fichier edges.csv (FromNode,ToNode),
    SANS construire de matrice NxN.
    
    edges_file : chemin vers edges.csv
    damping    : facteur d'amortissement (0.85 par défaut)
    max_iter   : nombre max d'itérations
    tol        : critère d'arrêt (variation)
    
    Retourne : un dictionnaire {node_id: pagerank_score}
    """

    print("[1] Lecture du edges.csv...")
    df_edges = pd.read_csv(edges_file)  # Suppose un header : FromNode,ToNode

    # Ensemble de tous les nœuds
    print("[2] Construction de l'ensemble des nœuds...")
    all_nodes = set(df_edges["FromNode"]).union(set(df_edges["ToNode"]))
    n = len(all_nodes)
    print(f"    Nombre total de nœuds = {n}")

    # Dictionnaires inbound_links et outdegree
    print("[3] Construction inbound_links & outdegree...")
    inbound_links = defaultdict(list)  # inbound_links[v] = liste des u qui pointent vers v
    outdegree = defaultdict(int)       # outdegree[u] = nombre de liens sortants depuis u

    for row in df_edges.itertuples(index=False):
        src = row.FromNode
        tgt = row.ToNode
        inbound_links[tgt].append(src)
        outdegree[src] += 1

    # Initialisation : PR(u) = 1/n au départ
    print("[4] Initialisation PageRank...")
    pagerank = {node: 1.0/n for node in all_nodes}

    # Itérations
    print("[5] Itérations PageRank (damping=", damping, ")...")
    for iteration in range(max_iter):
        new_pagerank = {}
        # La somme globale de PR(u) (ça devrait rester proche de 1, 
        # mais selon la formule on peut s'en servir pour d'éventuels cas de dangling)
        sum_pr = sum(pagerank.values())

        for v in all_nodes:
            # Contribution entrante
            in_sum = 0.0
            for u in inbound_links[v]:
                # PR(u) / outdegree(u)
                in_sum += pagerank[u] / outdegree[u]

            # PageRank(v)
            # (1 - d)*(somme PR(u)/n) + d*in_sum
            new_pagerank[v] = (1 - damping)*(sum_pr / n) + damping*in_sum

        # Évaluer la variation entre l'itération k et k+1
        diff = sum(abs(new_pagerank[v] - pagerank[v]) for v in all_nodes)

        pagerank = new_pagerank

        if diff < tol:
            print(f"    Convergence atteinte à l'itération {iteration+1}, diff={diff}")
            break

    return pagerank


def load_names(names_file):
    """
    Lit names.csv (qui possède une colonne 'Name')
    et renvoie une liste 'node_names' de longueur N.
    Index de ligne = ID du nœud, 
    Valeur dans la colonne 'Name' = nom du nœud.
    """
    df_names = pd.read_csv(names_file)  # Suppose un header "Name"
    # Si pas de header, adapter : header=None, names=["Name"]
    node_names = df_names["Name"].tolist()
    return node_names


if __name__ == "__main__":
    edges_file = "edges.csv"   # chemin vers edges.csv
    names_file = "names.csv"   # chemin vers names.csv (facultatif)

    # 1) Calcul du PageRank via inbound_links
    pagerank_scores = compute_pagerank_inbound(edges_file, damping=0.85, max_iter=100, tol=1e-8)

    # 2) Tri des nœuds par PageRank décroissant
    sorted_pr = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

    # 3) Chargement des noms (facultatif)
    try:
        node_names = load_names(names_file)  # liste
    except:
        node_names = []  # si on ne peut pas charger le fichier

    # 4) Affichage du top 20
    print("\n=== Top 20 (PageRank) ===")
    for i in range(20):
        node_id, score = sorted_pr[i]
        if node_id < len(node_names):
            print(f"{i+1}. Node={node_id}, Name='{node_names[node_id]}', score={score}")
        else:
            print(f"{i+1}. Node={node_id}, score={score}")
