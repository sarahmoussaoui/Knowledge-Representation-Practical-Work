# TP Reseaux Bayesiens avec DiscreteBayesianNetwork
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Fonction d'affichage de la structure (conservée)
def afficher_structure(model, nom):
    """Affiche la structure d'un reseau de maniere textuelle"""
    print(f"\n{nom}:")
    print("Structure des dependances:")
    for edge in model.edges():
        print(f"  {edge[0]} -> {edge[1]}")
    print(f"Total: {len(model.nodes())} noeuds, {len(model.edges())} aretes")
    print("Noeuds racines:", [node for node in model.nodes() if len(model.get_parents(node)) == 0])

# Fonction de visualisation (conservée)
def visualiser_reseau(model, titre):
    """Visualise un reseau bayesien avec matplotlib"""
    plt.figure(figsize=(10, 8))
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    pos = nx.spring_layout(G, k=2.5, iterations=50) # Ajustement de l'espacement
    
    nx.draw_networkx_nodes(G, pos, node_size=2000, 
                           node_color='lightcoral', alpha=0.9,
                           edgecolors='black', linewidths=1)
    
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                           arrowsize=25, arrowstyle='->', width=2)
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title(titre, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# =============================================================================
# ETAPE 5 : DIAGNOSTIC MEDICAL (Polyarthrite Rhumatoide)
# =============================================================================
print("\n" + "="*50)
print("ETAPE 5 : DIAGNOSTIC MEDICAL (Polyarthrite Rhumatoide)")
print("=====================================================")

# Structure du Réseau Bayésien (RB) pour le diagnostic 
model_medical = BayesianNetwork([
    ('Genetic', 'PR'),
    ('PR', 'Rigidite'),
    ('PR', 'TestPositif')
])

# Variables binaires (0=Non, 1=Oui/Présent)

# CPDs (Tables de Probabilités Conditionnelles)
# 1. Genetic (Racine)
cpd_genetic = TabularCPD(variable='Genetic', variable_card=2,
                         values=[[0.70], [0.30]]) # P(Genetic=0)=0.7, P(Genetic=1)=0.3

# 2. PR (Dépend de Genetic)
cpd_pr = TabularCPD(
    variable='PR',
    variable_card=2,
    values=[
        # P(PR=0 | Genetic=0), P(PR=0 | Genetic=1)
        [0.98, 0.70], 
        # P(PR=1 | Genetic=0), P(PR=1 | Genetic=1)
        [0.02, 0.30]
    ],
    evidence=['Genetic'],
    evidence_card=[2]
)

# 3. Rigidite (Dépend de PR)
cpd_rigidite = TabularCPD(variable='Rigidite', variable_card=2,
                          values=[
                              # P(Rigidite=0 | PR=0), P(Rigidite=0 | PR=1)
                              [0.90, 0.20],
                              # P(Rigidite=1 | PR=0), P(Rigidite=1 | PR=1)
                              [0.10, 0.80]
                          ],
                          evidence=['PR'], evidence_card=[2])

# 4. TestPositif (Dépend de PR)
cpd_test = TabularCPD(variable='TestPositif', variable_card=2,
                      values=[
                          # P(Test=0 | PR=0), P(Test=0 | PR=1) (Spécificité et Sensibilité)
                          [0.85, 0.10], 
                          # P(Test=1 | PR=0), P(Test=1 | PR=1)
                          [0.15, 0.90]
                      ],
                      evidence=['PR'], evidence_card=[2])

model_medical.add_cpds(cpd_genetic, cpd_pr, cpd_rigidite, cpd_test)

print("Modele medical valide:", model_medical.check_model())

# Inference
inference_medical = VariableElimination(model_medical)

# Scenario A: Rigidité matinale (symptôme) observée, Test Négatif
print("\n--- CAS MEDICAL A: Rigidite=oui + TestPositif=non ---")
res_pr_A = inference_medical.query(variables=['PR'],
                                   evidence={'Rigidite': 1, 'TestPositif': 0})
print(f"- P(PR=oui | Rigidite=oui, TestPositif=non) = {res_pr_A.values[1]:.3f}")

# Scenario B: Rigidité matinale (symptôme) observée, Test Positif
print("\n--- CAS MEDICAL B: Rigidite=oui + TestPositif=oui ---")
res_pr_B = inference_medical.query(variables=['PR'],
                                   evidence={'Rigidite': 1, 'TestPositif': 1})
print(f"- P(PR=oui | Rigidite=oui, TestPositif=oui) = {res_pr_B.values[1]:.3f}")

# Visualisation
visualiser_reseau(model_medical, "ETAPE 5 - DIAGNOSTIC DE POLYARTHRITE RHUMATOIDE")

afficher_structure(model_medical, "4. DIAGNOSTIC MEDICAL")

