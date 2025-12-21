from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def afficher_structure(model, nom):
    """Affiche la structure d'un reseau de maniere textuelle"""
    print(f"\n{nom}:")
    print("Structure des dependances:")
    for edge in model.edges():
        print(f"  {edge[0]} -> {edge[1]}")
    print(f"Total: {len(model.nodes())} noeuds, {len(model.edges())} aretes")
    print("Noeuds racines:", [node for node in model.nodes() if len(model.get_parents(node)) == 0])


# =============================================================================
# EXEMPLE 2 : PANNE DE VÉHICULE DE TRANSPORT
# =============================================================================
print("\n" + "="*50)
print("TRANSPORT : DIAGNOSTIC PANNE MOTEUR")
print("="*50)

# Structure :
# - AgeBatterie influence EtatElectrique
# - EtatElectrique et NiveauCarburant influencent DemarrageMoteur
# - DemarrageMoteur influence VoyantTableauBord
model_transport = BayesianNetwork([
    ('AgeBatterie', 'EtatElectrique'),
    ('EtatElectrique', 'DemarrageMoteur'),
    ('NiveauCarburant', 'DemarrageMoteur'),
    ('DemarrageMoteur', 'VoyantTableauBord')
])

# CPDs (0 = Vieux/Vide/Echec , 1 = Neuf/Plein/Succès)
cpd_age = TabularCPD(variable='AgeBatterie', variable_card=2, values=[[0.4], [0.6]])
cpd_carb = TabularCPD(variable='NiveauCarburant', variable_card=2, values=[[0.15], [0.85]])

# EtatElectrique (Parent: AgeBatterie)
cpd_elec = TabularCPD(
    variable='EtatElectrique', variable_card=2,
    values=[[0.8, 0.1],   # Probabilité Elec = HS
            [0.2, 0.9]],  # Probabilité Elec = OK
    evidence=['AgeBatterie'], evidence_card=[2]
)

# DemarrageMoteur (Parents: Elec, Carburant)
cpd_moteur = TabularCPD(
    variable='DemarrageMoteur', variable_card=2,
    values=[[1.0, 0.9, 0.8, 0.02], # Probabilité Echec
            [0.0, 0.1, 0.2, 0.98]], # Probabilité Succès
    evidence=['EtatElectrique', 'NiveauCarburant'], evidence_card=[2, 2]
)

# VoyantTableauBord (Parent: DemarrageMoteur)
cpd_voyant = TabularCPD(
    variable='VoyantTableauBord', variable_card=2,
    values=[[0.05, 0.95],  # Voyant Eteint
            [0.95, 0.05]], # Voyant Allumé (Alerte)
    evidence=['DemarrageMoteur'], evidence_card=[2]
)

model_transport.add_cpds(cpd_age, cpd_carb, cpd_elec, cpd_moteur, cpd_voyant)
infer_trans = VariableElimination(model_transport)

# Calcul : Si le voyant est allumé, quelle est la probabilité que le carburant soit vide ?
res_panne = infer_trans.query(variables=['NiveauCarburant'], 
                              evidence={'VoyantTableauBord': 1})
print(f"P(Carburant=Vide | Voyant=Allumé) = {res_panne.values[0]:.3f}")

# =============================================================================
# AFFICHAGE DES STRUCTURES (Utilisant votre fonction existante)
# =============================================================================
afficher_structure(model_transport, "RESEAU PANNE VEHICULE")



# =============================================================================
# VISUALISATION DES ReSEAUX BAYeSIENS
# =============================================================================
print("\n" + "="*50)
print("VISUALISATION DES RESEAUX")
print("="*50)

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    
    def visualiser_reseau(model, titre):
        """Visualise un reseau bayesien avec matplotlib"""
        plt.figure(figsize=(10, 8))
        
        # Creer un graphe oriente
        G = nx.DiGraph()
        G.add_edges_from(model.edges())
        
        # Disposition des nœuds
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Dessiner les nœuds
        nx.draw_networkx_nodes(G, pos, 
                              node_size=2000, 
                              node_color='lightblue',
                              alpha=0.9,
                              edgecolors='black',
                              linewidths=1)
        
        # Dessiner les arêtes
        nx.draw_networkx_edges(G, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=25,
                              arrowstyle='->',
                              width=2)
        
        # Dessiner les labels
        nx.draw_networkx_labels(G, pos, 
                               font_size=10, 
                               font_weight='bold')
        
        plt.title(titre, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Afficher les informations du reseau
        print(f"Reseau: {titre}")
        print(f"Nombre de noeuds: {len(G.nodes())}")
        print(f"Nombre d'aretes: {len(G.edges())}")
        print(f"Noeuds: {list(G.nodes())}")
        print("-" * 40)
    
    # Visualiser les trois reseaux
    print("Generation des visualisations...")
    

    # Reseau 2: Reseau medical complexe
    visualiser_reseau(model_transport, "ETAPE 5")
    

    
except ImportError:
    print("Bibliotheques de visualisation non disponibles.")
    print("Installation recommandee: pip install matplotlib networkx")
