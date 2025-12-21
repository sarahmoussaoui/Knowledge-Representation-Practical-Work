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
# EXEMPLE 1 : RISQUE DE COLLISION DE SATELLITE
# =============================================================================
print("\n" + "="*50)
print("SATELLITE : DETECTION DE RISQUE DE COLLISION")
print("="*50)

# Structure : 
# - DensiteDebris et PrecisionRadar influencent l'AlerteCollision
# - AlerteCollision et CapaciteManoeuvre influencent la CollisionEffective
model_collision = BayesianNetwork([
    ('DensiteDebris', 'AlerteCollision'),
    ('PrecisionRadar', 'AlerteCollision'),
    ('AlerteCollision', 'CollisionEffective'),
    ('CapaciteManoeuvre', 'CollisionEffective')
])

# CPDs (0 = Faible/Non, 1 = Elevé/Oui)
cpd_debris = TabularCPD(variable='DensiteDebris', variable_card=2, values=[[0.7], [0.3]])
cpd_radar = TabularCPD(variable='PrecisionRadar', variable_card=2, values=[[0.2], [0.8]])
cpd_manoeuvre = TabularCPD(variable='CapaciteManoeuvre', variable_card=2, values=[[0.1], [0.9]])

# AlerteCollision (Parents: Debris, Radar) - 2^2 = 4 colonnes
cpd_alerte = TabularCPD(
    variable='AlerteCollision', variable_card=2,
    values=[[0.99, 0.70, 0.40, 0.05],  # Probabilité Alerte = Non
            [0.01, 0.30, 0.60, 0.95]], # Probabilité Alerte = Oui
    evidence=['DensiteDebris', 'PrecisionRadar'], evidence_card=[2, 2]
)

# CollisionEffective (Parents: Alerte, Manoeuvre) - 2^2 = 4 colonnes
cpd_collision = TabularCPD(
    variable='CollisionEffective', variable_card=2,
    values=[[1.0, 0.95, 0.60, 0.10],   # Probabilité Collision = Non
            [0.0, 0.05, 0.40, 0.90]],  # Probabilité Collision = Oui
    evidence=['AlerteCollision', 'CapaciteManoeuvre'], evidence_card=[2, 2]
)

model_collision.add_cpds(cpd_debris, cpd_radar, cpd_manoeuvre, cpd_alerte, cpd_collision)
infer_coll = VariableElimination(model_collision)

# Calcul : Probabilité de collision si on a une alerte mais une faible capacité de manœuvre
res_coll = infer_coll.query(variables=['CollisionEffective'], 
                            evidence={'AlerteCollision': 1, 'CapaciteManoeuvre': 0})
print(f"P(Collision | Alerte=Oui, Manœuvre=Impossible) = {res_coll.values[1]:.3f}")

# =============================================================================
# AFFICHAGE DES STRUCTURES (Utilisant votre fonction existante)
# =============================================================================
afficher_structure(model_collision, "RESEAU COLLISION SATELLITE")



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
    
    # Reseau 1: Polyarbre medical
    visualiser_reseau(model_collision, "ETAPE 4 ")


    
except ImportError:
    print("Bibliotheques de visualisation non disponibles.")
    print("Installation recommandee: pip install matplotlib networkx")
