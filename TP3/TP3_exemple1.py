# TP 3 - Reseaux Bayesiens avec DiscreteBayesianNetwork
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

def afficher_structure(model, nom):
    """Affiche la structure d'un reseau de maniere textuelle"""
    print(f"\n{nom}:")
    print("Structure des dependances:")
    for edge in model.edges():
        print(f"  {edge[0]} -> {edge[1]}")
    print(f"Total: {len(model.nodes())} noeuds, {len(model.edges())} aretes")
    print("Noeuds racines:", [node for node in model.nodes() if len(model.get_parents(node)) == 0])


# =============================================================================
# ETAPE 4 : RISQUE DE COLLISION DE SATELLITE - VERSION SIMPLE
# =============================================================================
print("\n" + "="*50)
print("ETAPE 4 : DETECTION DE RISQUE DE COLLISION DE SATELLITE (POLY-ARBRE)")
print("="*50)

# Structure simple (poly-arbre)
# - DensiteDebris et PrecisionRadar influencent l'AlerteCollision
# - AlerteCollision et CapaciteManoeuvre influencent la CollisionEffective
model_collision_simple = BayesianNetwork([
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

model_collision_simple.add_cpds(cpd_debris, cpd_radar, cpd_manoeuvre, cpd_alerte, cpd_collision)
infer_coll_simple = VariableElimination(model_collision_simple)

# Calcul : Probabilité de collision si on a une alerte mais une faible capacité de manœuvre
res_coll_simple = infer_coll_simple.query(variables=['CollisionEffective'], 
                                         evidence={'AlerteCollision': 1, 'CapaciteManoeuvre': 0})
print(f"P(Collision | Alerte=Oui, Manœuvre=Impossible) = {res_coll_simple.values[1]:.3f}")

# Afficher la structure
afficher_structure(model_collision_simple, "RESEAU COLLISION SATELLITE SIMPLE")

# =============================================================================
# ETAPE 5 : SYSTEME AVANCE DE GESTION DE TRAFIC SPATIAL
# =============================================================================
print("\n" + "="*70)
print("ETAPE 5 : SYSTEME AVANCE DE GESTION DE TRAFIC SPATIAL")
print("="*70)
print("Graphe à connexions multiples avec structure complexe")

# Création d'un réseau bayésien COMPLEXE pour la gestion de trafic spatial
# Ce réseau présente plusieurs caractéristiques intéressantes:
# - Plusieurs chemins entre certains nœuds (connexions multiples)
# - Structure en diamant
# - V-structures
# - Connexions croisées
# - Boucles de causalité indirectes

model_traffic_complexe = BayesianNetwork([
    # Couche 1: Facteurs environnementaux (nœuds racines)
    ('ActiviteSolaire', 'PerturbationAtmospherique'),
    ('ActiviteSolaire', 'DegradationInstruments'),
    ('MeteorologieEspace', 'PerturbationAtmospherique'),
    
    # Couche 2: État des systèmes avec connexions croisées
    ('PerturbationAtmospherique', 'PrecisionRadar'),
    ('DegradationInstruments', 'PrecisionRadar'),
    ('PerturbationAtmospherique', 'QualiteCommunication'),
    ('DegradationInstruments', 'QualiteCommunication'),
    
    # Couche 3: Sources de données avec structure en diamant
    ('PrecisionRadar', 'DetectionDebris'),
    ('QualiteCommunication', 'DetectionDebris'),
    ('PrecisionRadar', 'TrackingPrecis'),
    ('QualiteCommunication', 'TrackingPrecis'),
    
    # Couche 4: Analyse de risque avec v-structure
    ('DetectionDebris', 'RisqueCollision'),
    ('TrackingPrecis', 'RisqueCollision'),
    ('DetectionDebris', 'ConfiancePrediction'),
    ('TrackingPrecis', 'ConfiancePrediction'),
    
    # Couche 5: Décision et action
    ('RisqueCollision', 'DecisionManoeuvre'),
    ('ConfiancePrediction', 'DecisionManoeuvre'),
    ('RisqueCollision', 'AlertePriorite'),
    ('ConfiancePrediction', 'AlertePriorite'),
    
    # Couche 6: Résultats avec convergence multiple
    ('DecisionManoeuvre', 'CollisionEvitee'),
    ('AlertePriorite', 'CollisionEvitee'),
    ('DecisionManoeuvre', 'CarburantConsomme'),
    ('AlertePriorite', 'CarburantConsomme'),
    
    # Dernière couche: Évaluation globale
    ('CollisionEvitee', 'MissionReussie'),
    ('CarburantConsomme', 'MissionReussie')
])

# CPDs pour les nœuds racines (probabilités a priori)
cpd_activite_solaire = TabularCPD(variable='ActiviteSolaire', variable_card=3,
                                 values=[[0.6], [0.3], [0.1]])  # Faible, Moyenne, Forte

cpd_meteo_espace = TabularCPD(variable='MeteorologieEspace', variable_card=2,
                             values=[[0.8], [0.2]])  # Calme, Perturbée

# CPDs pour les nœuds intermédiaires
# PerturbationAtmosphérique dépend de ActiviteSolaire et MeteorologieEspace (3x2=6 combinaisons)
cpd_perturb = TabularCPD(
    variable='PerturbationAtmospherique', variable_card=3,  # Faible, Moyenne, Forte
    values=[
        # Faible perturbation
        [0.90, 0.70, 0.50, 0.30, 0.20, 0.10],
        # Perturbation moyenne
        [0.08, 0.20, 0.30, 0.40, 0.30, 0.30],
        # Forte perturbation
        [0.02, 0.10, 0.20, 0.30, 0.50, 0.60]
    ],
    evidence=['ActiviteSolaire', 'MeteorologieEspace'], 
    evidence_card=[3, 2]
)

# DegradationInstruments dépend de ActiviteSolaire
cpd_degrad = TabularCPD(
    variable='DegradationInstruments', variable_card=3,  # Faible, Moyenne, Forte
    values=[
        [0.85, 0.60, 0.30],  # Faible dégradation
        [0.10, 0.25, 0.40],  # Dégradation moyenne
        [0.05, 0.15, 0.30]   # Forte dégradation
    ],
    evidence=['ActiviteSolaire'], 
    evidence_card=[3]
)

# PrecisionRadar dépend de PerturbationAtmosphérique et DegradationInstruments (3x3=9 combinaisons)
cpd_precision = TabularCPD(
    variable='PrecisionRadar', variable_card=3,  # Basse, Moyenne, Haute
    values=[
        # Basse précision
        [0.95, 0.80, 0.70, 0.85, 0.65, 0.55, 0.75, 0.50, 0.40],
        # Précision moyenne
        [0.04, 0.15, 0.20, 0.12, 0.25, 0.30, 0.20, 0.35, 0.35],
        # Haute précision
        [0.01, 0.05, 0.10, 0.03, 0.10, 0.15, 0.05, 0.15, 0.25]
    ],
    evidence=['PerturbationAtmospherique', 'DegradationInstruments'], 
    evidence_card=[3, 3]
)

# QualiteCommunication - même structure que PrecisionRadar
cpd_com = TabularCPD(
    variable='QualiteCommunication', variable_card=3,  # Mauvaise, Moyenne, Bonne
    values=[
        # Mauvaise qualité
        [0.90, 0.75, 0.65, 0.80, 0.60, 0.50, 0.70, 0.45, 0.35],
        # Qualité moyenne
        [0.08, 0.20, 0.25, 0.15, 0.30, 0.35, 0.25, 0.40, 0.40],
        # Bonne qualité
        [0.02, 0.05, 0.10, 0.05, 0.10, 0.15, 0.05, 0.15, 0.25]
    ],
    evidence=['PerturbationAtmospherique', 'DegradationInstruments'], 
    evidence_card=[3, 3]
)

# DetectionDebris dépend de PrecisionRadar et QualiteCommunication (3x3=9 combinaisons)
cpd_detection = TabularCPD(
    variable='DetectionDebris', variable_card=3,  # Faible, Moyenne, Bonne
    values=[
        # Faible détection
        [0.80, 0.60, 0.40, 0.65, 0.45, 0.25, 0.50, 0.30, 0.15],
        # Détection moyenne
        [0.15, 0.30, 0.40, 0.25, 0.40, 0.45, 0.35, 0.45, 0.40],
        # Bonne détection
        [0.05, 0.10, 0.20, 0.10, 0.15, 0.30, 0.15, 0.25, 0.45]
    ],
    evidence=['PrecisionRadar', 'QualiteCommunication'], 
    evidence_card=[3, 3]
)

# TrackingPrecis - même structure
cpd_tracking = TabularCPD(
    variable='TrackingPrecis', variable_card=3,  # Imprécis, Moyen, Précis
    values=[
        # Imprécis
        [0.85, 0.65, 0.45, 0.70, 0.50, 0.30, 0.55, 0.35, 0.20],
        # Moyen
        [0.10, 0.25, 0.35, 0.20, 0.35, 0.40, 0.30, 0.40, 0.35],
        # Précis
        [0.05, 0.10, 0.20, 0.10, 0.15, 0.30, 0.15, 0.25, 0.45]
    ],
    evidence=['PrecisionRadar', 'QualiteCommunication'], 
    evidence_card=[3, 3]
)

# RisqueCollision dépend de DetectionDebris et TrackingPrecis (3x3=9 combinaisons)
cpd_risque = TabularCPD(
    variable='RisqueCollision', variable_card=4,  # Nul, Faible, Moyen, Élevé
    values=[
        # Nul
        [0.70, 0.50, 0.30, 0.45, 0.25, 0.15, 0.20, 0.10, 0.05],
        # Faible
        [0.20, 0.30, 0.35, 0.35, 0.40, 0.30, 0.30, 0.25, 0.20],
        # Moyen
        [0.08, 0.15, 0.25, 0.15, 0.25, 0.35, 0.35, 0.40, 0.35],
        # Élevé
        [0.02, 0.05, 0.10, 0.05, 0.10, 0.20, 0.15, 0.25, 0.40]
    ],
    evidence=['DetectionDebris', 'TrackingPrecis'], 
    evidence_card=[3, 3]
)

# ConfiancePrediction - même structure
cpd_confiance = TabularCPD(
    variable='ConfiancePrediction', variable_card=3,  # Faible, Moyenne, Forte
    values=[
        # Faible
        [0.60, 0.40, 0.20, 0.35, 0.25, 0.15, 0.25, 0.15, 0.08],
        # Moyenne
        [0.30, 0.40, 0.45, 0.45, 0.45, 0.40, 0.40, 0.40, 0.32],
        # Forte
        [0.10, 0.20, 0.35, 0.20, 0.30, 0.45, 0.35, 0.45, 0.60]
    ],
    evidence=['DetectionDebris', 'TrackingPrecis'], 
    evidence_card=[3, 3]
)

# DecisionManoeuvre dépend de RisqueCollision et ConfiancePrediction (4x3=12 combinaisons)
cpd_decision = TabularCPD(
    variable='DecisionManoeuvre', variable_card=3,  # Non, Mineure, Majeure
    values=[
        # Pas de manœuvre
        [0.95, 0.80, 0.60, 0.85, 0.65, 0.45, 0.70, 0.50, 0.30, 0.50, 0.30, 0.15],
        # Manœuvre mineure
        [0.04, 0.15, 0.30, 0.10, 0.25, 0.40, 0.20, 0.35, 0.50, 0.35, 0.45, 0.45],
        # Manœuvre majeure
        [0.01, 0.05, 0.10, 0.05, 0.10, 0.15, 0.10, 0.15, 0.20, 0.15, 0.25, 0.40]
    ],
    evidence=['RisqueCollision', 'ConfiancePrediction'], 
    evidence_card=[4, 3]
)

# AlertePriorite - même structure
cpd_alerte_complexe = TabularCPD(
    variable='AlertePriorite', variable_card=3,  # Basse, Moyenne, Haute
    values=[
        # Basse priorité
        [0.90, 0.70, 0.50, 0.75, 0.55, 0.35, 0.60, 0.40, 0.25, 0.40, 0.25, 0.15],
        # Priorité moyenne
        [0.08, 0.20, 0.35, 0.20, 0.35, 0.45, 0.30, 0.40, 0.50, 0.45, 0.50, 0.45],
        # Haute priorité
        [0.02, 0.10, 0.15, 0.05, 0.10, 0.20, 0.10, 0.20, 0.25, 0.15, 0.25, 0.40]
    ],
    evidence=['RisqueCollision', 'ConfiancePrediction'], 
    evidence_card=[4, 3]
)

# CollisionEvitee dépend de DecisionManoeuvre et AlertePriorite (3x3=9 combinaisons)
cpd_collision_evitee = TabularCPD(
    variable='CollisionEvitee', variable_card=2,  # Non, Oui
    values=[
        # Collision non évitée
        [0.10, 0.15, 0.20, 0.08, 0.12, 0.18, 0.05, 0.08, 0.12],
        # Collision évitée
        [0.90, 0.85, 0.80, 0.92, 0.88, 0.82, 0.95, 0.92, 0.88]
    ],
    evidence=['DecisionManoeuvre', 'AlertePriorite'], 
    evidence_card=[3, 3]
)

# CarburantConsomme - même structure
cpd_carburant = TabularCPD(
    variable='CarburantConsomme', variable_card=3,  # Faible, Moyen, Élevé
    values=[
        # Faible consommation
        [0.95, 0.80, 0.60, 0.85, 0.65, 0.45, 0.70, 0.50, 0.30],
        # Consommation moyenne
        [0.04, 0.15, 0.30, 0.12, 0.25, 0.40, 0.25, 0.35, 0.45],
        # Consommation élevée
        [0.01, 0.05, 0.10, 0.03, 0.10, 0.15, 0.05, 0.15, 0.25]
    ],
    evidence=['DecisionManoeuvre', 'AlertePriorite'], 
    evidence_card=[3, 3]
)

# MissionReussie dépend de CollisionEvitee et CarburantConsomme (2x3=6 combinaisons)
cpd_mission = TabularCPD(
    variable='MissionReussie', variable_card=2,  # Échec, Réussite
    values=[
        # Mission échouée
        [0.30, 0.40, 0.50, 0.10, 0.15, 0.25],
        # Mission réussie
        [0.70, 0.60, 0.50, 0.90, 0.85, 0.75]
    ],
    evidence=['CollisionEvitee', 'CarburantConsomme'], 
    evidence_card=[2, 3]
)

# Ajout de toutes les CPDs au modèle
model_traffic_complexe.add_cpds(
    cpd_activite_solaire, cpd_meteo_espace, cpd_perturb, cpd_degrad,
    cpd_precision, cpd_com, cpd_detection, cpd_tracking,
    cpd_risque, cpd_confiance, cpd_decision, cpd_alerte_complexe,
    cpd_collision_evitee, cpd_carburant, cpd_mission
)

# Vérification du modèle
print("\nValidation du modèle complexe:", model_traffic_complexe.check_model())

# Inférence sur le modèle complexe
infer_complexe = VariableElimination(model_traffic_complexe)

# SCENARIO 1: Activité solaire forte et météo spatiale perturbée
print("\n" + "="*70)
print("SCENARIO 1: Conditions spatiales extrêmes")
print("="*70)
print("ActiviteSolaire = Forte (2), MeteorologieEspace = Perturbée (1)")
print("→ Analyse de l'impact sur la mission...")

# Calcul de plusieurs probabilités conditionnelles
res_mission_sc1 = infer_complexe.query(
    variables=['MissionReussie'],
    evidence={'ActiviteSolaire': 2, 'MeteorologieEspace': 1}
)
print(f"P(MissionReussie | Conditions extrêmes) = {res_mission_sc1.values[1]:.3f}")

# Impact sur la précision radar
res_precision_sc1 = infer_complexe.query(
    variables=['PrecisionRadar'],
    evidence={'ActiviteSolaire': 2, 'MeteorologieEspace': 1}
)
print(f"Distribution PrecisionRadar: Basse={res_precision_sc1.values[0]:.3f}, "
      f"Moyenne={res_precision_sc1.values[1]:.3f}, Haute={res_precision_sc1.values[2]:.3f}")

# SCENARIO 2: Bonnes conditions mais faible confiance
print("\n" + "="*70)
print("SCENARIO 2: Conditions favorables avec incertitude")
print("="*70)
print("ActiviteSolaire = Faible (0), MeteorologieEspace = Calme (0)")
print("ConfiancePrediction = Faible (0)")
print("→ Analyse du besoin de manœuvre...")

res_manoeuvre_sc2 = infer_complexe.query(
    variables=['DecisionManoeuvre'],
    evidence={
        'ActiviteSolaire': 0,
        'MeteorologieEspace': 0,
        'ConfiancePrediction': 0
    }
)
print(f"Decision Manœuvre: Non={res_manoeuvre_sc2.values[0]:.3f}, "
      f"Mineure={res_manoeuvre_sc2.values[1]:.3f}, Majeure={res_manoeuvre_sc2.values[2]:.3f}")

# SCENARIO 3: Diagnostic après collision évitée
print("\n" + "="*70)
print("SCENARIO 3: Diagnostic post-manœuvre")
print("="*70)
print("CollisionEvitee = Oui (1), CarburantConsomme = Élevé (2)")
print("→ Analyse de la réussite globale...")

res_mission_sc3 = infer_complexe.query(
    variables=['MissionReussie'],
    evidence={'CollisionEvitee': 1, 'CarburantConsomme': 2}
)
print(f"P(MissionReussie | Collision évitée mais carburant élevé) = {res_mission_sc3.values[1]:.3f}")

# Afficher la structure du réseau complexe
afficher_structure(model_traffic_complexe, "RESEAU COMPLEXE GESTION TRAFIC SPATIAL")

# =============================================================================
# COMPARAISON DES DEUX RESEAUX
# =============================================================================
print("\n" + "="*70)
print("COMPARAISON DES DEUX ARCHITECTURES DE RESEAUX")
print("="*70)

print("\n1. RESEAU SIMPLE (ETAPE 4):")
print("   - Type: Poly-arbre")
print("   - Noeuds: 5")
print("   - Arêtes: 4")
print("   - Complexité: Faible")
print("   - Objectif: Estimation rapide du risque de collision")

print("\n2. RESEAU COMPLEXE (ETAPE 5):")
print("   - Type: Graphe à connexions multiples")
print("   - Noeuds: 16")
print("   - Arêtes: 21")
print("   - Complexité: Élevée")
print("   - Objectif: Gestion complète du trafic spatial")
print("   - Caractéristiques spéciales:")
print("     • Structure en diamant")
print("     • V-structures")
print("     • Connexions croisées")
print("     • Plusieurs chemins alternatifs")

# =============================================================================
# VISUALISATION DES RESEAUX BAYESIENS
# =============================================================================
print("\n" + "="*70)
print("VISUALISATION DES RESEAUX")
print("="*70)

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    
    def visualiser_reseau(model, titre, layout='spring'):
        """Visualise un reseau bayesien avec matplotlib"""
        plt.figure(figsize=(12, 10))
        
        # Creer un graphe oriente
        G = nx.DiGraph()
        G.add_edges_from(model.edges())
        
        # Choisir la disposition
        if layout == 'spring':
            pos = nx.spring_layout(G, k=3, iterations=100)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'shell':
            pos = nx.shell_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Determiner la couche de chaque nœud pour le coloriage
        couleurs = []
        for node in G.nodes():
            parents = model.get_parents(node)
            if len(parents) == 0:
                couleurs.append('lightgreen')  # Nœuds racines
            elif len(model.get_children(node)) == 0:
                couleurs.append('lightcoral')  # Nœuds feuilles
            else:
                couleurs.append('lightblue')   # Nœuds intermédiaires
        
        # Dessiner les nœuds
        nx.draw_networkx_nodes(G, pos, 
                              node_size=1800, 
                              node_color=couleurs,
                              alpha=0.9,
                              edgecolors='black',
                              linewidths=1.5)
        
        # Dessiner les arêtes
        nx.draw_networkx_edges(G, pos,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=25,
                              arrowstyle='->',
                              width=2,
                              alpha=0.7)
        
        # Dessiner les labels
        nx.draw_networkx_labels(G, pos, 
                               font_size=9, 
                               font_weight='bold')
        
        plt.title(titre, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Afficher les informations du reseau
        print(f"\n{titre}:")
        print(f"Nombre de noeuds: {len(G.nodes())}")
        print(f"Nombre d'aretes: {len(G.edges())}")
        print(f"Densité: {len(G.edges()) / (len(G.nodes()) * (len(G.nodes()) - 1)):.3f}")
        
        # Compter les connexions multiples
        connexions_multiples = 0
        for node in G.nodes():
            parents = model.get_parents(node)
            children = model.get_children(node)
            if len(parents) > 1 or len(children) > 1:
                connexions_multiples += 1
        
        print(f"Noeuds avec connexions multiples: {connexions_multiples}")
        print("-" * 50)
    
    # Visualiser les deux reseaux
    print("\nGeneration des visualisations...")
    
    # Reseau 1: Simple (Etape 4)
    visualiser_reseau(model_collision_simple, "ETAPE 4 - RESEAU SIMPLE DE COLLISION", layout='shell')
    
    # Reseau 2: Complexe (Etape 5)
    visualiser_reseau(model_traffic_complexe, "ETAPE 5 - RESEAU COMPLEXE A CONNEXIONS MULTIPLES", layout='spring')
    
    # =============================================================================
    # ANALYSE DES RESULTATS
    # =============================================================================
    print("\n" + "="*70)
    print("ANALYSE DES RESULTATS")
    print("="*70)
    
    print("\nRESEAU SIMPLE (Etape 4):")
    print(f"- Probabilité de collision avec alerte mais sans manœuvre: {res_coll_simple.values[1]:.1%}")
    print("- Avantages: Calcul rapide, interprétation facile")
    print("- Limites: Modélisation simplifiée, pas d'intégration de facteurs secondaires")
    
    print("\nRESEAU COMPLEXE (Etape 5):")
    print(f"- Probabilité de mission réussie en conditions extrêmes: {res_mission_sc1.values[1]:.1%}")
    print(f"- Décision de manœuvre majeure en cas d'incertitude: {res_manoeuvre_sc2.values[2]:.1%}")
    print(f"- Mission réussie malgré consommation élevée: {res_mission_sc3.values[1]:.1%}")
    print("- Avantages: Modélisation réaliste, prise en compte multiple")
    print("- Limites: Complexité de calcul, besoin de nombreuses données")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("Les réseaux bayésiens à connexions multiples permettent:")
    print("1. Modéliser des systèmes complexes avec interdépendances")
    print("2. Prendre en compte plusieurs facteurs simultanément")
    print("3. Faire des inférences même avec informations partielles")
    print("4. Identifier les points critiques du système")
    print("\nApplication spatiale: Optimisation des manœuvres d'évitement")
    print("tout en considérant les contraintes opérationnelles.")
    
except ImportError as e:
    print("Bibliotheques de visualisation non disponibles.")
    print("Installation recommandee: pip install matplotlib networkx")
    
    # Afficher quand meme les résultats sans visualisation
    print("\n" + "="*70)
    print("RESUMÉ DES RÉSULTATS")
    print("="*70)
    
    print("\nRESEAU SIMPLE (Etape 4):")
    print(f"- Probabilité de collision: {res_coll_simple.values[1]:.1%}")
    
    print("\nRESEAU COMPLEXE (Etape 5):")
    print(f"- Mission réussie (conditions extrêmes): {res_mission_sc1.values[1]:.1%}")
    print(f"- Manœuvre majeure (incertitude): {res_manoeuvre_sc2.values[2]:.1%}")
    print(f"- Mission réussie (carburant élevé): {res_mission_sc3.values[1]:.1%}")