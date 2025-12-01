# TP 3 - Reseaux Bayesiens avec DiscreteBayesianNetwork
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
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
# ETAPE 5 : DETECTION D'INTRUSION CYBERSECURITE
# =============================================================================
print("\n" + "="*50)
print("ETAPE 5 : DETECTION D'INTRUSION CYBERSECURITE")
print("="*50)

model_cyber = BayesianNetwork([
    ('ScanReseau', 'Intrusion'),
    ('Phishing', 'Intrusion'),
    ('VulnerabiliteServeur', 'Intrusion'),
    ('Intrusion', 'Exfiltration'),
    ('Intrusion', 'RalentissementReseau'),
    ('Exfiltration', 'AlertSIEM'),
    ('RalentissementReseau', 'AlertSIEM')
])

# CPDs
cpd_scan = TabularCPD(variable='ScanReseau', variable_card=2,
                     values=[[0.85], [0.15]])

cpd_phishing = TabularCPD(variable='Phishing', variable_card=2,
                         values=[[0.90], [0.10]])

cpd_vuln = TabularCPD(variable='VulnerabiliteServeur', variable_card=2,
                     values=[[0.80], [0.20]])

# Intrusion has 3 binary parents -> 2^3 = 8 columns
cpd_intrusion = TabularCPD(
    variable='Intrusion',
    variable_card=2,
    values=[
        # P(Intrusion=0 | parents) (no intrusion)
        [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.05],
        # P(Intrusion=1 | parents) (intrusion)
        [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.95]
    ],
    evidence=['ScanReseau', 'Phishing', 'VulnerabiliteServeur'],
    evidence_card=[2, 2, 2]
)

cpd_exfil = TabularCPD(variable='Exfiltration', variable_card=2,
                      values=[[0.95, 0.30],
                              [0.05, 0.70]],
                      evidence=['Intrusion'], evidence_card=[2])

cpd_ralent = TabularCPD(variable='RalentissementReseau', variable_card=2,
                       values=[[0.96, 0.40],
                               [0.04, 0.60]],
                       evidence=['Intrusion'], evidence_card=[2])

# AlertSIEM depends on Exfiltration and RalentissementReseau (2x2)
cpd_alert = TabularCPD(variable='AlertSIEM', variable_card=2,
                      values=[[0.98, 0.60, 0.55, 0.10],
                              [0.02, 0.40, 0.45, 0.90]],
                      evidence=['Exfiltration', 'RalentissementReseau'],
                      evidence_card=[2, 2])

model_cyber.add_cpds(cpd_scan, cpd_phishing, cpd_vuln, cpd_intrusion,
                     cpd_exfil, cpd_ralent, cpd_alert)

print("Modele cyber valide:", model_cyber.check_model())

# Inference
inference_cyber = VariableElimination(model_cyber)

# Scenario A: SIEM alert + network slowdown observed
print("\n--- CAS CYBER A: AlertSIEM=oui + RalentissementReseau=oui ---")
res_intrusion_A = inference_cyber.query(variables=['Intrusion'],
                                        evidence={'AlertSIEM': 1, 'RalentissementReseau': 1})
print(f"- P(Intrusion=oui | AlertSIEM=oui, RalentissementReseau=oui) = {res_intrusion_A.values[1]:.3f}")

# Scenario B: SIEM alert but no slowdown (possible stealth exfiltration)
print("\n--- CAS CYBER B: AlertSIEM=oui + RalentissementReseau=non ---")
res_intrusion_B = inference_cyber.query(variables=['Intrusion'],
                                        evidence={'AlertSIEM': 1, 'RalentissementReseau': 0})
print(f"- P(Intrusion=oui | AlertSIEM=oui, RalentissementReseau=non) = {res_intrusion_B.values[1]:.3f}")

# Visualisation (will use your visualiser_reseau if available)
try:
    visualiser_reseau(model_cyber, "ETAPE 5 - DETECTION D'INTRUSION CYBERSECURITE")
except Exception as e:
    print("Visualisation ETAPE 5 echouee:", e)

# Textual structure
afficher_structure(model_cyber, "4. DETECTION D'INTRUSION CYBERSECURITE")

# Add to summary variables for later printing
res_cyber_summary_A = res_intrusion_A.values[1]
res_cyber_summary_B = res_intrusion_B.values[1]

# =============================================================================
# ETAPE 6 : DIAGNOSTIC DE PANNE SATELLITE
# =============================================================================
print("\n" + "="*50)
print("ETAPE 6 : DIAGNOSTIC DE PANNE SATELLITE")
print("="*50)

model_sat = BayesianNetwork([
    ('PannePropulsion', 'DeriveOrbite'),
    ('DegradationPanneaux', 'BaisseEnergie'),
    ('BaisseEnergie', 'InstrumentsOFF'),
    ('SurchauffeThermique', 'OrientationInstable'),
    ('OrientationInstable', 'DeriveOrbite'),
    ('DeriveOrbite', 'PerteSignal'),
    ('InstrumentsOFF', 'PerteSignal')
])

# CPDs
cpd_prop = TabularCPD(variable='PannePropulsion', variable_card=2,
                     values=[[0.93], [0.07]])

cpd_pan = TabularCPD(variable='DegradationPanneaux', variable_card=2,
                    values=[[0.88], [0.12]])

cpd_therm = TabularCPD(variable='SurchauffeThermique', variable_card=2,
                      values=[[0.95], [0.05]])

cpd_energie = TabularCPD(variable='BaisseEnergie', variable_card=2,
                        values=[[0.90, 0.25],
                                [0.10, 0.75]],
                        evidence=['DegradationPanneaux'], evidence_card=[2])

cpd_inst = TabularCPD(variable='InstrumentsOFF', variable_card=2,
                     values=[[0.93, 0.30],
                             [0.07, 0.70]],
                     evidence=['BaisseEnergie'], evidence_card=[2])

cpd_orient = TabularCPD(variable='OrientationInstable', variable_card=2,
                       values=[[0.94, 0.15],
                               [0.06, 0.85]],
                       evidence=['SurchauffeThermique'], evidence_card=[2])

# DeriveOrbite depends on PannePropulsion and OrientationInstable (2x2 -> 4 cols)
cpd_orbite = TabularCPD(variable='DeriveOrbite', variable_card=2,
                       values=[
                           [0.98, 0.70, 0.65, 0.10],
                           [0.02, 0.30, 0.35, 0.90]
                       ],
                       evidence=['PannePropulsion', 'OrientationInstable'],
                       evidence_card=[2, 2])

# PerteSignal depends on DeriveOrbite and InstrumentsOFF
cpd_signal = TabularCPD(variable='PerteSignal', variable_card=2,
                       values=[
                           [0.97, 0.50, 0.45, 0.05],
                           [0.03, 0.50, 0.55, 0.95]
                       ],
                       evidence=['DeriveOrbite', 'InstrumentsOFF'],
                       evidence_card=[2, 2])

model_sat.add_cpds(cpd_prop, cpd_pan, cpd_therm, cpd_energie, cpd_inst,
                   cpd_orient, cpd_orbite, cpd_signal)

print("Modele satellite valide:", model_sat.check_model())

# Inference
inference_sat = VariableElimination(model_sat)

# Scenario 1: Perte de signal + Instruments off (critique)
print("\n--- CAS SAT A: PerteSignal=oui + InstrumentsOFF=oui ---")
res_prop_A = inference_sat.query(variables=['PannePropulsion'],
                                 evidence={'PerteSignal': 1, 'InstrumentsOFF': 1})
print(f"- P(PannePropulsion=oui | PerteSignal=oui, InstrumentsOFF=oui) = {res_prop_A.values[1]:.3f}")

# Scenario 2: Perte de signal sans Instruments off (possible orientation/orbit issue)
print("\n--- CAS SAT B: PerteSignal=oui + InstrumentsOFF=non ---")
res_prop_B = inference_sat.query(variables=['PannePropulsion'],
                                 evidence={'PerteSignal': 1, 'InstrumentsOFF': 0})
print(f"- P(PannePropulsion=oui | PerteSignal=oui, InstrumentsOFF=non) = {res_prop_B.values[1]:.3f}")

# Visualisation
try:
    visualiser_reseau(model_sat, "ETAPE 6 - DIAGNOSTIC DE PANNE SATELLITE")
except Exception as e:
    print("Visualisation ETAPE 6 echouee:", e)

# Textual structure
afficher_structure(model_sat, "5. DIAGNOSTIC SATELLITE")

# Add to summary variables
res_sat_summary_A = res_prop_A.values[1]
res_sat_summary_B = res_prop_B.values[1]

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
    visualiser_reseau(model_cyber, "ETAPE 4 ")
    
    # Reseau 2: Reseau medical complexe
    visualiser_reseau(model_sat, "ETAPE 5")
    

    
except ImportError:
    print("Bibliotheques de visualisation non disponibles.")
    print("Installation recommandee: pip install matplotlib networkx")
