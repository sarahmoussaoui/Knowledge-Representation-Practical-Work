import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

# =============================================================================
# CONTRÔLEUR FLOU : ÉVITEMENT DE COLLISION ENTRE SATELLITES
# =============================================================================

print("\n" + "=" * 70)
print("CONTRÔLEUR FLOU : ÉVITEMENT DE COLLISION ENTRE SATELLITES")
print("=========================================================")

# === VARIABLES D'ENTRÉE ===

# 1. Distance Minimale d'Approche (DMA) en mètres
dma = ctrl.Antecedent(np.arange(0, 1001, 1), 'dma')
# 2. Probabilité de Collision (Pc) en % (0 à 100)
probabilite_collision = ctrl.Antecedent(np.arange(0, 101, 1), 'probabilite_collision')

# === VARIABLE DE SORTIE ===
# Impulsion nécessaire pour la manœuvre (ex: en Newton-secondes)
manoeuvre_impulsion = ctrl.Consequent(np.arange(0, 11, 1), 'manoeuvre_impulsion')

# === FONCTIONS D’APPARTENANCE (MFs) ===

# DMA (Distance Minimale d'Approche)
dma['critique'] = fuzz.trapmf(dma.universe, [0, 0, 50, 150])       # Très proche
dma['proche'] = fuzz.trimf(dma.universe, [100, 300, 500])         # Zone de risque
dma['sure'] = fuzz.trapmf(dma.universe, [400, 700, 1000, 1000])   # Sécuritaire

# Probabilité de Collision (Pc)
probabilite_collision['faible'] = fuzz.trimf(probabilite_collision.universe, [0, 0, 20])
probabilite_collision['moyenne'] = fuzz.trimf(probabilite_collision.universe, [10, 40, 70])
probabilite_collision['elevee'] = fuzz.trimf(probabilite_collision.universe, [60, 100, 100])

# Manœuvre d'Impulsion (Sortie)
manoeuvre_impulsion['aucune'] = fuzz.trimf(manoeuvre_impulsion.universe, [0, 0, 2])
manoeuvre_impulsion['faible'] = fuzz.trimf(manoeuvre_impulsion.universe, [1, 3, 5])
manoeuvre_impulsion['moyenne'] = fuzz.trimf(manoeuvre_impulsion.universe, [4, 6, 8])
manoeuvre_impulsion['forte'] = fuzz.trimf(manoeuvre_impulsion.universe, [7, 10, 10])

# === RÈGLES DE CONTRÔLE FLOU ===
regles_collision = [
    # CAS CRITIQUES (Sécurité maximale)
    ctrl.Rule(dma['critique'] | probabilite_collision['elevee'], manoeuvre_impulsion['forte']),
    
    # RÈGLES DE HAUT RISQUE
    ctrl.Rule(dma['proche'] & probabilite_collision['elevee'], manoeuvre_impulsion['forte']),
    ctrl.Rule(dma['critique'] & probabilite_collision['moyenne'], manoeuvre_impulsion['forte']),
    
    # RÈGLES DE RISQUE MODÉRÉ
    ctrl.Rule(dma['proche'] & probabilite_collision['moyenne'], manoeuvre_impulsion['moyenne']),
    ctrl.Rule(dma['critique'] & probabilite_collision['faible'], manoeuvre_impulsion['moyenne']),
    
    # RÈGLES DE FAIBLE RISQUE
    ctrl.Rule(dma['proche'] & probabilite_collision['faible'], manoeuvre_impulsion['faible']),
    ctrl.Rule(dma['sure'] & probabilite_collision['moyenne'], manoeuvre_impulsion['faible']),
    
    # RÈGLE D'ABSENCE DE RISQUE
    ctrl.Rule(dma['sure'] & probabilite_collision['faible'], manoeuvre_impulsion['aucune'])
]

# === SYSTÈME DE CONTRÔLE ET SIMULATION ===

systeme_collision = ctrl.ControlSystem(regles_collision)
controleur_collision = ctrl.ControlSystemSimulation(systeme_collision)

# === TESTS DES SCÉNARIOS DE COLLISION ===

print("\n--- TESTS DE SCÉNARIOS DE RISQUE ---")
print("DMA (m) | Pc (%) | Impulsion de Manœuvre (N.s)")
print("-" * 55)

scenarios_collision = [
    # 1. Risque Extrême (Proche et Certain)
    (50, 90),
    # 2. Risque Modéré-Élevé (Assez proche, Pc moyenne)
    (250, 45),
    # 3. Risque faible (Sécuritaire, Pc moyenne - Surveillance)
    (700, 40),
    # 4. Risque critique (Très proche, Pc faible - Urgence DMA)
    (100, 10),
    # 5. Scénario Sécuritaire (Aucune action)
    (900, 5)
]

for d, p in scenarios_collision:
    controleur_collision.input['dma'] = d
    controleur_collision.input['probabilite_collision'] = p
    controleur_collision.compute()
    result = controleur_collision.output['manoeuvre_impulsion']

    print(f"{d:7} | {p:6}% | {result:25.1f} N.s")

# === VISUALISATION DES FONCTIONS D'APPARTENANCE ET 3D ===

# 1. Visualisation des Fonctions d'Appartenance
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
dma.view(ax=ax1)
ax1.set_title('Distance Minimale d\'Approche (DMA)')
probabilite_collision.view(ax=ax2)
ax2.set_title('Probabilité de Collision (Pc)')
manoeuvre_impulsion.view(ax=ax3)
ax3.set_title('Manœuvre d\'Impulsion (Sortie)')
ax4.set_visible(False) # Masquer le 4ème sous-plot
plt.tight_layout()
plt.show()

# 2. Visualisation de la Surface de Contrôle (3D)
fig_3d = plt.figure(figsize=(10, 10))
ax_3d = fig_3d.add_subplot(111, projection='3d')
# Correction vérifiée : appel direct de view() sur l'objet ControlSystem
systeme_collision.view(ax=ax_3d) 
ax_3d.set_title('Surface de Contrôle Flou')

plt.tight_layout() # S'assurer que le titre est bien affiché
plt.show()

# === EXEMPLE D'ANALYSE D'UNE RÈGLE SPÉCIFIQUE ===
print("\n--- ANALYSE DÉTAILLÉE DU SCÉNARIO 2 (Risque Modéré-Élevé) ---")
d, p = 250, 45
controleur_collision.input['dma'] = d
controleur_collision.input['probabilite_collision'] = p
controleur_collision.compute()

# Visualisation détaillée de la sortie floue et défloutée
manoeuvre_impulsion.view(sim=controleur_collision) 
plt.title(f"Impulsion pour DMA={d}m et Pc={p}% (Résultat Défloué: {controleur_collision.output['manoeuvre_impulsion']:.1f} N.s)")
plt.show()