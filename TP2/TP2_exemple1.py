import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

# =============================================================================
# CONTRÔLEUR FLOU : ÉVITEMENT DE COLLISION ENTRE SATELLITES
# =============================================================================

# === VARIABLES D'ENTRÉE ===
dma = ctrl.Antecedent(np.arange(0, 1001, 1), 'dma') # Distance minimale d'approche en mètres 0 à 1000 m avec pas de 1 m
probabilite_collision = ctrl.Antecedent(np.arange(0, 101, 1), 'probabilite_collision') # Probabilité de collision en pourcentage 0 à 100 % avec pas de 1 %

# === VARIABLE DE SORTIE ===
manoeuvre_impulsion = ctrl.Consequent(np.arange(0, 11, 1), 'manoeuvre_impulsion') # Impulsion de manœuvre en N.s 0 à 10 N.s avec pas de 1 N.s

# === FONCTIONS D’APPARTENANCE ===

dma['critique'] = fuzz.trapmf(dma.universe, [0, 0, 50, 150])
dma['proche']   = fuzz.trimf(dma.universe, [100, 300, 500])
dma['sure']     = fuzz.trapmf(dma.universe, [400, 700, 1000, 1000])

probabilite_collision['faible']  = fuzz.trimf(probabilite_collision.universe, [0, 0, 20])
probabilite_collision['moyenne'] = fuzz.trimf(probabilite_collision.universe, [10, 40, 70])
probabilite_collision['elevee']  = fuzz.trimf(probabilite_collision.universe, [60, 100, 100])

manoeuvre_impulsion['aucune']  = fuzz.trimf(manoeuvre_impulsion.universe, [0, 0, 2])
manoeuvre_impulsion['faible']  = fuzz.trimf(manoeuvre_impulsion.universe, [1, 3, 5])
manoeuvre_impulsion['moyenne'] = fuzz.trimf(manoeuvre_impulsion.universe, [4, 6, 8])
manoeuvre_impulsion['forte']   = fuzz.trimf(manoeuvre_impulsion.universe, [7, 10, 10])

# =============================================================================
# GRAPHES DES FONCTIONS DE CROYANCE (SANS VALEUR CRISP)
# =============================================================================

def plot_membership_functions(var, title):
    plt.figure(figsize=(7,4))
    for label in var.terms:
        plt.plot(var.universe, var[label].mf, linewidth=2, label=label)
    plt.title(title)
    plt.xlabel(var.label)
    plt.ylabel("Degré d'appartenance")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_membership_functions(dma, "Fonctions de croyance – DMA")
plot_membership_functions(probabilite_collision, "Fonctions de croyance – Probabilité de collision")
plot_membership_functions(manoeuvre_impulsion, "Fonctions de croyance – Impulsion de manœuvre")

# =============================================================================
# RÈGLES FLOUES
# =============================================================================

regles_collision = [
    ctrl.Rule(dma['critique'] | probabilite_collision['elevee'], manoeuvre_impulsion['forte']), # Si DMA critique OU probabilité élevée, alors impulsion forte
    ctrl.Rule(dma['proche'] & probabilite_collision['elevee'], manoeuvre_impulsion['forte']), # Si DMA proche ET probabilité élevée, alors impulsion forte
    ctrl.Rule(dma['critique'] & probabilite_collision['moyenne'], manoeuvre_impulsion['forte']),
    ctrl.Rule(dma['proche'] & probabilite_collision['moyenne'], manoeuvre_impulsion['moyenne']),
    ctrl.Rule(dma['critique'] & probabilite_collision['faible'], manoeuvre_impulsion['moyenne']),
    ctrl.Rule(dma['proche'] & probabilite_collision['faible'], manoeuvre_impulsion['faible']),
    ctrl.Rule(dma['sure'] & probabilite_collision['moyenne'], manoeuvre_impulsion['faible']),
    ctrl.Rule(dma['sure'] & probabilite_collision['faible'], manoeuvre_impulsion['aucune'])
]

systeme_collision = ctrl.ControlSystem(regles_collision) # Création du système de contrôle flou
controleur_collision = ctrl.ControlSystemSimulation(systeme_collision) # Simulation du contrôleur flou

# =============================================================================
# FONCTION POUR AFFICHER ACTIVATION (ZONE COLORÉE)
# =============================================================================

def plot_membership_with_activation(universe, mf, value, label, color):
    mu = fuzz.interp_membership(universe, mf, value) # Degré d'appartenance mu pour la valeur donnée
    plt.plot(universe, mf, color=color, linewidth=2, label=label)
    plt.fill_between(universe, 0, np.minimum(mf, mu), color=color, alpha=0.4)
    plt.axvline(value, linestyle='--', color='k')
    plt.text(value, mu + 0.05, f"μ={mu:.2f}", ha='center')
    return mu

# =============================================================================
# EXEMPLE DE SCÉNARIO AVEC ACTIVATION
# =============================================================================

dma_value = 150
pc_value = 10

controleur_collision.input['dma'] = dma_value
controleur_collision.input['probabilite_collision'] = pc_value
controleur_collision.compute()

output_value = controleur_collision.output['manoeuvre_impulsion']

# --- DMA ---
plt.figure(figsize=(7,4))
plot_membership_with_activation(dma.universe, dma['critique'].mf, dma_value, 'Critique', 'red')
plot_membership_with_activation(dma.universe, dma['proche'].mf, dma_value, 'Proche', 'orange')
plot_membership_with_activation(dma.universe, dma['sure'].mf, dma_value, 'Sûre', 'green')
plt.title(f"Fuzzification DMA = {dma_value} m")
plt.xlabel("DMA (m)")
plt.ylabel("Degré d'appartenance")
plt.legend()
plt.grid(True)
plt.show()

# --- Probabilité de collision ---
plt.figure(figsize=(7,4))
plot_membership_with_activation(probabilite_collision.universe, probabilite_collision['faible'].mf, pc_value, 'Faible', 'green')
plot_membership_with_activation(probabilite_collision.universe, probabilite_collision['moyenne'].mf, pc_value, 'Moyenne', 'orange')
plot_membership_with_activation(probabilite_collision.universe, probabilite_collision['elevee'].mf, pc_value, 'Élevée', 'red')
plt.title(f"Fuzzification Pc = {pc_value} %")
plt.xlabel("Pc (%)")
plt.ylabel("Degré d'appartenance")
plt.legend()
plt.grid(True)
plt.show()

# --- Sortie ---
plt.figure(figsize=(7,4))
plot_membership_with_activation(manoeuvre_impulsion.universe, manoeuvre_impulsion['aucune'].mf, output_value, 'Aucune', 'blue')
plot_membership_with_activation(manoeuvre_impulsion.universe, manoeuvre_impulsion['faible'].mf, output_value, 'Faible', 'green')
plot_membership_with_activation(manoeuvre_impulsion.universe, manoeuvre_impulsion['moyenne'].mf, output_value, 'Moyenne', 'orange')
plot_membership_with_activation(manoeuvre_impulsion.universe, manoeuvre_impulsion['forte'].mf, output_value, 'Forte', 'red')
plt.title(f"Sortie floue – Impulsion = {output_value:.2f} N.s")
plt.xlabel("Impulsion")
plt.ylabel("Degré d'appartenance")
plt.legend()
plt.grid(True)
plt.show()
