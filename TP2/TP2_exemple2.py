import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

# =========================================================
# EXEMPLE 2 : CONTRÔLEUR FLOU DE VITESSE D’UN VÉHICULE
# =========================================================

# ---------- VARIABLES ----------
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')      # mètres
vitesse = ctrl.Antecedent(np.arange(0, 131, 1), 'vitesse')        # km/h
acceleration = ctrl.Consequent(np.arange(-5, 6, 1), 'acceleration')  # m/s²

# ---------- FONCTIONS DE CROYANCE ----------
# Distance
distance['proche'] = fuzz.trapmf(distance.universe, [0, 0, 20, 40])
distance['moyenne'] = fuzz.trimf(distance.universe, [30, 50, 70])
distance['loin'] = fuzz.trapmf(distance.universe, [60, 80, 100, 100])

# Vitesse
vitesse['lente'] = fuzz.trapmf(vitesse.universe, [0, 0, 30, 50])
vitesse['moyenne'] = fuzz.trimf(vitesse.universe, [40, 70, 100])
vitesse['rapide'] = fuzz.trapmf(vitesse.universe, [90, 110, 130, 130])

# Accélération
acceleration['freiner'] = fuzz.trimf(acceleration.universe, [-5, -3, -1])
acceleration['maintenir'] = fuzz.trimf(acceleration.universe, [-1, 0, 1])
acceleration['accelerer'] = fuzz.trimf(acceleration.universe, [1, 3, 5])

# =========================================================
# VISUALISATION DES FONCTIONS DE CROYANCE (SANS VALEURS)
# =========================================================

fig, axs = plt.subplots(3, 1, figsize=(8, 10))

distance.view(ax=axs[0])
axs[0].set_title("Fonctions de croyance — Distance")

vitesse.view(ax=axs[1])
axs[1].set_title("Fonctions de croyance — Vitesse")

acceleration.view(ax=axs[2])
axs[2].set_title("Fonctions de croyance — Accélération")

plt.tight_layout()
plt.show()

# =========================================================
# RÈGLES FLOUES
# =========================================================

rules = [
    ctrl.Rule(distance['proche'] & vitesse['rapide'], acceleration['freiner']),
    ctrl.Rule(distance['proche'] & vitesse['moyenne'], acceleration['freiner']),
    ctrl.Rule(distance['moyenne'] & vitesse['rapide'], acceleration['freiner']),
    ctrl.Rule(distance['moyenne'] & vitesse['moyenne'], acceleration['maintenir']),
    ctrl.Rule(distance['loin'] & vitesse['lente'], acceleration['accelerer']),
    ctrl.Rule(distance['loin'] & vitesse['moyenne'], acceleration['accelerer']),
]

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

# =========================================================
# EXEMPLE DE FUZZIFICATION (VALEURS CRISP)
# =========================================================

distance_val = 35   # m
vitesse_val = 90    # km/h

sim.input['distance'] = distance_val
sim.input['vitesse'] = vitesse_val
sim.compute()

output_acc = sim.output['acceleration']

# =========================================================
# FONCTION UTILITAIRE POUR COLORIER L’ACTIVATION
# =========================================================

def plot_activation(universe, mf, value, label, color):
    mu = fuzz.interp_membership(universe, mf, value)
    plt.plot(universe, mf, color=color, linewidth=2, label=label)
    plt.fill_between(universe, 0, np.minimum(mf, mu), alpha=0.4, color=color)
    plt.axvline(value, linestyle='--', color='k')
    return mu

# =========================================================
# FUZZIFICATION — DISTANCE
# =========================================================

plt.figure(figsize=(7,4))
plot_activation(distance.universe, distance['proche'].mf, distance_val, 'Proche', 'red')
plot_activation(distance.universe, distance['moyenne'].mf, distance_val, 'Moyenne', 'orange')
plot_activation(distance.universe, distance['loin'].mf, distance_val, 'Loin', 'green')
plt.title(f"Fuzzification Distance = {distance_val} m")
plt.xlabel("Distance (m)")
plt.ylabel("Degré d'appartenance")
plt.legend()
plt.grid()
plt.show()

# =========================================================
# FUZZIFICATION — VITESSE
# =========================================================

plt.figure(figsize=(7,4))
plot_activation(vitesse.universe, vitesse['lente'].mf, vitesse_val, 'Lente', 'green')
plot_activation(vitesse.universe, vitesse['moyenne'].mf, vitesse_val, 'Moyenne', 'orange')
plot_activation(vitesse.universe, vitesse['rapide'].mf, vitesse_val, 'Rapide', 'red')
plt.title(f"Fuzzification Vitesse = {vitesse_val} km/h")
plt.xlabel("Vitesse (km/h)")
plt.ylabel("Degré d'appartenance")
plt.legend()
plt.grid()
plt.show()

# =========================================================
# SORTIE FLOUE + DÉFLOUTAGE
# =========================================================

plt.figure(figsize=(7,4))
plot_activation(acceleration.universe, acceleration['freiner'].mf, output_acc, 'Freiner', 'red')
plot_activation(acceleration.universe, acceleration['maintenir'].mf, output_acc, 'Maintenir', 'blue')
plot_activation(acceleration.universe, acceleration['accelerer'].mf, output_acc, 'Accélérer', 'green')

plt.title(f"Sortie floue — Accélération = {output_acc:.2f} m/s²")
plt.xlabel("Accélération")
plt.ylabel("Degré d'appartenance")
plt.legend()
plt.grid()
plt.show()

print(f"Accélération finale défloutée : {output_acc:.2f} m/s²")
