import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

# =============================================================================
# CONTRÔLEUR FLOU 3 : GESTION THERMIQUE D'UN DATA CENTER (VERSION FINALE)
# =============================================================================

print("\n" + "=" * 70)
print("CONTRÔLEUR FLOU 3 : GESTION THERMIQUE D'UN DATA CENTER")
print("=" * 70)

# === VARIABLES D'ENTRÉE ===

temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
charge_cpu = ctrl.Antecedent(np.arange(0, 101, 1), 'charge_cpu')
humidite = ctrl.Antecedent(np.arange(0, 101, 1), 'humidite')

# === VARIABLE DE SORTIE ===
ventilation = ctrl.Consequent(np.arange(0, 101, 1), 'ventilation')

# === FONCTIONS D’APPARTENANCE ===

temperature['froide'] = fuzz.trimf(temperature.universe, [0, 0, 25])
temperature['normale'] = fuzz.trimf(temperature.universe, [20, 40, 60])
temperature['chaude'] = fuzz.trimf(temperature.universe, [55, 70, 85])
temperature['critique'] = fuzz.trimf(temperature.universe, [80, 100, 100])

charge_cpu['faible'] = fuzz.trimf(charge_cpu.universe, [0, 0, 40])
charge_cpu['moyenne'] = fuzz.trimf(charge_cpu.universe, [30, 50, 70])
charge_cpu['elevee'] = fuzz.trimf(charge_cpu.universe, [60, 100, 100])

humidite['basse'] = fuzz.trimf(humidite.universe, [0, 0, 35])
humidite['normale'] = fuzz.trimf(humidite.universe, [30, 50, 70])
humidite['haute'] = fuzz.trimf(humidite.universe, [60, 100, 100])

ventilation['faible'] = fuzz.trimf(ventilation.universe, [0, 0, 30])
ventilation['moyenne'] = fuzz.trimf(ventilation.universe, [20, 45, 70])
ventilation['forte'] = fuzz.trimf(ventilation.universe, [60, 75, 90])
ventilation['maximum'] = fuzz.trimf(ventilation.universe, [85, 100, 100])

# === RÈGLES ===

regles_thermiques = [

    # SURCHAUFFE
    ctrl.Rule(temperature['critique'] & charge_cpu['elevee'], ventilation['maximum']),
    ctrl.Rule(temperature['critique'] & charge_cpu['moyenne'], ventilation['forte']),
    ctrl.Rule(temperature['critique'], ventilation['maximum']),

    # CHAUX
    ctrl.Rule(temperature['chaude'] & charge_cpu['elevee'], ventilation['forte']),
    ctrl.Rule(temperature['chaude'] & charge_cpu['moyenne'], ventilation['forte']),
    ctrl.Rule(temperature['chaude'] & humidite['haute'], ventilation['forte']),

    # NORMALE
    ctrl.Rule(temperature['normale'] & charge_cpu['faible'], ventilation['faible']),
    ctrl.Rule(temperature['normale'] & charge_cpu['moyenne'], ventilation['moyenne']),
    ctrl.Rule(temperature['normale'] & humidite['haute'], ventilation['moyenne']),
    ctrl.Rule(temperature['normale'] & humidite['basse'], ventilation['faible']),

    # FROIDE
    ctrl.Rule(temperature['froide'], ventilation['faible']),

    # HUMIDITÉ
    ctrl.Rule(humidite['haute'] & charge_cpu['elevee'], ventilation['forte']),
    ctrl.Rule(humidite['basse'] & charge_cpu['faible'], ventilation['faible']),

    # RÈGLE DE COUVERTURE GÉNÉRALE
    ctrl.Rule(charge_cpu['moyenne'], ventilation['moyenne'])
]

systeme_thermique = ctrl.ControlSystem(regles_thermiques)
controleur_thermique = ctrl.ControlSystemSimulation(systeme_thermique)

# === TESTS ===

print("\n--- TESTS DU SYSTÈME THERMIQUE ---")
print("Temp | CPU | Humidité | Ventilation")
print("-" * 55)

scenarios_thermiques = [
    (85, 90, 70),
    (40, 50, 45),
    (15, 20, 30),
    (60, 80, 90),
    (70, 40, 20)
]

for temp, cpu, hum in scenarios_thermiques:
    controleur_thermique.input['temperature'] = temp
    controleur_thermique.input['charge_cpu'] = cpu
    controleur_thermique.input['humidite'] = hum
    controleur_thermique.compute()
    result = controleur_thermique.output['ventilation']

    print(f"{temp:5}°C | {cpu:3}% | {hum:8}% | {result:10.1f}%")

# === VISUALISATION ===

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
temperature.view(ax=ax1)
ax1.set_title('Température interne')
charge_cpu.view(ax=ax2)
ax2.set_title('Charge CPU')
humidite.view(ax=ax3)
ax3.set_title('Humidité')
ventilation.view(ax=ax4)
ax4.set_title('Ventilation')
plt.tight_layout()
plt.show()



# =============================================================================
# CONTRÔLEUR FLOU 4 : STABILISATION D'ALTITUDE D'UN DRONE (VERSION FINALE)
# =============================================================================

print("\n" + "=" * 70)
print("CONTRÔLEUR FLOU 4 : STABILISATION D'ALTITUDE D'UN DRONE")
print("=" * 70)

# === VARIABLES D'ENTRÉE ===

erreur_altitude = ctrl.Antecedent(np.arange(-10, 11, 1), 'erreur_altitude')
variation_altitude = ctrl.Antecedent(np.arange(-5, 6, 1), 'variation_altitude')
vent = ctrl.Antecedent(np.arange(0, 101, 1), 'vent')

# === VARIABLE DE SORTIE ===
poussee = ctrl.Consequent(np.arange(0, 101, 1), 'poussee')

# === FONCTIONS D’APPARTENANCE ===

erreur_altitude['basse'] = fuzz.trimf(erreur_altitude.universe, [-10, -5, 0])
erreur_altitude['moyenne'] = fuzz.trimf(erreur_altitude.universe, [-2, 0, 2])
erreur_altitude['haute'] = fuzz.trimf(erreur_altitude.universe, [0, 5, 10])

variation_altitude['descente_rapide'] = fuzz.trimf(variation_altitude.universe, [-5, -5, -2])
variation_altitude['descente'] = fuzz.trimf(variation_altitude.universe, [-3, -1, 0])
variation_altitude['stable'] = fuzz.trimf(variation_altitude.universe, [-1, 0, 1])
variation_altitude['montee'] = fuzz.trimf(variation_altitude.universe, [0, 1, 3])
variation_altitude['montee_rapide'] = fuzz.trimf(variation_altitude.universe, [2, 5, 5])

vent['faible'] = fuzz.trimf(vent.universe, [0, 0, 40])
vent['moyen'] = fuzz.trimf(vent.universe, [30, 50, 70])
vent['fort'] = fuzz.trimf(vent.universe, [60, 100, 100])

poussee['faible'] = fuzz.trimf(poussee.universe, [0, 0, 30])
poussee['moyenne'] = fuzz.trimf(poussee.universe, [20, 45, 70])
poussee['forte'] = fuzz.trimf(poussee.universe, [60, 75, 90])
poussee['tres_forte'] = fuzz.trimf(poussee.universe, [85, 100, 100])

# === RÈGLES ===

regles_drone = [

    # Compensation de chute
    ctrl.Rule(variation_altitude['descente_rapide'], poussee['tres_forte']),
    ctrl.Rule(variation_altitude['descente'] & erreur_altitude['haute'], poussee['forte']),
    ctrl.Rule(variation_altitude['descente'] & erreur_altitude['moyenne'], poussee['moyenne']),

    # Compensation de montée
    ctrl.Rule(variation_altitude['montee_rapide'], poussee['faible']),
    ctrl.Rule(variation_altitude['montee'] & erreur_altitude['basse'], poussee['faible']),
    ctrl.Rule(variation_altitude['montee'] & erreur_altitude['haute'], poussee['moyenne']),

    # Cas stables
    ctrl.Rule(variation_altitude['stable'] & erreur_altitude['basse'], poussee['faible']),
    ctrl.Rule(variation_altitude['stable'] & erreur_altitude['moyenne'], poussee['moyenne']),
    ctrl.Rule(variation_altitude['stable'] & erreur_altitude['haute'], poussee['forte']),

    # Effet du vent
    ctrl.Rule(vent['fort'] & erreur_altitude['haute'], poussee['tres_forte']),
    ctrl.Rule(vent['fort'] & erreur_altitude['moyenne'], poussee['forte']),
    ctrl.Rule(vent['moyen'] & variation_altitude['descente'], poussee['forte']),

    # Couverture générale
    ctrl.Rule(vent['faible'], poussee['moyenne'])
]

systeme_drone = ctrl.ControlSystem(regles_drone)
controleur_drone = ctrl.ControlSystemSimulation(systeme_drone)

# === TESTS ===

print("\n--- TESTS DU CONTRÔLEUR DRONE ---")
print("Erreur | Variation | Vent | Poussée")
print("-" * 50)

scenarios_drone = [
    (-8, -4, 80),
    (3, 0, 40),
    (7, -3, 20),
    (-2, 3, 60),
    (5, 1, 90),
]

for err, var, v in scenarios_drone:
    controleur_drone.input['erreur_altitude'] = err
    controleur_drone.input['variation_altitude'] = var
    controleur_drone.input['vent'] = v
    controleur_drone.compute()

    out = controleur_drone.output['poussee']
    print(f"{err:6}m | {var:8}m/s | {v:4}% | {out:6.1f}%")

# === VISUALISATION ===

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
erreur_altitude.view(ax=ax1)
ax1.set_title("Erreur d'altitude")
variation_altitude.view(ax=ax2)
ax2.set_title("Variation d'altitude")
vent.view(ax=ax3)
ax3.set_title("Vent")
poussee.view(ax=ax4)
ax4.set_title("Poussée moteur")
plt.tight_layout()
plt.show()
