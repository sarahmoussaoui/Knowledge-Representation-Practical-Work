import numpy as np
from itertools import chain, combinations

class DempsterShafer:
    def __init__(self, frame_of_discernment):
        self.theta = frame_of_discernment
        self.elements = list(self.powerset(self.theta))
        self.mass = {frozenset(elem): 0.0 for elem in self.elements}
        self.mass[frozenset()] = 0.0
    
    def powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    def set_mass(self, subset, value):
        self.mass[frozenset(subset)] = value
    
    def belief(self, hypothesis):
        hyp_set = frozenset(hypothesis)
        belief_value = 0.0
        for subset in self.mass:
            if subset.issubset(hyp_set) and subset != frozenset():
                belief_value += self.mass[subset]
        return belief_value
    
    def plausibility(self, hypothesis):
        hyp_set = frozenset(hypothesis)
        pl_value = 0.0
        for subset in self.mass:
            if subset.intersection(hyp_set) != frozenset():
                pl_value += self.mass[subset]
        return pl_value
    
    def doubt(self, hypothesis):
        return 1 - self.plausibility(hypothesis)
    
    def confidence_interval(self, hypothesis):
        return [self.belief(hypothesis), self.plausibility(hypothesis)]
    
    def normalize(self):
        total = sum(self.mass.values())
        if total > 0:
            for key in self.mass:
                self.mass[key] /= total
    
    def combine(self, other):
        combined = DempsterShafer(self.theta)
        
        K = 0
        for A in self.mass:
            for B in other.mass:
                if A.intersection(B) == frozenset():
                    K += self.mass[A] * other.mass[B]
        
        if K < 1:
            for A in self.mass:
                for B in other.mass:
                    intersection = A.intersection(B)
                    if intersection != frozenset():
                        combined.mass[intersection] += (self.mass[A] * other.mass[B]) / (1 - K)
        
        return combined
    
    def print_analysis(self, title):
        print("\n" + "="*60)
        print(f"ANALYSE DETAILLEE : {title}")
        print("="*60)
        
        print("\nMASSES DE CROYANCE :")
        for key, value in sorted(self.mass.items(), key=lambda x: -x[1]):
            if value > 0.001:
                print(f"  m{set(key)} = {value:.4f}")
        
        print("\nDEGRES DE CROYANCE ET PLAUSIBILITE :")
        elements = list(self.theta)
        for elem in elements:
            bel = self.belief([elem])
            pl = self.plausibility([elem])
            interval = self.confidence_interval([elem])
            print(f"  {elem:15}: Bel = {bel:.4f}, Pl = {pl:.4f}, Intervalle = [{interval[0]:.4f}, {interval[1]:.4f}]")
        
        best = max(elements, key=lambda x: self.belief([x]))
        print(f"\nCONCLUSION : {best} (Croyance la plus elevee = {self.belief([best]):.4f})")


def calculer_metriques(ds, nom_scenario):
    """Calcule des metriques avancees pour l'analyse"""
    print(f"\nMETRIQUES AVANCEES - {nom_scenario}")
    
    incertitude_totale = ds.mass[frozenset(ds.theta)]
    conflit = 1 - sum(ds.mass.values())
    
    print(f"- Incertitude totale : {incertitude_totale:.4f}")
    print(f"- Degre de conflit : {conflit:.4f}")
    
    # Entropie de la distribution
    entropy = -sum(m * np.log(m) if m > 0 else 0 for m in ds.mass.values())
    print(f"- Entropie de la distribution : {entropy:.4f}")
    
    # Clarte de la decision (difference entre 1er et 2eme)
    elements = list(ds.theta)
    croyances = [ds.belief([elem]) for elem in elements]
    croyances_triees = sorted(croyances, reverse=True)
    
    if len(croyances_triees) > 1:
        ecart_decision = croyances_triees[0] - croyances_triees[1]
        print(f"- Ecart de decision (1er-2eme) : {ecart_decision:.4f}")
    
    return incertitude_totale, conflit



# =============================================================================
# EXEMPLE 1 - SYSTÈME EXPERT DE DÉTECTION DE CYBERATTAQUES INTELLIGENT
# =============================================================================

print("EXEMPLE 1 - SYSTÈME EXPERT DE CYBERSÉCURITÉ AVANCÉ")
print("Cadre : Détection multi-couches de cybermenaces complexes")

# Cadre de discernement : types de cybermenaces
cyber_menaces = [
    'APT_Advanced', 
    'Ransomware', 
    'DDoS', 
    'Insider_Threat', 
    'ZeroDay_Exploit',
    'Credential_Theft'
]

# Source 1 : Analyse comportementale réseau (IA)
comportement_reseau = DempsterShafer(cyber_menaces)
comportement_reseau.set_mass(['APT_Advanced', 'ZeroDay_Exploit'], 0.25)
comportement_reseau.set_mass(['DDoS', 'APT_Advanced'], 0.20)
comportement_reseau.set_mass(['Credential_Theft'], 0.15)
comportement_reseau.set_mass(cyber_menaces, 0.40)  # Incertitude élevée

# Source 2 : Détection endpoint avancée
endpoint_detection = DempsterShafer(cyber_menaces)
endpoint_detection.set_mass(['Ransomware', 'Insider_Threat'], 0.30)
endpoint_detection.set_mass(['ZeroDay_Exploit'], 0.25)
endpoint_detection.set_mass(['APT_Advanced'], 0.15)
endpoint_detection.set_mass(cyber_menaces, 0.30)

# Source 3 : Intelligence des menaces externe
threat_intelligence = DempsterShafer(cyber_menaces)
threat_intelligence.set_mass(['APT_Advanced'], 0.35)
threat_intelligence.set_mass(['DDoS'], 0.20)
threat_intelligence.set_mass(['ZeroDay_Exploit', 'Ransomware'], 0.25)
threat_intelligence.set_mass(cyber_menaces, 0.20)

# Source 4 : Analyse des logs SIEM
siem_analysis = DempsterShafer(cyber_menaces)
siem_analysis.set_mass(['Insider_Threat', 'Credential_Theft'], 0.40)
siem_analysis.set_mass(['APT_Advanced'], 0.25)
siem_analysis.set_mass(cyber_menaces, 0.35)

print("\nFUSION MULTI-SOURCES DE CYBERSÉCURITÉ :")
combined_cyber1 = comportement_reseau.combine(endpoint_detection)
combined_cyber2 = combined_cyber1.combine(threat_intelligence)
combined_cyber_final = combined_cyber2.combine(siem_analysis)

comportement_reseau.print_analysis("Analyse Comportementale Réseau")
endpoint_detection.print_analysis("Détection Endpoint")
threat_intelligence.print_analysis("Threat Intelligence")
siem_analysis.print_analysis("Analyse SIEM")
combined_cyber_final.print_analysis("DÉTECTION CYBER FINALE")

# Analyse de conflit détaillée
conflit_etapes = [
    ("Réseau-Endpoint", comportement_reseau.combine(endpoint_detection)),
    ("+ThreatIntel", combined_cyber1.combine(threat_intelligence)),
    ("Final", combined_cyber_final)
]

print("\nANALYSE DE CONFLIT PAR ÉTAPE :")
for nom, combinaison in conflit_etapes:
    conflit = 1 - sum(combinaison.mass.values())
    print(f"  {nom:20}: Conflit = {conflit:.4f}")

# Décision automatisée avec seuil de confiance
def decision_securite(ds, seuil_confidence=0.6):
    menaces = list(ds.theta)
    croyances = {menace: ds.belief([menace]) for menace in menaces}
    plausibilites = {menace: ds.plausibility([menace]) for menace in menaces}
    
    menace_principale = max(croyances.items(), key=lambda x: x[1])
    
    print(f"\nDÉCISION AUTOMATISÉE (seuil: {seuil_confidence}):")
    print(f"Menace identifiée : {menace_principale[0]} (croyance: {menace_principale[1]:.4f})")
    
    if menace_principale[1] >= seuil_confidence:
        print(f"ACTION : Déclencher protocole de réponse pour {menace_principale[0]}")
    else:
        print("INCERTITUDE : Surveillance renforcée nécessaire")
        
    return menace_principale[0]

decision_securite(combined_cyber_final, 0.6)



# =============================================================================
# EXEMPLE 2 - SYSTÈME DE PRÉVISION ÉCONOMIQUE AVANCÉ
# =============================================================================

print("\n\nEXEMPLE 2 - SYSTÈME DE PRÉVISION ÉCONOMIQUE MULTIVARIÉ")
print("Cadre : Analyse prospective des tendances économiques mondiales")

# Cadre de discernement : scénarios économiques
scenarios_economiques = [
    'Recession_Globale',
    'Croissance_Moderee', 
    'Stagnation_Prolongee',
    'Inflation_Galopante',
    'Crise_Financiere',
    'Bulle_Speculative'
]

# Indicateur 1 : Marchés financiers (analyse technique)
marches_financiers = DempsterShafer(scenarios_economiques)
marches_financiers.set_mass(['Crise_Financiere', 'Bulle_Speculative'], 0.30)
marches_financiers.set_mass(['Recession_Globale'], 0.20)
marches_financiers.set_mass(['Inflation_Galopante', 'Stagnation_Prolongee'], 0.25)
marches_financiers.set_mass(scenarios_economiques, 0.25)

# Indicateur 2 : Données macroéconomiques réelles
macroeconomic = DempsterShafer(scenarios_economiques)
macroeconomic.set_mass(['Croissance_Moderee', 'Stagnation_Prolongee'], 0.35)
macroeconomic.set_mass(['Inflation_Galopante'], 0.25)
macroeconomic.set_mass(['Recession_Globale'], 0.15)
macroeconomic.set_mass(scenarios_economiques, 0.25)

# Indicateur 3 : Sentiment des investisseurs (IA NLP)
sentiment_investisseurs = DempsterShafer(scenarios_economiques)
sentiment_investisseurs.set_mass(['Bulle_Speculative'], 0.40)
sentiment_investisseurs.set_mass(['Croissance_Moderee'], 0.20)
sentiment_investisseurs.set_mass(['Crise_Financiere'], 0.15)
sentiment_investisseurs.set_mass(scenarios_economiques, 0.25)

# Indicateur 4 : Géopolitique et facteurs externes
geopolitique = DempsterShafer(scenarios_economiques)
geopolitique.set_mass(['Recession_Globale', 'Crise_Financiere'], 0.45)
geopolitique.set_mass(['Inflation_Galopante'], 0.20)
geopolitique.set_mass(scenarios_economiques, 0.35)

print("\nFUSION DES INDICATEURS ÉCONOMIQUES :")
combined_eco1 = marches_financiers.combine(macroeconomic)
combined_eco2 = combined_eco1.combine(sentiment_investisseurs)
combined_eco_final = combined_eco2.combine(geopolitique)

marches_financiers.print_analysis("Marchés Financiers")
macroeconomic.print_analysis("Données Macroéconomiques")
sentiment_investisseurs.print_analysis("Sentiment Investisseurs")
geopolitique.print_analysis("Facteurs Géopolitiques")
combined_eco_final.print_analysis("PRÉVISION ÉCONOMIQUE FINALE")

# Analyse de risque avancée
def analyse_risque_economique(ds):
    scenarios = list(ds.theta)
    
    print("\nANALYSE DE RISQUE DÉTAILLÉE :")
    
    # Scénarios pessimistes
    scenarios_pessimistes = ['Recession_Globale', 'Crise_Financiere', 'Inflation_Galopante']
    risque_global = ds.belief(scenarios_pessimistes)
    plausibilite_pessimiste = ds.plausibility(scenarios_pessimistes)
    
    print(f"Risque scénarios pessimistes : {risque_global:.4f}")
    print(f"Plausibilité pessimiste : {plausibilite_pessimiste:.4f}")
    print(f"Intervalle de risque : [{risque_global:.4f}, {plausibilite_pessimiste:.4f}]")
    
    # Indice de confiance économique
    scenarios_favorables = ['Croissance_Moderee']
    confiance = ds.belief(scenarios_favorables)
    
    print(f"Indice de confiance économique : {confiance:.4f}")
    
    if risque_global > 0.6:
        return "ALERTE_HAUTE"
    elif risque_global > 0.4:
        return "VIGILANCE"
    else:
        return "CONFIANCE"

niveau_alerte = analyse_risque_economique(combined_eco_final)
print(f"\nNIVEAU D'ALERTE ÉCONOMIQUE : {niveau_alerte}")

# Simulation d'évolution temporelle
def simulation_evolution_temporelle():
    """Simule l'évolution des croyances sur plusieurs périodes"""
    print("\nSIMULATION ÉVOLUTION TEMPORELLE (3 périodes) :")
    
    # Période 1
    per1 = marches_financiers.combine(macroeconomic)
    print("Période 1 - Scénario dominant :", max(per1.theta, key=lambda x: per1.belief([x])))
    
    # Période 2 (ajout sentiment)
    per2 = per1.combine(sentiment_investisseurs)
    print("Période 2 - Scénario dominant :", max(per2.theta, key=lambda x: per2.belief([x])))
    
    # Période 3 (ajout géopolitique)
    per3 = per2.combine(geopolitique)
    print("Période 3 - Scénario dominant :", max(per3.theta, key=lambda x: per3.belief([x])))
    
    return per3

final_simulation = simulation_evolution_temporelle()

# Métriques avancées pour les deux scénarios
print("\n" + "="*80)
print("ANALYSE COMPARATIVE DES DEUX SYSTÈMES EXPERTS")
print("="*80)

metriques_cyber = calculer_metriques(combined_cyber_final, "Système Cybersécurité")
metriques_economie = calculer_metriques(combined_eco_final, "Prévision Économique")

print(f"\nSYSTÈME LE PLUS DÉCISIF : {'Cybersécurité' if metriques_cyber[0] < metriques_economie[0] else 'Économie'}")
print(f"NIVEAU DE CONFLIT MOYEN : {(metriques_cyber[1] + metriques_economie[1])/2:.4f}")