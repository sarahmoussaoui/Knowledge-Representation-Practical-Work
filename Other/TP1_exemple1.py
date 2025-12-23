import numpy as np
from itertools import chain, combinations

class DempsterShafer:
    """
    Implémentation de la Théorie des Évidences de Dempster-Shafer (DST)
    pour la fusion d'informations incertaines.
    """
    def __init__(self, frame_of_discernment):
        self.theta = frame_of_discernment
        self.elements = list(self.powerset(self.theta))
        # Initialisation de la masse à 0 pour tous les sous-ensembles (focal elements)
        self.mass = {frozenset(elem): 0.0 for elem in self.elements}
        self.mass[frozenset()] = 0.0
    
    def powerset(self, iterable):
        """Calcule l'ensemble des parties (powerset) de l'ensemble de discernement."""
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    def set_mass(self, subset, value):
        """Attribue une masse de croyance à un sous-ensemble (élément focal)."""
        self.mass[frozenset(subset)] = value
    
    def belief(self, hypothesis):
        """Calcule le degré de croyance (Bel) pour une hypothèse."""
        hyp_set = frozenset(hypothesis)
        belief_value = 0.0
        for subset in self.mass:
            # Bel(H) est la somme des masses des sous-ensembles contenus dans H (sauf l'ensemble vide)
            if subset.issubset(hyp_set) and subset != frozenset():
                belief_value += self.mass[subset]
        return belief_value
    
    def plausibility(self, hypothesis):
        """Calcule la plausibilité (Pl) pour une hypothèse."""
        hyp_set = frozenset(hypothesis)
        pl_value = 0.0
        for subset in self.mass:
            # Pl(H) est la somme des masses des sous-ensembles qui intersectent H
            if subset.intersection(hyp_set) != frozenset():
                pl_value += self.mass[subset]
        return pl_value
    
    def confidence_interval(self, hypothesis):
        """Retourne l'intervalle de confiance [Bel, Pl]."""
        return [self.belief(hypothesis), self.plausibility(hypothesis)]
    
    def combine(self, other):
        """
        Applique la règle de combinaison de Dempster pour fusionner deux fonctions 
        de masse (m1 et m2).
        """
        combined = DempsterShafer(self.theta)
        
        # 1. Calcul du degré de conflit K (normalisation)
        K = 0
        for A in self.mass:
            for B in other.mass:
                if A.intersection(B) == frozenset():
                    K += self.mass[A] * other.mass[B]
        
        # 2. Application de la règle de combinaison m(C) = (1 / (1 - K)) * sum(m1(A) * m2(B))
        if K < 1:
            normalisation_factor = 1 / (1 - K)
            for A in self.mass:
                for B in other.mass:
                    intersection = A.intersection(B)
                    if intersection != frozenset():
                        combined.mass[intersection] += (self.mass[A] * other.mass[B]) * normalisation_factor
        
        # Si K=1, il y a un conflit, combinaison undéfinie
        
        return combined
    
    def print_analysis(self, title):
        """Affiche une analyse détaillée des masses, croyances et conclusions."""
        print("\n" + "="*60)
        print(f"ANALYSE DETAILLEE : {title}")
        print("="*60)
        
        print("\nMASSES DE CROYANCE (m) :")
        for key, value in sorted(self.mass.items(), key=lambda x: -x[1]):
            if value > 0.001:
                set_str = str(set(key)).replace("'", "") 
                print(f"  m({set_str}) = {value:.4f}")
        
        print("\nDEGRES DE CROYANCE (Bel) ET PLAUSIBILITE (Pl) :")
        elements = list(self.theta)
        for elem in elements:
            bel = self.belief([elem])
            pl = self.plausibility([elem])
            interval = self.confidence_interval([elem])
            print(f"  {elem:15}: Bel = {bel:.4f}, Pl = {pl:.4f}, Incertitude = {pl-bel:.4f}, Intervalle = [{interval[0]:.4f}, {interval[1]:.4f}]")
        
        best = max(elements, key=lambda x: self.belief([x]))
        print(f"\nCONCLUSION : La cause la plus probable est '{best}' (Croyance la plus élevée = {self.belief([best]):.4f})")

    def print_detailed_analysis(self, title):
        """Affiche une analyse complète incluant les ensembles composites."""
        print("\n" + "="*60)
        print(f"ANALYSE DETAILLEE : {title}")
        print("="*60)
        
        print("\nMASSES DE CROYANCE (m) :")
        for key, value in sorted(self.mass.items(), key=lambda x: -x[1]):
            if value > 0.001:
                set_str = str(set(key)).replace("'", "")
                print(f"  m({set_str}) = {value:.4f}")
        
        print("\nDEGRES DE CROYANCE (Bel) ET PLAUSIBILITE (Pl) :")
        
        # 1. D'abord les hypothèses élémentaires
        print("\n  --- HYPOTHÈSES ÉLÉMENTAIRES ---")
        elements = list(self.theta)
        for elem in elements:
            bel = self.belief([elem])
            pl = self.plausibility([elem])
            interval = self.confidence_interval([elem])
            print(f"  {elem:15}: Bel = {bel:.4f}, Pl = {pl:.4f}, "
                f"Incertitude = {pl-bel:.4f}")
        
        # 2. Ensuite les combinaisons importantes (celles avec masse > 0)
        print("\n  --- ENSEMBLES COMPOSITES IMPORTANTS ---")
        
        # Générer toutes les combinaisons de 2 éléments
        from itertools import combinations
        
        for combo_size in [2, 3]:  # Vous pouvez ajuster
            for combo in combinations(elements, combo_size):
                subset = list(combo)
                if subset in [list(k) for k in self.mass.keys()]:  
                    bel = self.belief(subset)
                    pl = self.plausibility(subset)
                    
                    # Ne montrer que si intéressant (masse > 0 ou Bel > 0)
                    if bel > 0.001 or pl > 0.5:
                        set_str = "{" + ", ".join(subset) + "}"
                        print(f"  {set_str:15}: Bel = {bel:.4f}, Pl = {pl:.4f}, "
                            f"Incertitude = {pl-bel:.4f}")
        
        # 3. L'ensemble complet Θ
        print("\n  --- ENSEMBLE COMPLET ---")
        bel_theta = self.belief(list(self.theta))
        pl_theta = self.plausibility(list(self.theta))
        print(f"  Θ (toutes causes): Bel = {bel_theta:.4f}, Pl = {pl_theta:.4f}")
        
        # Trouver la meilleure hypothèse (élémentaire ou composite)
        best_elem = max(elements, key=lambda x: self.belief([x]))
        best_bel = self.belief([best_elem])
        
        print(f"\nCONCLUSION (basée sur hypothèses élémentaires): "
            f"La cause la plus probable est '{best_elem}' "
            f"(Croyance = {best_bel:.4f})")

#  EXEMPLE D'UTILISATION DANS UN CONTEXTE MÉDICAL
def decision_medicale_acne(ds):
    """Fournit une recommandation thérapeutique basée sur le diagnostic DS."""
    print("\n" + "#"*40)
    print("RECOMMANDATION THÉRAPEUTIQUE AUTOMATISÉE")
    print("#"*40)
    
    meilleure_croyance = max(ds.theta, key=lambda x: ds.belief([x]))
    croyance_max = ds.belief([meilleure_croyance])
    
    # Seuil de décision pour la confiance
    seuil_confidence = 0.40 
    
    print(f"Diagnostic principal : **{meilleure_croyance}** (Croyance: {croyance_max:.4f})")
    
    if croyance_max >= seuil_confidence:
        if meilleure_croyance == 'Hormonale':
            print("Action : Forte suspicion de cause Hormonale. Envisager un traitement ciblé (ex: pilule ou anti-androgènes).")
        elif meilleure_croyance == 'Stress':
            print("Action : Forte suspicion de Stress. Recommandation : gestion du stress (yoga, méditation, techniques de relaxation).")
        elif meilleure_croyance == 'Alimentaire':
            print("Action : Forte suspicion Alimentaire. Prescrire un journal alimentaire, conseiller un régime faible en IG et produits laitiers.")
        elif meilleure_croyance == 'Hygiène':
            print("Action : Forte suspicion Hygiène. Réviser la routine de soins (nettoyage doux, non-comédogène).")
        elif meilleure_croyance == 'Pas_Acné':
            print("Action : Le diagnostic d'acné est faible. Reconsidérer d'autres problèmes dermatologiques.")
        else:
            print(f"Action : Cible principale - {meilleure_croyance}. Recommandation spécifique.")
    else:
        print("ACTION : **INCERTITUDE ÉLEVÉE**. Le niveau de croyance est insuffisant. Recommander des tests complémentaires ou un suivi rapproché pour affiner le diagnostic.")


# =============================================================================
# CAS D'ÉTUDE : DIAGNOSTIC MÉDICAL DE LA PROVENANCE DE L'ACNÉ
# =============================================================================

print("CAS D'ÉTUDE : SYSTÈME EXPERT DE DIAGNOSTIC DE LA PROVENANCE DE L'ACNÉ")
print("----------------------------------------------------------------------")

# Cadre de discernement (Theta) : les causes mutuellement exclusives possibles
causes_acne = [
    'Hormonale',
    'Alimentaire', 
    'Stress',
    'Hygiène',
    'Pas_Acné' # Pour modéliser l'absence d'acné
]

# --- SOURCE 1 : Observation Clinique du Dermatologue (m1) ---
# Le dermato observe des lésions profondes (hormonal/stress) et des comédons (hygiène/alimentaire)
dermatologue_obs = DempsterShafer(causes_acne)
dermatologue_obs.set_mass(['Hormonale', 'Stress'], 0.40)
dermatologue_obs.set_mass(['Hygiène', 'Alimentaire'], 0.25)
dermatologue_obs.set_mass(causes_acne, 0.35) # Incertitude résiduelle

# --- SOURCE 2 : Résultats des Tests Hormonaux (m2) ---
# Un test sanguin montre une perturbation hormonale claire
tests_hormonaux = DempsterShafer(causes_acne)
tests_hormonaux.set_mass(['Hormonale'], 0.55)
tests_hormonaux.set_mass(['Pas_Acné', 'Alimentaire'], 0.10) # Petite chance que ce soit autre chose
tests_hormonaux.set_mass(causes_acne, 0.35)

# --- SOURCE 3 : Historique du Patient (m3) ---
# Le patient rapporte une période de stress et des changements alimentaires récents
historique_patient = DempsterShafer(causes_acne)
historique_patient.set_mass(['Stress', 'Alimentaire'], 0.60)
historique_patient.set_mass(['Hormonale'], 0.10) # Laisse une faible possibilité d'une cause interne
historique_patient.set_mass(causes_acne, 0.30)

print("\nPROCESSUS DE FUSION (Règle de Dempster) :")
# Fusion m1 et m2
combined_acne1 = dermatologue_obs.combine(tests_hormonaux)
# Fusion (m1 * m2) et m3
combined_acne_final = combined_acne1.combine(historique_patient)

# Affichage des analyses individuelles
dermatologue_obs.print_detailed_analysis("Observation Dermatologue (m1)")
tests_hormonaux.print_detailed_analysis("Tests Hormonaux (m2)")
historique_patient.print_detailed_analysis("Historique Patient (m3)")

# Affichage du diagnostic final après fusion
combined_acne_final.print_detailed_analysis("DIAGNOSTIC FINAL APRÈS FUSION (m1 * m2 * m3)")

# Prise de décision basée sur le résultat final
decision_medicale_acne(combined_acne_final)