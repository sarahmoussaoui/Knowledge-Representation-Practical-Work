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
        
        # Si K=1, il y a un conflit total, la combinaison est indéfinie
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
                print(f"  m({set_str}) = {value:.4f}")
        
        print("\nDEGRES DE CROYANCE (Bel) ET PLAUSIBILITE (Pl) :")
        elements = list(self.theta)
        for elem in elements:
            bel = self.belief([elem])
            pl = self.plausibility([elem])
            interval = self.confidence_interval([elem])
            print(f"  {elem:15}: Bel = {bel:.4f}, Pl = {pl:.4f}, "
                  f"Incertitude = {pl-bel:.4f}, "
                  f"Intervalle = [{interval[0]:.4f}, {interval[1]:.4f}]")
        
        # Trouver la meilleure hypothèse (élémentaire avec la plus haute croyance)
        best = max(elements, key=lambda x: self.belief([x]))
        best_bel = self.belief([best])
        
        # Amélioration: Si toutes les croyances sont 0, on vérifie les ensembles composites
        if best_bel == 0.0:
            # Chercher parmi tous les ensembles non-vides
            all_subsets = [subset for subset in self.mass if subset != frozenset()]
            if all_subsets:
                best_composite = max(all_subsets, key=lambda x: self.mass[x])
                best_mass = self.mass[best_composite]
                set_str = str(set(best_composite)).replace("'", "")
                print(f"\nNOTE : Toutes les croyances élémentaires sont à 0.")
                print(f"L'ensemble composite avec la plus haute masse est {set_str} (m = {best_mass:.4f})")
        
        print(f"\nCONCLUSION (élémentaire): La cause la plus probable est '{best}' "
              f"(Croyance = {best_bel:.4f})")

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
        
        # 2. Ensuite les combinaisons importantes
        print("\n  --- ENSEMBLES COMPOSITES IMPORTANTS ---")
        
        # Liste pour stocker les ensembles affichés (éviter les doublons)
        displayed_subsets = []
        
        # D'abord afficher les ensembles avec masse > 0
        for subset, mass in sorted(self.mass.items(), key=lambda x: -x[1]):
            if mass > 0.001 and len(subset) > 1:  # Uniquement les ensembles composites
                subset_list = list(subset)
                bel = self.belief(subset_list)
                pl = self.plausibility(subset_list)
                set_str = "{" + ", ".join(subset_list) + "}"
                print(f"  {set_str:25}: Bel = {bel:.4f}, Pl = {pl:.4f}, "
                      f"Incertitude = {pl-bel:.4f}, m = {mass:.4f}")
                displayed_subsets.append(frozenset(subset_list))
        
        # Ensuite, générer d'autres combinaisons intéressantes
        from itertools import combinations
        
        # Générer les combinaisons de 2 éléments
        all_combinations = []
        for combo_size in [2, 3]:
            for combo in combinations(elements, combo_size):
                all_combinations.append(frozenset(combo))
        
        # Afficher les combinaisons qui n'ont pas encore été affichées et qui sont intéressantes
        for combo in all_combinations:
            if combo not in displayed_subsets and combo != frozenset():
                subset_list = list(combo)
                bel = self.belief(subset_list)
                pl = self.plausibility(subset_list)
                
                # Afficher si plausibilité > 0.5 OU si l'ensemble est intéressant
                if pl > 0.5 or bel > 0:
                    set_str = "{" + ", ".join(subset_list) + "}"
                    print(f"  {set_str:25}: Bel = {bel:.4f}, Pl = {pl:.4f}, "
                          f"Incertitude = {pl-bel:.4f}")
        
        # 3. L'ensemble complet Θ
        print("\n  --- ENSEMBLE COMPLET ---")
        bel_theta = self.belief(list(self.theta))
        pl_theta = self.plausibility(list(self.theta))
        print(f"  Θ (toutes causes): Bel = {bel_theta:.4f}, Pl = {pl_theta:.4f}")
        
        # Trouver la meilleure conclusion
        print("\n  --- CONCLUSION ---")
        
        # Option 1: Meilleure hypothèse élémentaire
        best_elem = max(elements, key=lambda x: self.belief([x]))
        best_elem_bel = self.belief([best_elem])
        
        # Option 2: Meilleur ensemble composite (avec la plus haute masse)
        composite_subsets = [s for s in self.mass if len(s) > 1 and self.mass[s] > 0]
        if composite_subsets:
            best_composite = max(composite_subsets, key=lambda x: self.mass[x])
            best_comp_mass = self.mass[best_composite]
            best_comp_bel = self.belief(list(best_composite))
            set_str = str(set(best_composite)).replace("'", "")
            
            print(f"  Hypothèse élémentaire la plus probable: '{best_elem}' "
                  f"(Bel = {best_elem_bel:.4f})")
            print(f"  Ensemble composite le plus soutenu: {set_str} "
                  f"(m = {best_comp_mass:.4f}, Bel = {best_comp_bel:.4f})")
            
            # Décision: choisir l'élémentaire si sa croyance est raisonnable, sinon le composite
            if best_elem_bel >= 0.1: # seuil aléatoire
                print(f"  RECOMMANDATION: Privilégier l'hypothèse '{best_elem}'")
            else:
                print(f"  RECOMMANDATION: L'incertitude est trop élevée. "
                      f"Considérer l'ensemble {set_str}")
        else:
            print(f"  La cause la plus probable est '{best_elem}' "
                  f"(Croyance = {best_elem_bel:.4f})")


def decision_medicale_acne_ameliorée(ds):
    """Fournit une recommandation thérapeutique améliorée basée sur le diagnostic DS."""
    print("\n" + "#"*60)
    print("RECOMMANDATION THÉRAPEUTIQUE AMÉLIORÉE")
    print("#"*60)
    
    elements = list(ds.theta)
    
    # 1. Meilleure hypothèse élémentaire
    meilleure_elementaire = max(elements, key=lambda x: ds.belief([x]))
    croyance_elementaire = ds.belief([meilleure_elementaire])
    
    # 2. Meilleur ensemble composite (masse la plus élevée)
    composite_subsets = [s for s in ds.mass if len(s) > 1 and ds.mass[s] > 0]
    if composite_subsets:
        meilleur_composite = max(composite_subsets, key=lambda x: ds.mass[x])
        masse_composite = ds.mass[meilleur_composite]
        set_str = str(set(meilleur_composite)).replace("'", "")
    else:
        meilleur_composite = None
    
    # 3. Décision avec plusieurs seuils
    seuil_fort = 0.40
    seuil_modere = 0.20
    seuil_faible = 0.10
    
    print(f"Diagnostic élémentaire : **{meilleure_elementaire}** "
          f"(Croyance: {croyance_elementaire:.4f})")
    
    if meilleur_composite:
        print(f"Diagnostic composite : **{set_str}** "
              f"(Masse: {masse_composite:.4f})")
    
    print("\n" + "-"*40)
    
    # Stratégie de décision
    if croyance_elementaire >= seuil_fort:
        print("NIVEAU DE CONFIANCE : ÉLEVÉ")
        if meilleure_elementaire == 'Hormonale':
            print("ACTION : Forte suspicion de cause Hormonale.")
            print("         Envisager un traitement ciblé (pilule, anti-androgènes).")
        elif meilleure_elementaire == 'Stress':
            print("ACTION : Forte suspicion de Stress.")
            print("         Recommandation : gestion du stress.")
        elif meilleure_elementaire == 'Alimentaire':
            print("ACTION : Forte suspicion Alimentaire.")
            print("         Journal alimentaire + régime faible IG.")
        elif meilleure_elementaire == 'Hygiène':
            print("ACTION : Forte suspicion Hygiène.")
            print("         Réviser la routine de soins.")
        elif meilleure_elementaire == 'Pas_Acné':
            print("ACTION : Le diagnostic d'acné est faible.")
            print("         Reconsidérer d'autres problèmes dermatologiques.")
    
    elif croyance_elementaire >= seuil_modere:
        print("NIVEAU DE CONFIANCE : MODÉRÉ")
        if meilleur_composite and masse_composite > croyance_elementaire:
            print(f"ACTION : Considérer l'ensemble {set_str}.")
            print(f"         Traitement couvrant les causes: {set_str}")
        else:
            print(f"ACTION : Suspicion modérée de {meilleure_elementaire}.")
            print("         Traitement ciblé + surveillance.")
    
    elif croyance_elementaire >= seuil_faible:
        print("NIVEAU DE CONFIANCE : FAIBLE")
        print("ACTION : **INCERTITUDE SIGNIFICATIVE**.")
        if meilleur_composite:
            print(f"         L'ensemble {set_str} est le plus soutenu (m={masse_composite:.4f}).")
            print("         Approche combinée recommandée.")
        else:
            print("         Approche diagnostique élargie recommandée.")
    
    else:
        print("NIVEAU DE CONFIANCE : TRÈS FAIBLE")
        print("ACTION : **INCERTITUDE ÉLEVÉE**.")
        print("         Tests complémentaires nécessaires.")
        print("         Suivi rapproché pour affiner le diagnostic.")


# =============================================================================
# CAS D'ÉTUDE : DIAGNOSTIC MÉDICAL DE LA PROVENANCE DE L'ACNÉ
# =============================================================================

print("CAS D'ÉTUDE : SYSTÈME EXPERT DE DIAGNOSTIC DE LA PROVENANCE DE L'ACNÉ")
print("="*80)

# Cadre de discernement (Theta) : les causes mutuellement exclusives possibles
causes_acne = [
    'Hormonale',
    'Alimentaire', 
    'Stress',
    'Hygiène',
    'Pas_Acné' # Pour modéliser l'absence d'acné
]

# --- SOURCE 1 : Observation Clinique du Dermatologue (m1) ---
dermatologue_obs = DempsterShafer(causes_acne)
dermatologue_obs.set_mass(['Hormonale', 'Stress'], 0.40)
dermatologue_obs.set_mass(['Hygiène', 'Alimentaire'], 0.25)
dermatologue_obs.set_mass(causes_acne, 0.35) # Incertitude résiduelle

# --- SOURCE 2 : Résultats des Tests Hormonaux (m2) ---
tests_hormonaux = DempsterShafer(causes_acne)
tests_hormonaux.set_mass(['Hormonale'], 0.55)
tests_hormonaux.set_mass(['Pas_Acné', 'Alimentaire'], 0.10)
tests_hormonaux.set_mass(causes_acne, 0.35)

# --- SOURCE 3 : Historique du Patient (m3) ---
historique_patient = DempsterShafer(causes_acne)
historique_patient.set_mass(['Stress', 'Alimentaire'], 0.60)
historique_patient.set_mass(['Hormonale'], 0.10)
historique_patient.set_mass(causes_acne, 0.30)

print("\nPROCESSUS DE FUSION (Règle de Dempster) :")
print("-"*80)

# Fusion m1 et m2
combined_acne1 = dermatologue_obs.combine(tests_hormonaux)
# Fusion (m1 * m2) et m3
combined_acne_final = combined_acne1.combine(historique_patient)

print("\n" + "="*80)
print("ANALYSES INDIVIDUELLES DES SOURCES")
print("="*80)

# Affichage des analyses individuelles
dermatologue_obs.print_detailed_analysis("1. Observation Dermatologue (m1)")
tests_hormonaux.print_detailed_analysis("2. Tests Hormonaux (m2)")
historique_patient.print_detailed_analysis("3. Historique Patient (m3)")

print("\n" + "="*80)
print("DIAGNOSTIC FINAL APRÈS FUSION DES TROIS SOURCES")
print("="*80)

# Affichage du diagnostic final après fusion
combined_acne_final.print_detailed_analysis("DIAGNOSTIC FINAL (m1 * m2 * m3)")

# Prise de décision améliorée
decision_medicale_acne_ameliorée(combined_acne_final)

# Option: Afficher aussi l'analyse simple pour comparaison
print("\n" + "="*80)
print("ANALYSE SIMPLIFIÉE DU RÉSULTAT FINAL")
print("="*80)
combined_acne_final.print_analysis("Vue simplifiée du diagnostic final")