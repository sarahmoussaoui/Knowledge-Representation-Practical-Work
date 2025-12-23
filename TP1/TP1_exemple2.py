import numpy as np
from itertools import chain, combinations

# =============================================================================
# CLASSE DEMPSTER-SHAFER AMÉLIORÉE
# =============================================================================

class DempsterShafer:
    def __init__(self, frame_of_discernment):
        self.theta = frame_of_discernment
        self.elements = list(self.powerset(self.theta))
        self.mass = {frozenset(e): 0.0 for e in self.elements}
        self.mass[frozenset()] = 0.0

    def powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def set_mass(self, subset, value):
        self.mass[frozenset(subset)] = value

    def belief(self, hypothesis):
        H = frozenset(hypothesis)
        return sum(v for k, v in self.mass.items() if k.issubset(H) and k)

    def plausibility(self, hypothesis):
        H = frozenset(hypothesis)
        return sum(v for k, v in self.mass.items() if k & H)

    def confidence_interval(self, hypothesis):
        """Retourne l'intervalle de confiance [Bel, Pl]."""
        return [self.belief(hypothesis), self.plausibility(hypothesis)]
    
    def combine(self, other):
        combined = DempsterShafer(self.theta)
        K = 0.0

        for A, mA in self.mass.items():
            for B, mB in other.mass.items():
                if A & B == frozenset():
                    K += mA * mB

        if K == 1:
            return combined  # conflit total

        for A, mA in self.mass.items():
            for B, mB in other.mass.items():
                inter = A & B
                if inter:
                    combined.mass[inter] += (mA * mB) / (1 - K)

        return combined

    def print_analysis(self, title):
        """Affiche une analyse simplifiée des masses et croyances."""
        print("\n" + "="*60)
        print(f"ANALYSE DETAILLEE : {title}")
        print("="*60)

        print("\nMASSES DE CROYANCE (m) :")
        for k, v in sorted(self.mass.items(), key=lambda x: -x[1]):
            if v > 0.001:
                set_str = str(set(k)).replace("'", "")
                print(f"  m({set_str}) = {v:.4f}")

        print("\nDEGRES DE CROYANCE (Bel) ET PLAUSIBILITE (Pl) :")
        for h in self.theta:
            bel = self.belief([h])
            pl = self.plausibility([h])
            interval = self.confidence_interval([h])
            print(f"  {h:15}: Bel = {bel:.4f}, Pl = {pl:.4f}, "
                  f"Incertitude = {pl-bel:.4f}, "
                  f"Intervalle = [{interval[0]:.4f}, {interval[1]:.4f}]")

        # Trouver la meilleure hypothèse (élémentaire avec la plus haute croyance)
        best = max(self.theta, key=lambda x: self.belief([x]))
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
            if best_elem_bel >= 0.1:  # seuil ajustable
                print(f"  RECOMMANDATION: Privilégier l'hypothèse '{best_elem}'")
            else:
                print(f"  RECOMMANDATION: L'incertitude est trop élevée. "
                      f"Considérer l'ensemble {set_str}")
        else:
            print(f"  La cause la plus probable est '{best_elem}' "
                  f"(Croyance = {best_elem_bel:.4f})")


def decision_couscous_ameliorée(ds):
    """Fournit une recommandation culinaire améliorée basée sur l'analyse DS."""
    print("\n" + "#"*60)
    print("RECOMMANDATION CULINAIRE AMÉLIORÉE")
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
    
    # Stratégie de décision spécifique au couscous
    if croyance_elementaire >= seuil_fort:
        print("NIVEAU DE CONFIANCE : ÉLEVÉ")
        if meilleure_elementaire == 'Sel':
            print("ACTION : Problème de sel détecté avec certitude.")
            print("         Solution : Gouter avant de saler, ajuster en cours de cuisson.")
        elif meilleure_elementaire == 'Cuisson':
            print("ACTION : Problème de cuisson identifié.")
            print("         Solution : Vérifier temps/température, utiliser couscoussier.")
        elif meilleure_elementaire == 'Ingredients':
            print("ACTION : Problème d'ingrédients confirmé.")
            print("         Solution : Vérifier qualité/fraîcheur des ingrédients.")
        elif meilleure_elementaire == 'Pas_Probleme':
            print("ACTION : Le couscous était probablement bon.")
            print("         Solution : Reconsidérer les attentes ou vérifier les convives.")
    
    elif croyance_elementaire >= seuil_modere:
        print("NIVEAU DE CONFIANCE : MODÉRÉ")
        if meilleur_composite and masse_composite > croyance_elementaire:
            print(f"ACTION : Considérer l'ensemble {set_str}.")
            print(f"         Vérifier plusieurs aspects : {set_str}")
            if 'Sel' in set_str and 'Cuisson' in set_str:
                print("         Astuce : Le sel doit être ajouté au bon moment de la cuisson.")
        else:
            print(f"ACTION : Suspicion modérée de {meilleure_elementaire}.")
            print("         Faire un test ciblé + surveiller les autres aspects.")
    
    elif croyance_elementaire >= seuil_faible:
        print("NIVEAU DE CONFIANCE : FAIBLE")
        print("ACTION : **INCERTITUDE SIGNIFICATIVE**.")
        if meilleur_composite:
            print(f"         L'ensemble {set_str} est le plus soutenu (m={masse_composite:.4f}).")
            print("         Approche systématique recommandée : vérifier tous les aspects.")
        else:
            print("         Approche diagnostique élargie recommandée.")
            print("         Vérifier séquentiellement : sel → cuisson → ingrédients.")
    
    else:
        print("NIVEAU DE CONFIANCE : TRÈS FAIBLE")
        print("ACTION : **INCERTITUDE ÉLEVÉE**.")
        print("         Tests complémentaires nécessaires :")
        print("         1. Faire goûter à plusieurs personnes")
        print("         2. Préparer un échantillon test avec paramètres contrôlés")
        print("         3. Comparer avec une recette de référence")


# =============================================================================
# CAS D'ÉTUDE : POURQUOI LE COUSCOUS N'ÉTAIT PAS BON ?
# =============================================================================

print("CAS D'ÉTUDE : ANALYSE EXPERTE - POURQUOI LE COUSCOUS N'ÉTAIT PAS BON ?")
print("="*80)
print("""
Scénario : Vous avez préparé un couscous pour un repas familial.
Les avis sont partagés sur la raison du problème. Nous allons fusionner
les opinions des experts culinaires de la famille pour trouver la cause.
""")

# Cadre de discernement (Theta) : les causes possibles
causes_couscous = [
    'Sel',          # Trop/moins de sel
    'Cuisson',      # Temps/température inadéquats
    'Ingredients',  # Qualité des ingrédients
    'Pas_Probleme'  # En fait, il était bon!
]

# --- SOURCE 1 : LA GRAND-MÈRE (expérience traditionnelle) ---
print("\nSOURCE 1 : LA GRAND-MÈRE (60 ans d'expérience)")
print("  - Elle goûte et trouve que 'ça manque de coeur'")
print("  - Elle soupçonne surtout la cuisson, peut-être le sel aussi")
print("  - Elle a confiance en sa méthode traditionnelle")

grand_mere = DempsterShafer(causes_couscous)
grand_mere.set_mass(['Cuisson'], 0.6)           # Forte conviction sur la cuisson
grand_mere.set_mass(['Sel', 'Cuisson'], 0.2)    # Possible combinaison sel + cuisson
grand_mere.set_mass(causes_couscous, 0.2)       # Incertitude résiduelle

# --- SOURCE 2 : LE CHEF (formation professionnelle) ---
print("\nSOURCE 2 : LE CHEF (formation professionnelle)")
print("  - Il examine la texture et l'apparence")
print("  - Il soupçonne surtout la qualité des ingrédients")
print("  - Il reconnaît aussi une possible sous-cuisson")

chef = DempsterShafer(causes_couscous)
chef.set_mass(['Ingredients'], 0.6)             # Conviction sur les ingrédients
chef.set_mass(['Cuisson'], 0.1)                 # Légère suspicion sur la cuisson
chef.set_mass(causes_couscous, 0.3)             # Incertitude résiduelle

# --- SOURCE 3 : LA MÈRE (praticienne régulière) ---
print("\nSOURCE 3 : LA MÈRE (praticienne régulière)")
print("  - Elle a préparé avec vous et a surveillé le sel")
print("  - Elle pense peut-être qu'il n'y a pas de problème")
print("  - Elle est moins catégorique que les autres")

mere = DempsterShafer(causes_couscous)
mere.set_mass(['Sel'], 0.3)                     # Suspicion sur le sel
mere.set_mass(['Pas_Probleme'], 0.3)            # Peut-être qu'il était bon
mere.set_mass(causes_couscous, 0.4)             # Beaucoup d'incertitude

print("\n" + "="*80)
print("PROCESSUS DE FUSION DES OPINIONS (Règle de Dempster)")
print("="*80)

# Fusion des opinions
print("\nÉtape 1 : Fusion Grand-mère + Chef")
fusion_1 = grand_mere.combine(chef)

print("Étape 2 : Fusion (Grand-mère*Chef) + Mère")
fusion_finale = fusion_1.combine(mere)

print("\n" + "="*80)
print("ANALYSES INDIVIDUELLES DES SOURCES")
print("="*80)

# Affichage des analyses individuelles détaillées
grand_mere.print_detailed_analysis("1. Analyse de la Grand-mère")
chef.print_detailed_analysis("2. Analyse du Chef")
mere.print_detailed_analysis("3. Analyse de la Mère")

print("\n" + "="*80)
print("DIAGNOSTIC FINAL APRÈS FUSION DES TROIS OPINIONS")
print("="*80)

# Affichage du diagnostic final après fusion
fusion_finale.print_detailed_analysis("DIAGNOSTIC FINAL (Grand-mère * Chef * Mère)")

# Prise de décision améliorée
decision_couscous_ameliorée(fusion_finale)

# Option: Afficher aussi l'analyse simple pour comparaison
print("\n" + "="*80)
print("ANALYSE SIMPLIFIÉE DU RÉSULTAT FINAL")
print("="*80)
fusion_finale.print_analysis("Vue simplifiée du diagnostic final")

# =============================================================================
# ANALYSE COMPARATIVE
# =============================================================================
print("\n" + "="*80)
print("ANALYSE COMPARATIVE : ÉVOLUTION DES CROYANCES")
print("="*80)

print("\nCroyances élémentaires à chaque étape :")
print(f"{'Hypothèse':<15} {'Grand-mère':<10} {'Chef':<10} {'Mère':<10} {'Fusion':<10}")
print("-" * 60)

for cause in causes_couscous:
    bel_gm = grand_mere.belief([cause])
    bel_chef = chef.belief([cause])
    bel_mere = mere.belief([cause])
    bel_final = fusion_finale.belief([cause])
    
    print(f"{cause:<15} {bel_gm:<10.4f} {bel_chef:<10.4f} {bel_mere:<10.4f} {bel_final:<10.4f}")

# =============================================================================
# RECOMMANDATIONS PRATIQUES
# =============================================================================
print("\n" + "="*80)
print("RECOMMANDATIONS PRATIQUES POUR LA PROCHAINE FOIS")
print("="*80)

# Analyser les résultats pour donner des conseils pratiques
final_beliefs = {cause: fusion_finale.belief([cause]) for cause in causes_couscous}
sorted_causes = sorted(final_beliefs.items(), key=lambda x: x[1], reverse=True)

print("\nOrdre de priorité pour investigation :")
for i, (cause, belief) in enumerate(sorted_causes, 1):
    if cause != 'Pas_Probleme':
        print(f"{i}. {cause} (croyance: {belief:.4f})")

print("\nPlan d'action recommandé :")
print("1. Vérifier d'abord : " + sorted_causes[0][0])
print("2. Puis vérifier : " + sorted_causes[1][0])
print("3. Enfin vérifier : " + sorted_causes[2][0])
print("\nSi aucune cause n'est trouvée : reconsidérer l'hypothèse 'Pas_Probleme'")