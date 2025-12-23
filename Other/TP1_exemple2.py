import numpy as np
from itertools import chain, combinations

# =============================================================================
# CLASSE DEMPSTER-SHAFER
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
        print("\n" + "="*60)
        print(title)
        print("="*60)

        print("\nMasses de croyance :")
        for k, v in sorted(self.mass.items(), key=lambda x: -x[1]):
            if v > 0.01:
                print(f" m({set(k)}) = {v:.3f}")

        print("\nCroyances et plausibilités :")
        for h in self.theta:
            bel = self.belief([h])
            pl = self.plausibility([h])
            print(f" {h:12} → Bel={bel:.3f}, Pl={pl:.3f}, Incertitude={pl-bel:.3f}")

        best = max(self.theta, key=lambda h: self.belief([h]))
        print(f"\nConclusion : cause la plus crédible → {best}")

# =============================================================================
# EXEMPLE : LE COUSCOUS N'ÉTAIT PAS BON
# =============================================================================

print("CAS D'ÉTUDE : POURQUOI LE COUSCOUS N'ÉTAIT PAS BON ?")
print("----------------------------------------------------")

causes = ['Sel', 'Cuisson', 'Ingredients', 'Pas_Probleme']

# GRAND-MÈRE
grand_mere = DempsterShafer(causes)
grand_mere.set_mass(['Cuisson'], 0.6)
grand_mere.set_mass(['Sel', 'Cuisson'], 0.2)
grand_mere.set_mass(causes, 0.2)

# CHEF
chef = DempsterShafer(causes)
chef.set_mass(['Ingredients'], 0.6)
chef.set_mass(['Cuisson'], 0.1)
chef.set_mass(causes, 0.3)

# MÈRE
mere = DempsterShafer(causes)
mere.set_mass(['Sel'], 0.3)
mere.set_mass(['Pas_Probleme'], 0.3)
mere.set_mass(causes, 0.4)

# =============================================================================
# FUSION DES CROYANCES
# =============================================================================

fusion_1 = grand_mere.combine(chef)
fusion_finale = fusion_1.combine(mere)

# =============================================================================
# ANALYSES
# =============================================================================

grand_mere.print_analysis("Croyance de la Grand-mère")
chef.print_analysis("Croyance du Chef")
mere.print_analysis("Croyance de la Mère")

fusion_finale.print_analysis("Fusion Finale des Croyances")

