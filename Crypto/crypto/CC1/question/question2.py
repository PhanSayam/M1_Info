
def cle_rsa_valid(n, e):
    """
    Vérifie qu'une clé RSA (n, e) est valide :
    - n doit être produit de deux nombres premiers distincts
    - e premier avec phi(n) = (p-1)*(q-1)
    """
    tab = decompose(n)  # nécessite la fonction decompose du TP1
    if len(tab) == 2 and tab[0][1] == 1 and tab[1][1] == 1:
        p = tab[0][0]
        q = tab[1][0]
        phi = (p-1)*(q-1)
        if pgcd(e, phi) == 1:  # nécessite la fonction pgcd du TP1
            print("clé valide")
            return True
        else:
            print("e non premier avec phi(n)")
            return False
    else:
        print("n n'est pas un produit de deux nombres premiers")
        return False

import random
from sage.all import *

p = 4171849679533027504677776769862406473833407270227837441302815640277772901915313574263597827907
q = 5214812099416284380847220962328008092291759087784796801628519550347216127394141967829497282999
n = p*q
phi_n = n*prod([Integer(1) - Integer(1)/p for p in prime_divisors(n)])
e = random.randint(1,phi_n)
d = xgcd(e,phi_n)

cle_rsa_valid(n,e)


