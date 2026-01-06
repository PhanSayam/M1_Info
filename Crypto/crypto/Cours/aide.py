"""
BOÎTE À OUTILS : ARITHMÉTIQUE ET CRYPTOGRAPHIE (RSA, PRIMALITÉ, GROUPES)
Ce module implémente les algorithmes fondamentaux sans dépendances externes.
"""

import random

# =============================================================================
# 1. ARITHMÉTIQUE DE BASE
# =============================================================================

def pgcd(a, b):
    """
    Calcule le Plus Grand Commun Diviseur (Algorithme d'Euclide).
    Propriété : pgcd(a, b) = pgcd(b, a % b).
    Complexité : O(log(min(a, b))).
    """
    while b != 0:
        a, b = b, a % b
    return abs(a)

def euclide_etendu(a, b):
    """
    Algorithme d'Euclide Étendu.
    Trouve (u, v, g) tels que : a*u + b*v = pgcd(a, b) = g.
    Essentiel pour trouver l'inverse modulaire.
    """
    u, v, r = 1, 0, a
    u_prime, v_prime, r_prime = 0, 1, b
    
    while r_prime != 0:
        q = r // r_prime
        r, r_prime = r_prime, r - q * r_prime
        u, u_prime = u_prime, u - q * u_prime
        v, v_prime = v_prime, v - q * v_prime
    
    return u, v, r

def inverse_modulaire(a, n):
    """
    Calcule x tel que (a * x) % n == 1.
    L'inverse existe si et seulement si pgcd(a, n) == 1.
    """
    u, v, g = euclide_etendu(a, n)
    if g != 1:
        raise ValueError(f"L'inverse de {a} mod {n} n'existe pas car pgcd != 1")
    return u % n

# =============================================================================
# 2. FACTORISATION ET FONCTION D'EULER
# =============================================================================

def decompose(n):
    """
    Décomposition en facteurs premiers.
    Renvoie une liste de tuples (p, e) où n = produit(p^e).
    Algorithme par divisions successives optimisé jusqu'à sqrt(n).
    """
    facteurs = []
    temp = abs(n)
    
    # Cas du facteur 2
    if temp % 2 == 0:
        count = 0
        while temp % 2 == 0:
            count += 1
            temp //= 2
        facteurs.append((2, count))
    
    # Facteurs impairs
    d = 3
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                count += 1
                temp //= d
            facteurs.append((d, count))
        d += 2
    
    if temp > 1:
        facteurs.append((temp, 1))
    return facteurs

def euler_phi(n):
    """
    Indicateur d'Euler phi(n).
    Nombre d'entiers k < n tels que pgcd(k, n) == 1.
    Formule : phi(n) = n * produit(1 - 1/p) pour chaque p diviseur de n.
    """
    if n == 1: return 1
    res = n
    # On utilise la décomposition pour appliquer la formule
    for p, e in decompose(n):
        res -= res // p
    return res

# =============================================================================
# 3. SYMBOLE DE JACOBI ET PRIMALITÉ
# =============================================================================

def jacobi(a, n):
    """
    Calcule le symbole de Jacobi (a/n).
    Généralisation du symbole de Legendre (a/p).
    Propriétés : 
    1. (2/n) = 1 si n = 1,7 mod 8 ; -1 si n = 3,5 mod 8.
    2. Loi de réciprocité quadratique si a, n impairs.
    """
    if n % 2 == 0: return 0
    a %= n
    res = 1
    while a != 0:
        # Sortir les puissances de 2
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                res = -res
        # Réciprocité quadratique
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            res = -res
        a %= n
    return res if n == 1 else 0

def solovay_strassen(n, t=10):
    """
    Test de primalité probabiliste de Solovay-Strassen.
    Basé sur le critère d'Euler : a^((n-1)/2) = (a/n) mod n.
    Si n est composé, la probabilité de passer le test est < 1/2^t.
    """
    if n < 2: return False
    if n in (2, 3): return True
    if n % 2 == 0: return False

    for _ in range(t):
        a = random.randint(2, n - 1)
        j = jacobi(a, n)
        if j == 0: return False # pgcd(a, n) > 1
        
        # Calcul de a^((n-1)/2) mod n
        exponent = pow(a, (n - 1) // 2, n)
        if exponent != (j % n):
            return False
    return True

# =============================================================================
# 4. THÉORIE DES GROUPES (ORDRE ET GÉNÉRATEURS)
# =============================================================================

def ordre_multiplicatif(a, n):
    """
    Plus petit k > 0 tel que a^k = 1 mod n.
    k doit diviser phi(n).
    """
    if pgcd(a, n) != 1: return None
    phi = euler_phi(n)
    diviseurs = []
    # Trouver les diviseurs de phi(n)
    for i in range(1, int(phi**0.5) + 1):
        if phi % i == 0:
            diviseurs.append(i)
            if i*i != phi:
                diviseurs.append(phi // i)
    diviseurs.sort()
    
    for d in diviseurs:
        if pow(a, d, n) == 1:
            return d
    return phi

def est_generateur(g, p):
    """
    Vérifie si g engendre (Z/pZ)*.
    Condition : g^((p-1)/q) != 1 mod p pour tout q diviseur premier de p-1.
    """
    phi = p - 1
    for q, e in decompose(phi):
        if pow(g, phi // q, p) == 1:
            return False
    return True

def trouver_generateur(p):
    """Trouve un générateur (racine primitive) de (Z/pZ)*."""
    if p == 2: return 1
    while True:
        g = random.randint(2, p - 1)
        if est_generateur(g, p):
            return g

# =============================================================================
# 5. CRYPTOGRAPHIE RSA
# =============================================================================

def cle_rsa_valid(p, q, e):
    """
    Vérifie la validité d'une clé RSA.
    p, q doivent être premiers et e doit être premier avec phi(n).
    """
    if not (solovay_strassen(p) and solovay_strassen(q)):
        return False, "p ou q n'est pas premier"
    
    phi = (p - 1) * (q - 1)
    if pgcd(e, phi) != 1:
        return False, "e n'est pas premier avec phi(n)"
    
    return True, "Clé valide"

def rsa_chiffrement(m, e, n):
    """ c = m^e mod n """
    return pow(m, e, n)

def rsa_dechiffrement(c, d, n):
    """ m = c^d mod n """
    return pow(c, d, n)

# =============================================================================
# 1. OUTILS POUR LA GÉNÉRATION DE CLÉS RSA
# =============================================================================

def racine_carree_entiere(n):
    """
    Calcule la partie entière de la racine carrée de n (Méthode de Newton).
    Indispensable pour manipuler des nombres dépassant les capacités de math.sqrt().
    """
    if n < 0: raise ValueError
    if n == 0: return 0
    x = 2**(n.bit_length() // 2 + 1)
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y

def miller_rabin(n, k=40):
    """
    Test de primalité de Miller-Rabin (plus robuste que Solovay-Strassen).
    Probabilité d'erreur < (1/4)^k.
    """
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0: return False

    # Trouver r et d tels que n - 1 = 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generer_premier(bits):
    """Génère un nombre premier aléatoire de la taille spécifiée."""
    while True:
        p = random.getrandbits(bits)
        # S'assurer qu'il est impair et a la bonne taille
        p |= (1 << bits - 1) | 1
        if miller_rabin(p):
            return p

# =============================================================================
# 2. ATTAQUES CLASSIQUES RSA
# =============================================================================

def factorisation_fermat(n):
    """
    Attaque de Fermat : factorise n si p et q sont trop proches.
    Utilisé dans la Question 5 de votre examen.
    """
    a = racine_carree_entiere(n)
    if a * a < n:
        a += 1
    # On limite les itérations pour le TP
    for _ in range(1000000):
        b2 = a*a - n
        b = racine_carree_entiere(b2)
        if b*b == b2:
            p = a - b
            q = a + b
            return p, q
        a += 1
    return None

# =============================================================================
# 3. GESTION DES FORMATS ET CCA2
# =============================================================================

def hex_to_int(h):
    """Convertit une chaîne hexadécimale (avec ou sans 0x) en entier."""
    return int(h, 16)

def int_to_hex(i):
    """Convertit un entier en chaîne hexadécimale."""
    return hex(i)

def attaque_oracle_cca2(c, e, n, oracle_func):
    """
    Simule la logique de l'attaque CCA2.
    Propriété : (c * r^e)^d = c^d * r = m * r [mod n]
    """
    r = 2
    c_blinded = (c * pow(r, e, n)) % n
    
    # m_blinded est la valeur renvoyée par l'oracle
    m_blinded = oracle_func(c_blinded)
    
    # Retirer le facteur d'aveuglement
    m = (m_blinded * pow(r, -1, n)) % n
    return m

def crt(a1, n1, a2, n2):
    u, v, g = euclide_etendu(n1, n2)
    if g != 1:
        raise ValueError
    return (a1 * v * n2 + a2 * u * n1) % (n1 * n2)

def rsa_dechiffrement_crt(c, p, q, d):
    dp = d % (p - 1)
    dq = d % (q - 1)
    mp = pow(c, dp, p)
    mq = pow(c, dq, q)
    return crt(mp, p, mq, q)

def hastad_attack(c_list, n_list, e):
    x = 0
    N = 1
    for n in n_list:
        N *= n
    for c, n in zip(c_list, n_list):
        m = N // n
        inv = inverse_modulaire(m, n)
        x += c * inv * m
    x %= N
    return racine_entiere_e(x, e)

def rsa_vulnerable_fermat(p, q, bound=1_000_000):
    return abs(p - q) < bound

def generer_cle_rsa(bits=2048, e=65537):
    while True:
        p = generer_premier(bits // 2)
        q = generer_premier(bits // 2)
        if p == q:
            continue
        phi = (p - 1) * (q - 1)
        if pgcd(e, phi) == 1:
            d = inverse_modulaire(e, phi)
            return (e, p*q), (d, p, q)

def is_quadratic_residue_prime(a, p):
    if p < 2 or p % 2 == 0:
        raise ValueError
    return pow(a, (p - 1) // 2, p) == 1
def sqrt_mod_prime(a, p):
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
        return None

    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)

    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    c = pow(z, q, p)
    x = pow(a, (q + 1) // 2, p)
    t = pow(a, q, p)
    m = s

    while t != 1:
        i = 1
        temp = pow(t, 2, p)
        while temp != 1:
            temp = pow(temp, 2, p)
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        x = (x * b) % p
        t = (t * b * b) % p
        c = (b * b) % p
        m = i
    return x
def roots_mod_prime(a, e, p):
    res = []
    for x in range(p):
        if pow(x, e, p) == a % p:
            res.append(x)
    return res
def eth_root_mod_prime(a, e, p):
    inv = pow(e, -1, p - 1)
    return pow(a, inv, p)
def miller_rabin_det(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    if n in small_primes:
        return True
    for p in small_primes:
        if n % p == 0:
            return False

    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True
def miller_rabin_witness(a, n):
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    x = pow(a, d, n)
    if x in (1, n - 1):
        return False
    for _ in range(r - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return False
    return True
def is_qr_mod_composite(a, p, q):
    return (
        pow(a, (p - 1) // 2, p) == 1 and
        pow(a, (q - 1) // 2, q) == 1
    )
def sqrt_mod_rsa(a, p, q):
    r1 = sqrt_mod_prime(a, p)
    r2 = sqrt_mod_prime(a, q)
    if r1 is None or r2 is None:
        return []

    def crt(a1, n1, a2, n2):
        u, v, _ = euclide_etendu(n1, n2)
        return (a1 * v * n2 + a2 * u * n1) % (n1 * n2)

    roots = []
    for s1 in [r1, (-r1) % p]:
        for s2 in [r2, (-r2) % q]:
            roots.append(crt(s1, p, s2, q))
    return roots
def rabin_encrypt(m, n):
    return pow(m, 2, n)

# =============================================================================
# 4. RÉSUMÉ DES RELATIONS RSA (COURS)
# =============================================================================
"""
RELATIONS FONDAMENTALES :
1. n = p * q
2. phi(n) = (p-1) * (q-1)
3. e * d ≡ 1 [mod phi(n)]  => d = pow(e, -1, phi)
4. Chiffrement : c = m^e [mod n]
5. Déchiffrement : m = c^d [mod n]

ÉCHECS DE SÉCURITÉ COURANTS :
- p et q trop proches => Attaque de Fermat (Question 5).
- Exposant e trop petit => Attaque par racine n-ième.
- Oracle de déchiffrement disponible => Attaque CCA2 (Question 7).


RELATIONS FONDAMENTALES :
1. n = p · q
   Modulus public, difficulté de factorisation = hypothèse de sécurité centrale.

2. φ(n) = (p − 1)(q − 1)
   Ordre du groupe multiplicatif (Z/nZ)* lorsque n est semi-premier.

3. e · d ≡ 1 [mod φ(n)]
   d est l’inverse modulaire de e dans Z/φ(n)Z.
   Condition nécessaire et suffisante pour la correction du déchiffrement.

4. Chiffrement :
   c ≡ m^e [mod n]

5. Déchiffrement :
   m ≡ c^d [mod n]

6. Correction mathématique :
   m^{ed} ≡ m [mod p] et [mod q] ⇒ [mod n]
   Justifié par le théorème d’Euler + CRT.

7. Version CRT (optimisation) :
   d_p = d mod (p−1), d_q = d mod (q−1)
   Accélération ≈ ×4 du déchiffrement.

ERREURS DE GÉNÉRATION :
- p ou q trop petits
- p ≈ q  → Fermat
- p−1 ou q−1 lisse → Pollard p−1
- Réutilisation de p ou q → attaque GCD massive

ERREURS SUR e :
- e trop petit (e = 3, 5) + message non paddé → attaque par racine
- Même message, même e, modules différents → attaque de Hastad

ERREURS D’IMPLÉMENTATION :
- Oracle de déchiffrement → CCA2
- Temps d’exécution variable → side-channel timing
- CRT mal protégé → attaque de Bellcore (fault attack)

ERREURS CRYPTOGRAPHIQUES :
- Absence de padding → RSA déterministe
- Padding maison → vulnérable (Bleichenbacher)

PADDING STANDARD :
- OAEP (Optimal Asymmetric Encryption Padding)
  Rend RSA IND-CPA sûr dans le modèle de l’oracle aléatoire.

SIGNATURE RSA :
- Signature brute : s = m^d mod n (interdite)
- Standard : RSA-PSS

HYBRIDATION :
- RSA ne chiffre pas des gros messages
- Utilisé pour échanger une clé symétrique (AES)

"""

# =============================================================================
# EXEMPLE D'APPLICATION POUR LE TP
# =============================================================================

if __name__ == "__main__":
    # 1. Génération rapide
    p = generer_premier(512)
    q = generer_premier(512)
    n = p * q
    print(f"n généré ({n.bit_length()} bits)")

    # 2. Test de Fermat sur un cas vulnérable
    p_proche = generer_premier(256)
    # q est le premier juste après p
    q_proche = p_proche + 2
    while not miller_rabin(q_proche):
        q_proche += 2
    
    n_faible = p_proche * q_proche
    print("\n--- Test Attaque Fermat ---")
    res = factorisation_fermat(n_faible)
    if res:
        print(f"Succès ! p trouvé : {res[0] == p_proche}")

    print("--- Test Arithmétique ---")
    print(f"Inverse de 7 mod 26 : {inverse_modulaire(7, 26)}")
    print(f"Facteurs de 133 : {decompose(133)}")
    print(f"Phi(133) : {euler_phi(133)}")
    
    print("\n--- Test Primalité (Solovay-Strassen) ---")
    p_test = 104729 # Un nombre premier connu
    print(f"Est-ce que {p_test} est premier ? {solovay_strassen(p_test)}")
    
    print("\n--- Test RSA ---")
    p, q = 61, 53
    n = p * q
    e = 17
    phi = (p-1)*(q-1)
    d = inverse_modulaire(e, phi)
    
    msg = 42
    chiffre = rsa_chiffrement(msg, e, n)
    clair = rsa_dechiffrement(chiffre, d, n)
    
    print(f"Message original : {msg}")
    print(f"Message chiffré : {chiffre}")
    print(f"Message déchiffré : {clair}")
    
    
    