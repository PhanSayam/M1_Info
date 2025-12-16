def pgcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

# Extended Euclidean Algorithm
def euclide_e(a,n):
    u = 1
    v = 0
    u1 = 0
    v1 = 1

    while n!=0 :
        u0 = u
        v0 = v
        
        u = u1
        v = v1
        
        q = a//n

        u1 = u0 - q*u1
        v1 = v0 - q*v1
        
        temp = a % n
        a = n
        n = temp

    return [u,v,a]

# Calcul de l'inverse de a modulo p
def inverse(a, p):
    orig = p
    u, v = 1, 0
    u1, v1 = 0, 1
    while p != 0:
        q = a // p

        u, u1 = u1, u - q * u1
        v, v1 = v1, v - q * v1

        a, p = p, a % p

    return u % orig

def euler_phi(n):
    c = 0
    for i in range(1, n):
        a, b = i, n
        while b != 0:
            a, b = b, a % b
        if a == 1:
            c += 1
    return c

# Décomposition en facteurs premiers
def decompose(n):
    facteurs = []
    e = 0
    while n % 2 == 0:
        n //= 2
        e += 1
    if e > 0:
        facteurs.append([2, e])

    i = 3
    while n > 1:
        e = 0
        while n % i == 0:
            n //= i
            e += 1
        if e > 0:
            facteurs.append([i, e])
        i += 2
        if i * i > n and n > 1:
            facteurs.append([n, 1])
            break

    return facteurs

# Calcul de la fonction phi d'Euler
def euler_phi(n):
    phi = 1
    for p, e in decompose(n):
        phi *= p**e - p**(e-1)
    return phi

# Calcul de l'ordre de a dans le groupe additif mod n
def ord(a, n):
    x = a % n
    y = n
    while x != 0:
        t = x
        x = y % x
        y = t
    return n // y

def generateurs(n):
    a = 1
    gens = []
    while a < n:
        if pgcd(a, n) == 1:
            gens.append(a)
        a += 1
    return gens

def sous_groupe_gen_add(a, n):
    elems = []
    temp = a % n
    while temp != 0:
        elems.append(temp)
        temp = (temp + a) % n
    elems.append(0)
    return elems

# Calcul des éléments du sous-groupe engendré par a dans le groupe multiplicatif mod n
def sous_groupe_gen_mult(a, n):
    elems = []
    temp = a % n
    while temp != 1:
        elems.append(temp)
        temp = (temp * a) % n
    elems.append(1)
    return elems

# Calcul de l'ordre de a dans le groupe multiplicatif mod p
def ord(a, p):
    n = p - 1
    i = 1
    res = n
    while i * i <= n:
        if n % i == 0:
            if pow(a, i, p) == 1:
                return i
            j = n // i
            if pow(a, j, p) == 1:
                res = j
        i += 1
    return res

# Calcul des générateurs du groupe multiplicatif mod p
def generateurs(p):
    gens = []
    for g in range(1, p):
        ok = True
        for k in decompose(p-1):
            if pow(g, (p-1)//k, p) == 1:
                ok = False
                break
        if ok:
            gens.append(g)
    return gens

from random import *

# Génération aléatoire d'un générateur du groupe multiplicatif mod p
def generateur(p):
    g = randrange(2, p)
    test1 = pow(g, 2, p) - 1
    test2 = pow(g, (p-1)//2, p) - 1
    while test1 * test2 == 0:
        g = randrange(2, p)
        test1 = pow(g, 2, p) - 1
        test2 = pow(g, (p-1)//2, p) - 1
    return g

# Calcul de la puissance mod n
def puissance(x, y, n):
    z = 1
    x = x % n
    while y != 0:
        z = z * (x if (y & 1) else 1) % n
        x = (x * x) % n
        y = y >> 1
    return z


# Calcul du symbole de Jacobi (m/n)
def jacobi(m, n):
    j = 1
    m = m % n
    while m != 0:
        # retirer un facteur de 2 à chaque tour
        if m % 2 == 0:
            m //= 2
            r = n % 8
            j *= 1 if r in (1,7) else -1
        else:
            # réciprocité quadratique
            m, n = n, m
            if m % 4 == 3 and n % 4 == 3:
                j = -j
            m = m % n
    return j if n == 1 else 0

# Calcul du symbole de Jacobi (n/q) où q est impair
def jacobi2(n, q):
    S = 1
    n = n % q

    if n == 0:
        return 0
    if n == 1:
        return 1

    # Extraire les facteurs de 2
    if n % 2 == 0:
        e = 0
        while n % 2 == 0:
            e += 1
            n //= 2

        if e % 2 == 1:
            mod_q_8 = q % 8
            if mod_q_8 == 3 or mod_q_8 == 5:
                S = -S

    # Décomposition en facteurs premiers impairs
    p = 3
    m = n
    while m > 1:
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1

            if e % 2 == 1:
                if p % 4 == 3 and q % 4 == 3:
                    S = -S
                S = S * jacobi2(q % p, p)
        p += 2

    return S


# Test de primalité de Solovay-Strassen
def solovay_strassen(n, t):
    if n < 2:
        return 0
    if n == 2:
        return 1
    if n % 2 == 0:
        return 0

    a = 2
    for _ in range(t):
        jac = jacobi(a, n) % n
        # jac = 0 détecte automatiquement si a et n ne sont pas premiers entre eux
        if jac == 0 or pow(a, (n - 1)//2, n) != jac:
            return 0
        a += 1
        if a == n:
            a = 2
    return 1


# Calcul des résidus quadratiques et non quadratiques mod n
def residu_quadratique(n):
    l_qr = []
    l_nqr = []

    for i in range(1, n):
        puissance = (n - 1) // 2
        calcul = pow(i, puissance, n)

        if calcul == 1:
            l_qr.append(i)
        else:
            l_nqr.append(i)

    return l_qr, l_nqr



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

# Chiffrement RSA
def rsa_chiffrement(m, e, n):
    """
    m : entier message < n
    e : exposant public
    n : module
    renvoie le cryptogramme c
    """
    return pow(m, e, n)

# Déchiffrement RSA
def rsa_dechiffrement(c, d, n):
    """
    c : cryptogramme
    d : exposant privé
    n : module
    renvoie le message m
    """
    return pow(c, d, n)