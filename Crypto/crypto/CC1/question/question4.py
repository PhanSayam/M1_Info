c = 7648206613298249017496999958367362009237338049829306609893105166080044950117196556524261951636429915926756330703606036931978673542392960427312553803492971672836719943074283855662828673688
e = 3093875975443154419198142401204087736373750493351542332362554257
p = 4171849679533027504677776769862406473833407270227837441302815640277772901915313574263597829749
q = 5214812099416284380847220962328008092291759087784796801628519550347216127394141967829497286481

def pgcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

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

def euler_phi(n):
    phi = 1
    for p, e in decompose(n):
        phi *= p**e - p**(e-1)
    return phi

def puissance(x, y, n):
    z = 1
    x = x % n
    while y != 0:
        z = z * (x if (y & 1) else 1) % n
        x = (x * x) % n
        y = y >> 1
    return z

def rsa_dechiffrement(c, d, n):
    return puissance(c, d, n)


from sage.all import *
n = p*q
phi = euler_phi(n)
d = xgcd(e,phi)

print(d)
print(rsa_dechiffrement(c,d,n))