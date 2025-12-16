def inverse(a,p):
    u = 1
    v = 0
    u1 = 0
    v1 = 1

    while p!=0 :
        u0 = u
        v0 = v
        
        u = u1
        v = v1
        
        q = a//p

        u1 = u0 - q*u1
        v1 = v0 - q*v1
        
        temp = a % p
        a = p
        p = temp

    return u%p

def inverse(a,p):
    orig_p = p
    u = 1
    v = 0
    u1 = 0
    v1 = 1

    while p != 0:
        u0 = u
        v0 = v
        
        u = u1
        v = v1
        
        q = a // p

        u1 = u0 - q*u1
        v1 = v0 - q*v1
        
        temp = a % p
        a = p
        p = temp

    return u % orig_p

print(inverse(1368,1531))

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

        if i*i > n and n > 1:
            facteurs.append([n, 1])
            break

    return facteurs