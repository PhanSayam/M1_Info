import random

def eea(a,p):
    u1=1
    v1=0
    u=0
    v=1
    while p != 0:
        r = a % p
        q = a // p
        a = p
        p = r
        aux1 = u1-q*u
        aux2 = v1-q*v
        u1=u
        v1=v
        u=aux1
        v=aux2
    return [u1,v1,a]

def square_root(a,p):
    if (p&3)==3:
        a1=pow(a,(p+1)//4,p)
    else:
        m = random.randrange(1,p)
        while pow(m,(p-1)//2,p)==1:
            m = random.randrange(1,p)
        t = (p-1)>>1
        s = 1
        while (t&1)==0:
            t = t >> 1
            s = s + 1
        B = pow(a,t,p)
        X = pow(a,(t+1)//2,p)
        z = pow(m,t,p)
        Y = z
        R = s-1
        while R>=1:
            if pow(B, (1 << R-1),p)==1:
                Y = pow(Y,2,p)
            else:
                B = B*(pow(Y,2,p)) % p
                X = X*Y % p
                Y = pow(Y,2,p)
            R = R - 1
        a1 = X
    return a1

def solve_eq(a,p,q):

    # a est un résidu quadratique modulo n ? 
    p_1_div2 = (p-1)//2
    q_1_div2 = (q-1)//2
    test = pow(a,p_1_div2,p)
    test2 = pow(a,q_1_div2,q)

    if (test!= 1 or test2!=1):
        return []
    else : 
        x1 = square_root(a,p)
        z1 = square_root(a,q)

        x2 = -x1 % p
        z2 = -z1 % q

        [u,v,b] = eea(p,q)

        n = p*q

        y1 = (v % p)*q % n
        y2 = (u % q)*p % n

        val1 = x1*y1
        val2 = z1*y2

        r1 = (val1 + val2)%n
        r2 = -r1 %n

        r3 = (val1 - (val2 % q))%n
        r4 = -r3 %n

    return [r1,r2,r3,r4]

#print(solve_eq(1,7,29))
#print(solve_eq(58,17,19))


def solve_multi_system(a, m):
    longueur_m = len(m)
    x = 0
    M = 1
    for mi in m:
        M *= mi

    for i in range(longueur_m):
        Mi = M // m[i]
        [u, v, d] = eea(Mi, m[i])
        k = u % m[i]
        yi = Mi * k
        x = (x + a[i] * yi) % M

    return x


