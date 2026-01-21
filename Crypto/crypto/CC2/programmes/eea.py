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