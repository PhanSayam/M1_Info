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