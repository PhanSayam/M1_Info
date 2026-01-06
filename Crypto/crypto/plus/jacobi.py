def jacobi(m,n):
    if n == 1:
        return 1 
    m = m % n
    s = 1
    while(m >= 2):
        if (m//2 == 0):
            m = m/2
            nmod8 = n%8
            if (nmod8==3) or (nmod8==5):
                s =-s
        else : 
            m,n = n,m
            if(n%4==3) and (m%4==3):
                s=-s
            m=m%n
    if m==0:
        return 0
    else : 
        return s

""" print(jacobi(122,237)) """

import random 

def solovay_strassen(n,t):
    for i in range(1,t+1):
        a = random.randint(2,n-1)
        # ajouter pgcd
        j = jacobi(a,n)
        if j == 0 : 
            return 0
        else : 
            temp = pow(a,(n-1)//2,n)
            if temp != j%n : 
                return 0
    return 1

print(solovay_strassen(5646542641215435615151651515124,20))
