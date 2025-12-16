import random

def rabin_miller(n,k):
    s = 0
    n = int(n)
    n_1 = n-1
    t = n_1 
    pgcd = 2
    
    while(t & 1)==0:
        s += 1
        t = t >> 1

    
    for _ in range(k):
        a = random.randint(2,n_1)
        
        temp_a, temp_n = a, n 
        while temp_n != 0:
            temp_a, temp_n = temp_n, temp_a % temp_n        
        while temp_a != 1:
            a = random.randint(2, n_1)
            temp_a, temp_n = a, n 
            while temp_n != 0:
                temp_a, temp_n = temp_n, temp_a % temp_n
            
        at = pow(a,t,n)

        if(at!=1) and (at!=n_1): 
            premier = False
            for _ in range(s-1):
                at = (at**2) % n
                if at == n_1:
                    premier = True
            if not premier:
                return False
    return True
    
    
print(rabin_miller(2461621572185378719489232151655868788959601584166591523174513,22))