# def table_alpha(P):
#     L = [1]
#     bit_poid_fort = 1 << len(bin(P)[3:]) #0b1000
#     temp = P ^ bit_poid_fort
#     binary_temp = bin(temp)
    
#     degre = len(bin(bit_poid_fort))-2 
    
#     i = 1
#     # for i in range(degre):
#     #     L[i] = 1 << i
#     #     if i >= degre-1:
#     #         L[i] = multbyalpha(temp,P)
#     #     i+=1
    
#     while(L[i] != L[0]):
#         L[i] = 1 << i
#         if i >= degre-1:
#             L[i] = multbyalpha(temp,P)
#         i+=1
    
#     return L 

# table_alpha(13)

def table_alpha(P):
    L = [1]
    element_courant = 1
    
    while True:
        suivant = multbyalpha(element_courant, P)
        if suivant == 1:
            break
        L.append(suivant)
        element_courant = suivant
    return L

def table_log(P):
    m = len(bin(P)) - 3
    taille = 1 << m
    L = [-1] * taille
    val = 1
    for i in range(taille - 1):
        L[val] = i
        val = multbyalpha(val, P)
    return L


def multiplie(x,y,P):
    if (x==0) or (y ==0) :
        return 0
        
    exp_a = log_table(x)
    exp_b = log_table(y)
    
    m = len(bin(P))-2
    
    somme = (exp_a + exp_b) % (1 << m)-1
    return alpha_table(somme)


def evalue(Q, y, P):
    resultat = Q[-1]
    for i in range(len(Q) - 2, -1, -1):
        resultat = multiplie(resultat, y, P)
        resultat = resultat ^ Q[i]
        
    return resultat

def multbyalpha(b, f, p):
    m = len(b)
    coeff_high = b[m-1]
    res = [0] * m
    
    for i in range(m):
        if i == 0:
            shifted = 0
        else:
            shifted = b[i-1]
        res[i] = (shifted - coeff_high * f[i]) % p
        
    return res