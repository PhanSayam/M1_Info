
# def multbyalpha(b,f):
#     Fbinaire = bin(f)
#     temp = b << 1
#     Bbinaire = bin(temp)
#     taille = len(Fbinaire)
    
#     taille_manquant = len(Fbinaire[2:]) - len(Bbinaire[2:])
    
#     aux = Bbinaire[2:]
#     print(0 | int(aux,2))
#     if (aux[0]=="1"):
#         aux = int(aux,2) ^ int(Fbinaire[2:],2)
#     return aux

def multbyalpha(b, f):
    m = len(bin(f)) - 3
    res = b << 1
    if (res >> m) & 1:
        res = res ^ f
        
    return res
print(multbyalpha(3,13))

# def multiplication(b,c,f):
#     aux = b.copy()
    
#     while i < len(c): 
#         if c[i]!=0:
#             somme = somme ^ aux
#         i+=1
#         aux = multbyalpha(b,f)
#     return somme

def multiplication(b, c, f):
    somme = 0
    while c > 0:
        if (c & 1):
            somme = somme ^ b
        b = multbyalpha(b, f)
        c = c >> 1
        
    return somme