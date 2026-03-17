import math

dico = {}

p=1934459 
g=762973 
b=1191663

# p=6127307 
# g=4268803 
# b=2108349

#p=1157792987421359 
#g=611838183432519 
#b=1014212026928503

# Phase 1
# m = partie entiere de la racine carrée de p
m = int(math.sqrt(p))

val = 1
dico[val] = 0
for i in range(1,m):
    val = (val * g) % p
    dico[val] = i
    
# Phase 2
j = 0
g_m = pow(g, -m, p)

z = b 
while j <= m :    
    if z in dico : 
        x = m*j + dico[z]
        break 
    z = (z * g_m) % p
    j = j +1   


g_x = pow(g,x,p)
print(z, x)
print("g^x = ", g_x)
print("b   = ", b)


