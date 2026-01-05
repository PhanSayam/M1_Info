p = 67108879
g = 3
A = 3412971
B = 2525736

# Retrouver le secret d'Alice x tel que g^x = A [mod p]
x = 0
current = 1
while current != A:
    x += 1
    current = (current * g) % p

# Clé partagée K = B^x [mod p]
cle_partagee = pow(B, x, p)
print(f"cle_partagee={cle_partagee}")