

def pgcd(a,b):
    while b > 0 :
        temp = a 
        a = b
        b = temp % b

    return a

print(pgcd(13,21))

print(54//39)
def euclide_e(a,n):
    u = 1
    v = 0
    u1 = 0
    v1 = 1

    while n!=0 :
        u0 = u
        v0 = v
        
        u = u1
        v = v1
        
        q = a//n

        u1 = u0 - q*u1
        v1 = v0 - q*v1
        
        temp = a % n
        a = n
        n = temp

    return [u,v,a]

print(euclide_e(54,39))


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

    return u%a

print(inverse(1301,1597))

def euler_phi(n):
    j=0
    for i in range(1,n):
        a = i
        b = n
        while b > 0 :
            temp = a 
            a = b
            b = temp % b
            if b == 1 :
                j+=1
    return j
euler_phi(9)

print(2%2)

def decompose(n):
    list = []
    occurence = 0
    while(n%2 == 0):
        n = n//2
        occurence+=1
    if occurence != 0 :
        list.append([2,occurence])
        
    i = 3
    
    while n <= 1 :
        occurence = 0
        while(n%i == 0) :
            n = n//i
            occurence+=1
        list.append([i,occurence])
        i += 2
    return list

decompose(500)



