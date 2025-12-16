def sous_groupe_gen_add(a,n):
    list = []
    temp = a
    while temp != 0 :
        list.append(temp)
        temp = (temp + a)%n
    list.append(0)
    return list

def sous_groupe_gen_mult(a,n):
    list = []
    temp = a
    while temp != 1 :
        list.append(temp)
        temp = (temp * a)%n
    list.append(1)
    return list


# a^j = 1(p)
# def ord(a,p):
#     i = 1
#     temp = 0
#     while(i*i<=(p-1)):
#         if(pow(a,i,p)==1):
#             return i
#         i += 1
#         j = (p-1)//i
#         if(pow(a,j,p)==1):
#             temp = j
#         return temp
    
def ord(a, p):
    n = p - 1
    i = 1
    temp = n
    while i*i <= n:
        if n % i == 0:
            if pow(a, i, p) == 1:
                return i
            j = n // i
            if pow(a, j, p) == 1:
                temp = j
        i += 1
    return temp


# print(ord(9263,19231))
# print(ord(16169,19273))
# print(ord(7159,19231))


# def generateurs(p):
#     list = []
    
#     k = 1
    
        
#     if (pow(g,(p-1)//k,p)!=1):
        
    
    
    
#     return None
    
    