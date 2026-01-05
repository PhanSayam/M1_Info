import random

def square_root(a,p):
    s = 0
    p_1 = p-1
    m = p_1
    m_1 = m-1
    
    while(m & 1)==0:
        s += 1
        m = m >> 1

    m_1_div_2 = m_1//2

    z = random.randint(1,p_1)
    p_1_div2 = p_1 // 2

    legendre = pow(z,p_1_div2,p)

    while legendre != p_1 :
        z = random.randint(1,p)
        legendre = pow(z,p_1_div2,p)

    c = pow(z,m,p)
    u = pow(a,m,p)
    v = pow(a,m_1_div_2,p)
    i = s-1

    c_2 = pow(c,2,p)

    while(i>=1):
        # u_pow_2_s_1 = pow(u,i)
        if (pow(u,pow(2,i-1),p))==p_1: #remplacer ca
            u = (u*c_2)%p
            v = (v*c)%p
        c = c_2
        i -= i
    return [v,p-v]

print(square_root(4,7))
print(square_root(109102494119399808625840050889440521300365346858060078837046588683349313841497,115792089237316195423570985008687907853269984665640564039457584007913129640233))