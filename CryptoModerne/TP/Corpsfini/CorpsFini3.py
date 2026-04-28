def add(x,y):
    return [(x[0]+y[0])%5,(x[1]+y[1])%5]

def multbyalpha(x):
    return [(3*x[1])%5,(x[0]+x[1])%5]

#  x^2+4x+2 = 0 -> x2 = -4x -2 = x +3
# x(3x + 2 ) = 3x^2 + 2x = 3(x+3) +2x= 0+4 = [4,0] faire ca avec a0 et a1
print(multbyalpha([3,4])) #[4,0]


def mult(x, y):
    c0 = (x[0] * y[0] + 3 * x[1] * y[1]) % 5
    c1 = (x[0] * y[1] + x[1] * y[0] + x[1] * y[1]) % 5
    return [c0, c1]

print(mult([4, 3],[1, 4])) #[0, 1]

def multbyalpha(x):
    return [(3 * x[2]) % 7, x[0], (x[1] + x[2]) % 7]

def mult(x, y):
    c0 = x[0] * y[0]
    c1 = x[0] * y[1] + x[1] * y[0]
    c2 = x[0] * y[2] + x[1] * y[1] + x[2] * y[0]
    c3 = x[1] * y[2] + x[2] * y[1]
    c4 = x[2] * y[2]
    
    r0 = (c0 + 3 * c3 + 3 * c4) % 7
    r1 = (c1 + 3 * c4) % 7
    r2 = (c2 + c3 + c4) % 7
    
    return [r0, r1, r2]