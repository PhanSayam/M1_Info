def add(x,y):
    return [(x[0]+y[0])%5,(x[1]+y[1])%5]

def multbyalpha(x):
    return [(3*x[1])%5,(x[0]+x[1])%5]

#  x^2+4x+2 = 0 -> x2 = -4x -2 = x +3
# x(3x + 2 ) = 3x^2 + 2x = 3(x+3) +2x= 0+4 = [4,0] faire ca avec a0 et a1
print(multbyalpha([3,4])) #[4,0]


def mult(x,y):
    return 

print(mult([4, 3],[1, 4])) #[0, 1]