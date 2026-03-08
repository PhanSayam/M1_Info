import numpy as np

# =============================================================================
# PARTIE 1 : RÉSOLUTION D'ÉQUATIONS f(x) = 0 (DIMENSION 1)
# =============================================================================

# --- Exercice 1 : f(x) = 1/x - a ---
def solve_inverse(a, x0, omega):
    """Point fixe et Newton pour l'inverse [cite: 13, 14, 28]"""
    # Point fixe : x = x + w(1/x - a)
    gw = lambda x: x + omega * (1/x - a) [cite: 13, 14]
    # Newton : x = x(2 - ax)
    newton = lambda x: x * (2 - a * x) [cite: 28]
    return gw, newton

# --- Exercice 2 : f(x) = x^2 - a ---
def solve_sqrt(a):
    """Méthode de Héron (Newton pour racine carrée) [cite: 57, 60]"""
    # x_{n+1} = x_n/2 + a/(2x_n)
    heron = lambda x: 0.5 * x + a / (2 * x) [cite: 60]
    return heron

# --- Exercice 7 & CC1 Ex 1 : f(x) = ln(x) ---
def solve_ln():
    """Point fixe et Newton pour logarithme [cite: 158, 164]"""
    # Point fixe : x = x + w*ln(x)
    gw = lambda x, omega: x + omega * np.log(x) [cite: 158]
    # Newton : x = x(1 - ln(x))
    newton = lambda x: x * (1 - np.log(x)) [cite: 164]
    return gw, newton


# =============================================================================
# PARTIE 2 : OPTIMISATION MULTIVARIABLE (TD2)
# =============================================================================

# --- Exercice 1 : Fonction de Rosenbrock ---
def rosenbrock(x, alpha=1.0):
    """f(x1, x2) = alpha(x2 - x1^2)^2 + (1 - x1)^2 [cite: 171]"""
    return alpha * (x[1] - x[0]**2)**2 + (1 - x[0])**2 [cite: 171]

def grad_rosenbrock(x, alpha=1.0):
    """Gradient de Rosenbrock [cite: 190]"""
    return np.array([
        -4 * alpha * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        2 * alpha * (x[1] - x[0]**2)
    ]) [cite: 190]

# --- Exercice 2 : Ellipsoïde ---
def grad_ellipsoide(x, a, b):
    """Gradient de (x/a)^2 + (y/b)^2 [cite: 218]"""
    return np.array([2*x[0]/(a**2), 2*x[1]/(b**2)]) [cite: 218]


# =============================================================================
# PARTIE 3 : MÉTHODES DE GRADIENT PROJETÉ (PGP) ET NEWTON (CC1)
# =============================================================================

# Données du système linéaire Bx = b (Exercice 3 CC1)
B_MAT = np.array([[14, -34, -18], [-34, 152, 90], [-18, 90, 54]], float) [cite: 379, 432]
B_VEC = np.array([-8, -50, -36], float) [cite: 379, 433]

def f_quad(x):
    """f(x) = 1/2 ||Bx - b||^2 [cite: 381, 481]"""
    return 0.5 * np.linalg.norm(B_MAT @ x - B_VEC)**2 [cite: 481]

def grad_f_quad(x):
    """Gradient : B(Bx - b) [cite: 425, 435]"""
    return B_MAT @ (B_MAT @ x - B_VEC) [cite: 435]

# --- Algorithmes de Descente ---

def pgp_cst(delta, eps=1e-7, nmax=1000000):
    """PGP à pas constant [cite: 436]"""
    x = np.zeros(3)
    for n in range(nmax):
        g = grad_f_quad(x)
        if np.linalg.norm(g) < eps: break
        x -= delta * g [cite: 439]
    return x, n

def pgp_opt(eps=1e-7, nmax=1000000):
    """PGP à pas optimal [cite: 464]"""
    x = np.zeros(3)
    for n in range(nmax):
        g = grad_f_quad(x)
        if np.linalg.norm(g) < eps: break
        # lam = ||g||^2 / ||Bg||^2 [cite: 467]
        lam = (g @ g) / (g @ (B_MAT @ B_MAT @ g)) [cite: 467]
        x -= lam * g [cite: 468]
    return x, n

def pgp_armijo(c=0.5, rho=0.5, eps=1e-7, nmax=1000000):
    """PGP avec recherche linéaire d'Armijo [cite: 482]"""
    x = np.zeros(3)
    for n in range(nmax):
        g = grad_f_quad(x)
        if np.linalg.norm(g) < eps: break
        lam = 1.0
        # Condition d'Armijo [cite: 477, 488]
        while f_quad(x - lam * g) > f_quad(x) - c * lam * (g @ g):
            lam *= rho [cite: 489]
        x -= lam * g [cite: 490]
    return x, n

def newton_system(B, b):
    """Newton pour système linéaire (1 itération) [cite: 393, 396]"""
    # x = B^-1 * b [cite: 400]
    return np.linalg.solve(B, b) [cite: 400]


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("--- CC1 Exercice 3 : Comparaison des méthodes ---")
    
    # Newton
    sol_newton = newton_system(B_MAT, B_VEC)
    print(f"Newton (exact) : {sol_newton}")
    
    # Détermination du delta optimal pour le pas constant (par balayage)
    vals = np.linalg.eigvals(B_MAT @ B_MAT) [cite: 444]
    delta_opt = 2 * vals.min() / (vals.max()**2) # Heuristique document [cite: 447, 501]
    
    sol_cst, it_cst = pgp_cst(delta_opt * 0.9)
    sol_opt, it_opt = pgp_opt()
    sol_arm, it_arm = pgp_armijo()
    
    print(f"Pas Constant : {it_cst} itérations")
    print(f"Pas Optimal  : {it_opt} itérations")
    print(f"Armijo       : {it_arm} itérations")