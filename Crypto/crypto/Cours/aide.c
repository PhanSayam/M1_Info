#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gmp.h>

/*
    COMPILATION : gcc crypto_tool.c -o crypto_tool -lgmp
    USAGE : ./crypto_tool
*/

// --- 1. TEST DE PRIMALITÉ : MILLER-RABIN ---
// n : nombre à tester, k : nombre d'itérations
int miller_rabin(mpz_t n, int k, gmp_randstate_t state) {
    if (mpz_cmp_ui(n, 2) < 0) return 0;
    if (mpz_cmp_ui(n, 2) == 0 || mpz_cmp_ui(n, 3) == 0) return 1;
    if (mpz_tstbit(n, 0) == 0) return 0; // Pair

    mpz_t d, nmu, a, x;
    mpz_inits(d, nmu, a, x, NULL);
    mpz_sub_ui(nmu, n, 1);
    
    // Trouver d tel que n-1 = 2^s * d
    unsigned long s = 0;
    mpz_set(d, nmu);
    while (mpz_tstbit(d, 0) == 0) {
        mpz_fdiv_q_2exp(d, d, 1);
        s++;
    }

    int is_prime = 1;
    for (int i = 0; i < k; i++) {
        // Choisir a dans [2, n-2]
        mpz_t limit;
        mpz_init_set(limit, n);
        mpz_sub_ui(limit, limit, 4); 
        mpz_urandomm(a, state, limit);
        mpz_add_ui(a, a, 2);
        mpz_clear(limit);

        mpz_powm(x, a, d, n);

        if (mpz_cmp_ui(x, 1) == 0 || mpz_cmp(x, nmu) == 0) continue;

        int composite = 1;
        for (unsigned long r = 1; r < s; r++) {
            mpz_powm_ui(x, x, 2, n);
            if (mpz_cmp(x, nmu) == 0) {
                composite = 0;
                break;
            }
        }
        if (composite) {
            is_prime = 0;
            break;
        }
    }

    mpz_clears(d, nmu, a, x, NULL);
    return is_prime;
}

// --- 2. GÉNÉRATION DE CLÉS RSA ---
void generate_rsa_keys(int bits, mpz_t n, mpz_t e, mpz_t d) {
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, time(NULL));

    mpz_t p, q, phi, p1, q1;
    mpz_inits(p, q, phi, p1, q1, NULL);

    // 1. Générer deux nombres premiers p et q
    do { mpz_urandomb(p, state, bits / 2); } while (!miller_rabin(p, 25, state));
    do { mpz_urandomb(q, state, bits / 2); } while (!miller_rabin(q, 25, state));

    // 2. n = p * q
    mpz_mul(n, p, q);

    // 3. phi(n) = (p-1)(q-1)
    mpz_sub_ui(p1, p, 1);
    mpz_sub_ui(q1, q, 1);
    mpz_mul(phi, p1, q1);

    // 4. Choisir e (souvent 65537) et calculer d = e^-1 mod phi
    mpz_set_ui(e, 65537);
    if (mpz_invert(d, e, phi) == 0) {
        // Si 65537 n'est pas inversible, on recommence (rare)
        generate_rsa_keys(bits, n, e, d);
    }

    mpz_clears(p, q, phi, p1, q1, NULL);
    gmp_randclear(state);
}

// --- 3. ALGORITHME DE NEWTON (RACINE CARRÉE ENTIÈRE) ---
// Utile pour l'attaque de Fermat (facteurs proches)
void integer_sqrt(mpz_t rop, const mpz_t n) {
    if (mpz_cmp_ui(n, 0) == 0) { mpz_set_ui(rop, 0); return; }
    mpz_t x, y, temp;
    mpz_inits(x, y, temp, NULL);
    
    mpz_set_ui(x, 1);
    mpz_mul_2exp(x, x, (mpz_sizeinbase(n, 2) / 2) + 1);

    while (1) {
        mpz_tdiv_q(y, n, x);
        mpz_add(y, y, x);
        mpz_tdiv_q_2exp(y, y, 1);
        if (mpz_cmp(y, x) >= 0) break;
        mpz_set(x, y);
    }
    mpz_set(rop, x);
    mpz_clears(x, y, temp, NULL);
}

/**
 * COMPILATION : gcc rsa_advanced_tool.c -o rsa_advanced_tool -lgmp

gcc fichier.c -o fichier -lgmp

valgrind fichier
 */

// --- 1. THÉORIE DES GROUPES ET ORDRE ---

/* Calcule l'ordre multiplicatif de a modulo n */
void ordre_multiplicatif(mpz_t res, const mpz_t a, const mpz_t n) {
    mpz_t phi, d, temp;
    mpz_inits(phi, d, temp, NULL);
    
    // L'ordre divise phi(n). Pour n=p*q, on peut utiliser (p-1)(q-1)
    // Ici, on suppose n premier pour simplifier l'exemple du TP
    mpz_sub_ui(phi, n, 1); 
    mpz_set(res, phi);

    // On teste les diviseurs de phi de manière exhaustive (ou optimisée)
    for (mpz_set_ui(d, 1); mpz_mul(temp, d, d), mpz_cmp(temp, phi) <= 0; mpz_add_ui(d, d, 1)) {
        if (mpz_divisible_p(phi, d)) {
            mpz_powm(temp, a, d, n);
            if (mpz_cmp_ui(temp, 1) == 0) { mpz_set(res, d); break; }
            
            mpz_divexact(temp, phi, d); // Teste le diviseur complémentaire
            mpz_powm(temp, a, temp, n);
            // ... logique de réduction de l'ordre ...
        }
    }
    mpz_clears(phi, d, temp, NULL);
}

// --- 2. RACINE CARRÉE ET ATTAQUE DE FERMAT ---

/* Racine carrée entière (Newton) pour les très grands nombres */
void racine_carree_entiere(mpz_t res, const mpz_t n) {
    if (mpz_cmp_ui(n, 0) <= 0) { mpz_set_ui(res, 0); return; }
    mpz_t x, y;
    mpz_inits(x, y, NULL);
    mpz_set_ui(x, 1);
    mpz_mul_2exp(x, x, (mpz_sizeinbase(n, 2) / 2) + 1);
    while (1) {
        mpz_tdiv_q(y, n, x);
        mpz_add(y, y, x);
        mpz_tdiv_q_2exp(y, y, 1);
        if (mpz_cmp(y, x) >= 0) break;
        mpz_set(x, y);
    }
    mpz_set(res, x);
    mpz_clears(x, y, NULL);
}

/* Attaque de Fermat : factorise n si |p-q| est faible */
void attaque_fermat(mpz_t p, mpz_t q, const mpz_t n) {
    mpz_t a, b2, b;
    mpz_inits(a, b2, b, NULL);
    racine_carree_entiere(a, n);
    if (mpz_mul(b2, a, a), mpz_cmp(b2, n) < 0) mpz_add_ui(a, a, 1);

    while (1) {
        mpz_mul(b2, a, a);
        mpz_sub(b2, b2, n);
        racine_carree_entiere(b, b2);
        mpz_mul(p, b, b);
        if (mpz_cmp(p, b2) == 0) { // Carré parfait trouvé
            mpz_sub(p, a, b);
            mpz_add(q, a, b);
            break;
        }
        mpz_add_ui(a, a, 1);
    }
    mpz_clears(a, b2, b, NULL);
}

// --- 3. ATTAQUE CCA2 (HOMOMORPHISME) ---

/* Prépare le message aveuglé pour l'attaque CCA2 */
void blind_message(mpz_t c_prime, const mpz_t c, const mpz_t r, const mpz_t e, const mpz_t n) {
    mpz_t re;
    mpz_init(re);
    mpz_powm(re, r, e, n); // r^e mod n
    mpz_mul(c_prime, c, re);
    mpz_mod(c_prime, c_prime, n); // c' = c * r^e mod n
    mpz_clear(re);
}

/* Retire l'aveuglement après déchiffrement */
void unblind_message(mpz_t m, const mpz_t m_prime, const mpz_t r, const mpz_t n) {
    mpz_t r_inv;
    mpz_init(r_inv);
    mpz_invert(r_inv, r, n); // r^-1 mod n
    mpz_mul(m, m_prime, r_inv);
    mpz_mod(m, m, n); // m = m' * r^-1 mod n
    mpz_clear(r_inv);
}

// --- 4. RÉSUMÉ DES RELATIONS RSA ---
/** 
 * Relations de base :
 * - n = p * q
 * - phi(n) = (p-1)(q-1)
 * - d = e^-1 mod phi(n)
 * - c = m^e mod n
 * - m = c^d mod n
 */

int main() {
    mpz_t n, e, d, m, c, r;
    mpz_inits(n, e, d, m, c, r, NULL);

    printf("--- 1. GÉNÉRATION DE CLÉS RSA (1024 bits) ---\n");
    generate_rsa_keys(1024, n, e, d);
    gmp_printf("n : %Zd\ne : %Zd\nd : %Zd\n\n", n, e, d);

    printf("--- 2. CHIFFREMENT / DÉCHIFFREMENT ---\n");
    mpz_set_str(m, "1234567890abcdef", 16);
    gmp_printf("Message original (hex) : %Zx\n", m);

    // Chiffrement : c = m^e mod n
    mpz_powm(c, m, e, n);
    gmp_printf("Cryptogramme : %Zx\n", c);

    // Déchiffrement : r = c^d mod n
    mpz_powm(r, c, d, n);
    gmp_printf("Message déchiffré : %Zx\n\n", r);

    printf("--- 3. SYMBOLE DE JACOBI ---\n");
    mpz_t j_a, j_n;
    mpz_init_set_ui(j_a, 1233);
    mpz_init_set_ui(j_n, 1121);
    int jac = mpz_jacobi(j_a, j_n);
    printf("Jacobi (1233/1121) = %d\n", jac);

    mpz_clears(n, e, d, m, c, r, j_a, j_n, NULL);

    mpz_t n, p, q;
    mpz_inits(n, p, q, NULL);

    // Exemple Fermat
    mpz_set_str(n, "de65b503063f", 16); // Petit exemple pour démonstration
    printf("Tentative factorisation Fermat...\n");
    attaque_fermat(p, q, n);
    gmp_printf("p: %Zd, q: %Zd\n", p, q);

    mpz_clears(n, p, q, NULL);
    return 0;
}