#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <gmp.h>
#include "../LogarithmeDiscret/c/hashutils.h"

static void afficher(const char *label, const mpz_t x) {
    gmp_printf("%s = %Zd\n", label, x);
}

static void free_hashtable(HashTable ht, unsigned int size) {
    for (unsigned int i = 0; i < size; i++) {
        Maillon *cur = ht[i];
        while (cur != NULL) {
            Maillon *next = cur->next;
            free(cur);
            cur = next;
        }
    }
    free(ht);
}

static int nombre_solutions(const mpz_t a, const mpz_t b, const mpz_t n) {
    mpz_t g;
    mpz_init(g);
    mpz_gcd(g, a, n);
    int nb = mpz_divisible_p(b, g) ? (int)mpz_get_ui(g) : 0;
    mpz_clear(g);
    return nb;
}

static int resoudre_lineaire(mpz_t x0, const mpz_t a, const mpz_t b, const mpz_t n) {
    mpz_t g, a_red, b_red, n_red, inv;
    mpz_init(g);
    mpz_init(a_red);
    mpz_init(b_red);
    mpz_init(n_red);
    mpz_init(inv);

    mpz_gcd(g, a, n);
    if (!mpz_divisible_p(b, g)) {
        mpz_clear(g);
        mpz_clear(a_red);
        mpz_clear(b_red);
        mpz_clear(n_red);
        mpz_clear(inv);
        return 0;
    }

    mpz_divexact(a_red, a, g);
    mpz_divexact(b_red, b, g);
    mpz_divexact(n_red, n, g);

    if (!mpz_invert(inv, a_red, n_red)) {
        mpz_clear(g);
        mpz_clear(a_red);
        mpz_clear(b_red);
        mpz_clear(n_red);
        mpz_clear(inv);
        return 0;
    }

    mpz_mul(x0, b_red, inv);
    mpz_mod(x0, x0, n_red);

    int nb = (int)mpz_get_ui(g);
    mpz_clear(g);
    mpz_clear(a_red);
    mpz_clear(b_red);
    mpz_clear(n_red);
    mpz_clear(inv);
    return nb;
}

static void square_and_multiply(mpz_t resultat, const mpz_t base, const mpz_t expo, const mpz_t mod) {
    mpz_t z, x, j;
    mpz_init_set_ui(z, 1);
    mpz_init_set(x, base);
    mpz_init_set(j, expo);

    while (mpz_cmp_ui(j, 0) != 0) {
        if (mpz_odd_p(j)) {
            mpz_mul(z, z, x);
            mpz_mod(z, z, mod);
        }
        mpz_mul(x, x, x);
        mpz_mod(x, x, mod);
        mpz_fdiv_q_2exp(j, j, 1);
    }

    mpz_set(resultat, z);
    mpz_clear(z);
    mpz_clear(x);
    mpz_clear(j);
}

static unsigned long bsgs(const mpz_t g, const mpz_t beta, const mpz_t p) {
    mpz_t temp;
    mpz_init(temp);
    mpz_sqrt(temp, p);
    unsigned long m = mpz_get_ui(temp) + 1;
    mpz_clear(temp);

    unsigned int table_size = (unsigned int)(2 * m + 1);
    HashTable ht = calloc(table_size, sizeof(Maillon *));
    if (ht == NULL) {
        fprintf(stderr, "Erreur allocation table\n");
        exit(1);
    }

    mpz_t aux;
    mpz_init_set_ui(aux, 1);
    hash_insert(&ht, mpz_get_ui(aux), 0, table_size);

    for (unsigned long i = 1; i < m; i++) {
        mpz_mul(aux, aux, g);
        mpz_mod(aux, aux, p);
        hash_insert(&ht, mpz_get_ui(aux), i, table_size);
    }

    mpz_t ginvm;
    mpz_init(ginvm);
    mpz_powm_ui(ginvm, g, m, p);
    if (!mpz_invert(ginvm, ginvm, p)) {
        free_hashtable(ht, table_size);
        mpz_clear(aux);
        mpz_clear(ginvm);
        return 0;
    }

    mpz_t z;
    mpz_init_set(z, beta);

    unsigned long logd = 0;
    int trouve = 0;
    unsigned long i_match = 0;

    for (unsigned long j = 0; j <= m && !trouve; j++) {
        unsigned long val = mpz_get_ui(z);
        if (hash_find(&ht, val, &i_match, table_size)) {
            logd = i_match + j * m;
            trouve = 1;
        } else {
            mpz_mul(z, z, ginvm);
            mpz_mod(z, z, p);
        }
    }

    free_hashtable(ht, table_size);
    mpz_clear(aux);
    mpz_clear(ginvm);
    mpz_clear(z);
    return trouve ? logd : 0;
}

static void diffie_hellman(const mpz_t p, const mpz_t g, unsigned long x, unsigned long y) {
    mpz_t A, B, K1, K2;
    mpz_init(A);
    mpz_init(B);
    mpz_init(K1);
    mpz_init(K2);

    mpz_powm_ui(A, g, x, p);
    mpz_powm_ui(B, g, y, p);
    mpz_powm_ui(K1, B, x, p);
    mpz_powm_ui(K2, A, y, p);

    afficher("A", A);
    afficher("B", B);
    afficher("K1", K1);
    afficher("K2", K2);

    mpz_clear(A);
    mpz_clear(B);
    mpz_clear(K1);
    mpz_clear(K2);
}

static void elgamal_chiffrer(mpz_t c1, mpz_t c2, const mpz_t m, const mpz_t a, const mpz_t b, const mpz_t p, unsigned long k) {
    mpz_t bk;
    mpz_init(bk);
    mpz_powm_ui(c1, a, k, p);
    mpz_powm_ui(bk, b, k, p);
    mpz_mul(c2, m, bk);
    mpz_mod(c2, c2, p);
    mpz_clear(bk);
}

static void elgamal_dechiffrer(mpz_t m_dec, const mpz_t c1, const mpz_t c2, const mpz_t x, const mpz_t p) {
    mpz_t s, s_inv;
    mpz_init(s);
    mpz_init(s_inv);
    mpz_powm(s, c1, x, p);
    if (!mpz_invert(s_inv, s, p)) {
        mpz_clear(s);
        mpz_clear(s_inv);
        return;
    }
    mpz_mul(m_dec, c2, s_inv);
    mpz_mod(m_dec, m_dec, p);
    mpz_clear(s);
    mpz_clear(s_inv);
}

static void dsa_signer(mpz_t gamma, mpz_t delta, const mpz_t Hm, const mpz_t a, const mpz_t alpha, const mpz_t p, const mpz_t q, unsigned long k) {
    mpz_t k_mpz, k_inv, tmp;
    mpz_init_set_ui(k_mpz, k);
    mpz_init(k_inv);
    mpz_init(tmp);

    mpz_powm(gamma, alpha, k_mpz, p);
    mpz_mod(gamma, gamma, q);

    mpz_mul(tmp, a, gamma);
    mpz_add(tmp, Hm, tmp);
    mpz_mod(tmp, tmp, q);
    if (!mpz_invert(k_inv, k_mpz, q)) {
        mpz_set_ui(delta, 0);
    } else {
        mpz_mul(delta, tmp, k_inv);
        mpz_mod(delta, delta, q);
    }

    mpz_clear(k_mpz);
    mpz_clear(k_inv);
    mpz_clear(tmp);
}

static int dsa_verifier(const mpz_t Hm, const mpz_t gamma, const mpz_t delta, const mpz_t alpha, const mpz_t beta, const mpz_t p, const mpz_t q) {
    mpz_t d_inv, e1, e2, t1, t2, v;
    mpz_init(d_inv);
    mpz_init(e1);
    mpz_init(e2);
    mpz_init(t1);
    mpz_init(t2);
    mpz_init(v);

    if (mpz_cmp_ui(gamma, 0) <= 0 || mpz_cmp(gamma, q) >= 0 || mpz_cmp_ui(delta, 0) <= 0 || mpz_cmp(delta, q) >= 0) {
        mpz_clear(d_inv);
        mpz_clear(e1);
        mpz_clear(e2);
        mpz_clear(t1);
        mpz_clear(t2);
        mpz_clear(v);
        return 0;
    }

    if (!mpz_invert(d_inv, delta, q)) {
        mpz_clear(d_inv);
        mpz_clear(e1);
        mpz_clear(e2);
        mpz_clear(t1);
        mpz_clear(t2);
        mpz_clear(v);
        return 0;
    }

    mpz_mul(e1, Hm, d_inv);
    mpz_mod(e1, e1, q);
    mpz_mul(e2, gamma, d_inv);
    mpz_mod(e2, e2, q);
    mpz_powm(t1, alpha, e1, p);
    mpz_powm(t2, beta, e2, p);
    mpz_mul(v, t1, t2);
    mpz_mod(v, v, p);
    mpz_mod(v, v, q);

    int ok = (mpz_cmp(v, gamma) == 0);

    mpz_clear(d_inv);
    mpz_clear(e1);
    mpz_clear(e2);
    mpz_clear(t1);
    mpz_clear(t2);
    mpz_clear(v);
    return ok;
}

static void rsa_generer_cles(mpz_t n, mpz_t e, mpz_t d, const mpz_t p, const mpz_t q) {
    mpz_t phi, p1, q1;
    mpz_init(phi);
    mpz_init(p1);
    mpz_init(q1);
    mpz_mul(n, p, q);
    mpz_sub_ui(p1, p, 1);
    mpz_sub_ui(q1, q, 1);
    mpz_mul(phi, p1, q1);
    mpz_set_ui(e, 65537);
    if (!mpz_invert(d, e, phi)) {
        mpz_set_ui(d, 0);
    }
    mpz_clear(phi);
    mpz_clear(p1);
    mpz_clear(q1);
}

static void crt_deux(mpz_t x, const mpz_t a1, const mpz_t m1, const mpz_t a2, const mpz_t m2) {
    mpz_t inv, diff, tmp;
    mpz_init(inv);
    mpz_init(diff);
    mpz_init(tmp);
    mpz_invert(inv, m1, m2);
    mpz_sub(diff, a2, a1);
    mpz_mod(diff, diff, m2);
    mpz_mul(tmp, diff, inv);
    mpz_mod(tmp, tmp, m2);
    mpz_mul(tmp, tmp, m1);
    mpz_add(x, a1, tmp);
    mpz_clear(inv);
    mpz_clear(diff);
    mpz_clear(tmp);
}

static void demo(void) {
    mpz_t a, b, n, x0;
    mpz_init_set_ui(a, 6);
    mpz_init_set_ui(b, 4);
    mpz_init_set_ui(n, 8);
    mpz_init(x0);

    printf("Solutions lineaires: %d\n", nombre_solutions(a, b, n));
    if (resoudre_lineaire(x0, a, b, n) > 0) {
        afficher("x0", x0);
    }

    mpz_t p, g, beta, logd;
    mpz_init_set_ui(p, 11);
    mpz_init_set_ui(g, 2);
    mpz_init_set_ui(beta, 5);
    mpz_init(logd);
    mpz_set_ui(logd, bsgs(g, beta, p));
    afficher("logd", logd);

    mpz_t dh_p, dh_g;
    mpz_init_set_ui(dh_p, 11);
    mpz_init_set_ui(dh_g, 2);
    diffie_hellman(dh_p, dh_g, 3, 4);

    mpz_t p_rsa, q_rsa, n_rsa, e_rsa, d_rsa, m, c;
    mpz_init_set_ui(p_rsa, 61);
    mpz_init_set_ui(q_rsa, 53);
    mpz_init(n_rsa);
    mpz_init(e_rsa);
    mpz_init(d_rsa);
    mpz_init_set_ui(m, 42);
    mpz_init(c);
    rsa_generer_cles(n_rsa, e_rsa, d_rsa, p_rsa, q_rsa);
    afficher("n", n_rsa);
    afficher("e", e_rsa);
    afficher("d", d_rsa);
    mpz_powm(c, m, e_rsa, n_rsa);
    afficher("c", c);

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(n);
    mpz_clear(x0);
    mpz_clear(p);
    mpz_clear(g);
    mpz_clear(beta);
    mpz_clear(logd);
    mpz_clear(dh_p);
    mpz_clear(dh_g);
    mpz_clear(p_rsa);
    mpz_clear(q_rsa);
    mpz_clear(n_rsa);
    mpz_clear(e_rsa);
    mpz_clear(d_rsa);
    mpz_clear(m);
    mpz_clear(c);
}

int main(int argc, char *argv[]) {
    if (argc == 4) {
        mpz_t p, g, beta;
        mpz_init(p);
        mpz_init(g);
        mpz_init(beta);
        mpz_set_str(p, argv[1], 10);
        mpz_set_str(g, argv[2], 10);
        mpz_set_str(beta, argv[3], 10);
        unsigned long x = bsgs(g, beta, p);
        printf("Log discret : %lu\n", x);
        mpz_clear(p);
        mpz_clear(g);
        mpz_clear(beta);
        return 0;
    }

    demo();
    return 0;
}