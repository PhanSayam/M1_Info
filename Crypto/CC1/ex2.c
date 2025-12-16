#include <stdio.h>
#include <gmp.h>
#include <time.h>

// ---------- PGCD ----------
void pgcd(mpz_t r, const mpz_t a, const mpz_t b) {
    mpz_t x, y, tmp;
    mpz_inits(x, y, tmp, NULL);
    mpz_set(x, a);
    mpz_set(y, b);

    while (mpz_cmp_ui(y, 0) != 0) {
        mpz_set(tmp, y);
        mpz_mod(y, x, y);
        mpz_set(x, tmp);
    }

    mpz_set(r, x);
    mpz_clears(x, y, tmp, NULL);
}

// ---------- Inverse modulo ----------
int inverse(mpz_t r, const mpz_t a, const mpz_t n) {
    return mpz_invert(r, a, n);
}

// ---------- Décomposition naïve ----------
void factoriser(mpz_t n, mpz_t *f, mpz_t *e, int *sz) {
    mpz_t tmp, i, exp;
    mpz_inits(tmp, i, exp, NULL);
    mpz_set(tmp, n);
    *sz = 0;

    // facteur 2
    mpz_set_ui(i, 2);
    mpz_set_ui(exp, 0);
    while (mpz_divisible_ui_p(tmp, 2)) {
        mpz_divexact_ui(tmp, tmp, 2);
        mpz_add_ui(exp, exp, 1);
    }
    if (mpz_cmp_ui(exp, 0) > 0) {
        mpz_init_set(f[*sz], i);
        mpz_init_set(e[*sz], exp);
        (*sz)++;
    }

    // impair
    mpz_set_ui(i, 3);
    while (mpz_cmp_ui(tmp, 1) > 0) {
        mpz_set_ui(exp, 0);
        while (mpz_divisible_p(tmp, i)) {
            mpz_divexact(tmp, tmp, i);
            mpz_add_ui(exp, exp, 1);
        }
        if (mpz_cmp_ui(exp, 0) > 0) {
            mpz_init_set(f[*sz], i);
            mpz_init_set(e[*sz], exp);
            (*sz)++;
        }
        mpz_add_ui(i, i, 2);
    }

    mpz_clears(tmp, i, exp, NULL);
}

// ---------- Jacobi ----------
int jacobi(const mpz_t a_in, const mpz_t n_in) {
    mpz_t a, n, tmp;
    mpz_inits(a, n, tmp, NULL);
    mpz_set(a, a_in);
    mpz_set(n, n_in);

    int j = 1;

    mpz_mod(a, a, n);
    while (mpz_cmp_ui(a, 0) != 0) {
        while (mpz_even_p(a)) {
            mpz_divexact_ui(a, a, 2);
            mpz_mod_ui(tmp, n, 8);
            unsigned long r = mpz_get_ui(tmp);
            if (r == 3 || r == 5) j = -j;
        }
        mpz_swap(a, n);
        if (mpz_congruent_ui_p(a, 3, 4) &&
            mpz_congruent_ui_p(n, 3, 4)) j = -j;
        mpz_mod(a, a, n);
    }

    int res = (mpz_cmp_ui(n, 1) == 0) ? j : 0;
    mpz_clears(a, n, tmp, NULL);
    return res;
}

// ---------- RSA ----------
void rsa_chiffrer(mpz_t c, const mpz_t m, const mpz_t e, const mpz_t n) {
    mpz_powm(c, m, e, n);
}

void rsa_dechiffrer(mpz_t m, const mpz_t c, const mpz_t d, const mpz_t n) {
    mpz_powm(m, c, d, n);
}

int main() {
    mpz_t n, e, d, m, c;
    mpz_inits(n, e, d, m, c, NULL);

    mpz_set_ui(n, 55);
    mpz_set_ui(e, 3);
    mpz_set_ui(m, 42);

    rsa_chiffrer(c, m, e, n);
    gmp_printf("Cryptogramme : %Zd\n", c);

    mpz_set_ui(d, 27); // inverse correct modulo phi(55)=40
    rsa_dechiffrer(m, c, d, n);
    gmp_printf("Déchiffré : %Zd\n", m);

    mpz_clears(n, e, d, m, c, NULL);
    return 0;
}
