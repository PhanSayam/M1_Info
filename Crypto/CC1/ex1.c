#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gmp.h>

int main(int argc, char *argv[]) {
    if(argc != 2) {
        printf("Usage: %s taille\n", argv[0]);
        return 1;
    }

    int taille = atoi(argv[1]);
    gmp_randstate_t state;

    // Initialisation du générateur aléatoire
    gmp_randinit_default(state);
    gmp_randseed_ui(state, time(NULL));

    // Déclaration des entiers
    mpz_t p, q, n, phi, e, d, m, c, m_dech;
    mpz_inits(p, q, n, phi, e, d, m, c, m_dech, NULL);

    // Partie 1 : génération de p et q
    do {
        mpz_urandomb(p, state, taille/2);
        mpz_setbit(p, taille/2 - 1);   // assurer la taille exacte
    } while(!mpz_probab_prime_p(p, 25));

    do {
        mpz_urandomb(q, state, taille/2);
        mpz_setbit(q, taille/2 - 1);  
    } while(!mpz_probab_prime_p(q, 25));

    mpz_mul(n, p, q);                // n = p*q

    mpz_t p1, q1;
    mpz_inits(p1, q1, NULL);
    mpz_sub_ui(p1, p, 1);
    mpz_sub_ui(q1, q, 1);
    mpz_mul(phi, p1, q1);            // phi = (p-1)*(q-1)

    gmp_printf("p = %Zd\nq = %Zd\nn = %Zd\nphi(n) = %Zd\n\n", p, q, n, phi);

    // Partie 2 : génération de e premier avec phi(n)
    do {
        mpz_urandomm(e, state, phi);
        mpz_gcd(d, e, phi);           // ici d temporaire pour stocker le pgcd
    } while(mpz_cmp_ui(e, 1) <= 0 || mpz_cmp_ui(d, 1) != 0);

    mpz_invert(d, e, phi);           // d = e^-1 mod phi
    gmp_printf("e = %Zd\nd = %Zd\n\n", e, d);

    // Partie 3 : chiffrement et déchiffrement
    mpz_urandomm(m, state, n);       // message aléatoire m < n
    gmp_printf("Message m = %Zd\n", m);

    mpz_powm(c, m, e, n);            // c = m^e mod n
    gmp_printf("Chiffre c = %Zd\n", c);

    mpz_powm(m_dech, c, d, n);       // m_dech = c^d mod n
    gmp_printf("Déchiffre m' = %Zd\n", m_dech);

    if(mpz_cmp(m, m_dech) == 0)
        printf("Déchiffrement correct !\n");
    else
        printf("Erreur de déchiffrement !\n");

    // Libération de la mémoire
    mpz_clears(p, q, n, phi, e, d, m, c, m_dech, p1, q1, NULL);
    gmp_randclear(state);

    return 0;
}



#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <time.h>

// -------------------- PGCD --------------------
void pgcd(mpz_t res, mpz_t a, mpz_t b) {
    mpz_t temp;
    mpz_init(temp);
    mpz_set(res, a);
    mpz_t b_copy;
    mpz_init_set(b_copy, b);

    while (mpz_cmp_ui(b_copy, 0) != 0) {
        mpz_set(temp, b_copy);
        mpz_mod(b_copy, res, b_copy);
        mpz_set(res, temp);
    }

    mpz_clear(temp);
    mpz_clear(b_copy);
}

// -------------------- Inverse modulo --------------------
int inverse(mpz_t res, mpz_t a, mpz_t p) {
    // renvoie 1 si l'inverse existe, 0 sinon
    return mpz_invert(res, a, p);
}

// -------------------- Décomposition en facteurs premiers --------------------
void decompose(mpz_t n, mpz_t *factors, mpz_t *exponents, int *size) {
    mpz_t tmp, i, zero, one, e;
    mpz_init_set(tmp, n);
    mpz_init_set_ui(i, 2);
    mpz_init(zero);
    mpz_init_set_ui(one, 1);
    mpz_init_set_ui(e, 0);

    *size = 0;

    // Facteur 2
    while (mpz_divisible_ui_p(tmp, 2)) {
        mpz_divexact_ui(tmp, tmp, 2);
        mpz_add_ui(e, e, 1);
    }
    if (mpz_cmp_ui(e, 0) != 0) {
        mpz_init_set(factors[*size], i);
        mpz_init_set(exponents[*size], e);
        (*size)++;
    }

    mpz_set_ui(i, 3);
    while (mpz_cmp_ui(tmp, 1) > 0) {
        mpz_set_ui(e, 0);
        while (mpz_divisible_p(tmp, i)) {
            mpz_divexact(tmp, tmp, i);
            mpz_add_ui(e, e, 1);
        }
        if (mpz_cmp_ui(e, 0) != 0) {
            mpz_init_set(factors[*size], i);
            mpz_init_set(exponents[*size], e);
            (*size)++;
        }
        mpz_add_ui(i, i, 2);
    }

    mpz_clear(tmp);
    mpz_clear(i);
    mpz_clear(zero);
    mpz_clear(one);
    mpz_clear(e);
}

// -------------------- Puissance modulaire --------------------
void puissance_mod(mpz_t res, mpz_t x, mpz_t y, mpz_t n) {
    mpz_powm(res, x, y, n);
}

// -------------------- Symbole de Jacobi --------------------
int jacobi(mpz_t m, mpz_t n) {
    mpz_t a, b, temp;
    mpz_init_set(a, m);
    mpz_init_set(b, n);
    mpz_init(temp);
    int j = 1;

    mpz_mod(a, a, b);

    while (mpz_cmp_ui(a, 0) != 0) {
        while (mpz_even_p(a)) {
            mpz_divexact_ui(a, a, 2);
            mpz_t r;
            mpz_init(r);
            mpz_mod_ui(r, b, 8);
            unsigned long r_ui = mpz_get_ui(r);
            if (r_ui == 3 || r_ui == 5) j = -j;
            mpz_clear(r);
        }
        mpz_swap(a, b);
        if (mpz_congruent_ui_p(a, 3, 4) && mpz_congruent_ui_p(b, 3, 4)) j = -j;
        mpz_mod(a, a, b);
    }

    int result = (mpz_cmp_ui(b, 1) == 0) ? j : 0;

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(temp);
    return result;
}

// -------------------- Résidus quadratiques --------------------
void residu_quadratique(int n) {
    printf("Résidus quadratiques : ");
    for (int i = 1; i < n; i++) {
        mpz_t base, exp, mod, res;
        mpz_init_set_ui(base, i);
        mpz_init_set_ui(exp, (n-1)/2);
        mpz_init_set_ui(mod, n);
        mpz_init(res);
        mpz_powm(res, base, exp, mod);
        if (mpz_cmp_ui(res, 1) == 0) {
            gmp_printf("%Zd ", base);
        }
        mpz_clear(base); mpz_clear(exp); mpz_clear(mod); mpz_clear(res);
    }
    printf("\n");
}

// -------------------- Clé RSA valide --------------------
int cle_rsa_valid(mpz_t n, mpz_t e) {
    // Suppose n = p*q et e premier avec phi(n)
    mpz_t factors[2], exponents[2];
    int size;
    decompose(n, factors, exponents, &size);
    if (size != 2) return 0;

    mpz_t phi, tmp1, tmp2;
    mpz_init(phi);
    mpz_init(tmp1);
    mpz_init(tmp2);

    mpz_sub_ui(tmp1, factors[0], 1);
    mpz_sub_ui(tmp2, factors[1], 1);
    mpz_mul(phi, tmp1, tmp2);

    int g = mpz_cmp_ui(phi, 0) != 0 && mpz_cmp_ui(e, 0) != 0 && mpz_gcd_ui(NULL, mpz_get_ui(e), mpz_get_ui(phi)) == 1;

    mpz_clear(phi); mpz_clear(tmp1); mpz_clear(tmp2);
    return g;
}

// -------------------- RSA chiffrement / déchiffrement --------------------
void rsa_chiffrement(mpz_t c, mpz_t m, mpz_t e, mpz_t n) {
    mpz_powm(c, m, e, n);
}

void rsa_dechiffrement(mpz_t m, mpz_t c, mpz_t d, mpz_t n) {
    mpz_powm(m, c, d, n);
}

// -------------------- Exemple --------------------
int main() {
    mpz_t n, e, d, m, c;
    mpz_inits(n,e,d,m,c,NULL);

    mpz_set_ui(n, 55);
    mpz_set_ui(e, 3);
    mpz_set_ui(m, 42);

    rsa_chiffrement(c, m, e, n);
    gmp_printf("Cryptogramme : %Zd\n", c);

    // Déchiffrement avec d (supposons d = 27)
    mpz_set_ui(d, 27);
    rsa_dechiffrement(m, c, d, n);
    gmp_printf("Message déchiffré : %Zd\n", m);

    mpz_clears(n,e,d,m,c,NULL);
    return 0;
}


#include <gmp.h>
#include <stdio.h>

int main() {
    mpz_t p, q, n, phi, e, d, C, M;
    mpz_inits(p, q, n, phi, e, d, C, M, NULL);

    mpz_set_str(p,
        "4aa55829181056994b47e8c26e3ed27780892a2679901510ab2769bcec3ea77f098a03d28be3c7834978d92ba57f74f19aff",
        16);

    mpz_set_str(q,
        "f4197a54665c00d21df5ca59a6d8c1632b2c781e29284573d10dfcd0d06c251f858fcf5b86914a9858157a727c2e62e2fdadb",
        16);

    mpz_set_str(e, "3", 16);

    mpz_mul(n, p, q);

    mpz_t p1, q1;
    mpz_inits(p1, q1, NULL);

    mpz_sub_ui(p1, p, 1);
    mpz_sub_ui(q1, q, 1);
    mpz_mul(phi, p1, q1);

    mpz_invert(d, e, phi);

    gmp_printf("d = %Zx\n", d);

    // Pour déchiffrer ton message, remplace ici :
    // mpz_set_str(C, "TON_CRYPTOGRAMME", 16);
    // mpz_powm(M, C, d, n);
    // gmp_printf("M = %Zx\n", M);

    mpz_clears(p,q,n,phi,e,d,C,M,p1,q1,NULL);
    return 0;
}

#include <stdio.h>
#include <gmp.h>

int main() {
    mpz_t n, p, q, r;
    mpz_inits(n, p, q, r, NULL);

    // Charger n en hexadécimal
    mpz_set_str(n,
        "e341cd2f4351d0fec4f1d5062655f983a5b45d16bebf3c2710bc55eedcd84f23"
        "10dd485a0f35c32dc1adbe9f3a99a4ca82ce874c3ea8aedbbd8a2895232ce193"
        "f2f4bd8c031136a1b3d61e46421a0472887093c47fe2cf91d389af0b5d8e4fcb"
        "9302b5f2a427c9877013f457256694e6f0d52f5c6166356ed816970887062fc9"
        "1a36a7b13267678b3ff1d8739a5e4b8c6cb91768cab5e77891fd08de0acff5f7"
        "7d116f54896f6c1b058f85fae7444a2035f06a708d6ceca994c9a94d7d110719"
        "a279062a072c63a418a0c15660dfaa617bb79212aaf9e667f2ac70f548afa250"
        "a4c1c3f19bac0c030d8b1ebd2d7723808b1784ce1e49d3ce03f4441b701b50f",
        16);

    // p = racine entière de n
    mpz_sqrt(p, n);

    // Ajuster p jusqu'à trouver un diviseur
    while (1) {
        mpz_mod(r, n, p);
        if (mpz_cmp_ui(r, 0) == 0) break; // trouvé !
        mpz_add_ui(p, p, 1);              // essayer p+1
    }

    // Calculer q = n / p
    mpz_div(q, n, p);

    // Affichage
    printf("p = ");
    mpz_out_str(stdout, 16, p);
    printf("\n\n");

    printf("q = ");
    mpz_out_str(stdout, 16, q);
    printf("\n");

    mpz_clears(n, p, q, r, NULL);
    return 0;
}
