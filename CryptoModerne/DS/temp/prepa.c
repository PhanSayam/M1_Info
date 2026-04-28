/*
 * =============================================================================
 * RÉFÉRENCE COMPLÈTE — CRYPTOGRAPHIE EN C
 * Bibliothèques : GMP uniquement + hashutils.h (structure du TP)
 * Compilation : gcc crypto_reference_gmp.c -o crypto_ref -lgmp -lm
 * =============================================================================
 *
 * SOMMAIRE :
 * BLOC  1 — Opérations de base GMP (init, set, affichage)
 * BLOC  2 — Arithmétique modulaire (add, mul, mod, pow)
 * BLOC  3 — Inverse modulaire
 * BLOC  4 — PGCD et équations linéaires modulaires ax ≡ b (mod n)
 * BLOC  5 — Exponentiation modulaire (Square-and-Multiply)
 * BLOC  6 — Baby-Step Giant-Step (BSGS)
 * BLOC  7 — Table de hachage (hashutils.h — structure du TP)
 * BLOC  8 — Recherche naïve du logarithme discret
 * BLOC  9 — Diffie-Hellman manuel
 * BLOC 10 — ElGamal (chiffrement / déchiffrement)
 * BLOC 11 — DSA (signature / vérification)
 * BLOC 12 — Attaque réutilisation nonce DSA
 * BLOC 13 — RSA manuel
 * BLOC 14 — Pohlig-Hellman + CRT
 * BLOC 15 — Structure type d'un programme TP (main complet)
 * BLOC 16 — Tableau récapitulatif de toutes les fonctions GMP
 * BLOC 17 — Pièges classiques
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <gmp.h>
#include "hashutils.h"


/* =============================================================================
 * BLOC 1 — OPÉRATIONS DE BASE GMP
 * =============================================================================
 *
 * RÈGLE ABSOLUE :
 *   mpz_init  AVANT toute utilisation
 *   mpz_clear APRÈS pour libérer la mémoire
 */

void exemple_init_et_affectation(void) {
    mpz_t a, b, c;

    /* --- Initialisation --- */
    mpz_init(a);                        /* initialiser à 0 */
    mpz_init_set_ui(b, 42);            /* initialiser et affecter unsigned long */
    mpz_init_set_str(c, "123456789012345678901234567890", 10);
                                        /* initialiser depuis string, base 10 */

    /* --- Affectation après init --- */
    mpz_set_ui(a, 100);                /* depuis unsigned long */
    mpz_set_str(a, "9999999999999999", 10); /* depuis string */
    mpz_set(a, b);                     /* copier b dans a */

    /* --- Affichage --- */
    gmp_printf("a = %Zd\n", a);       /* %Zd pour mpz_t */
    gmp_printf("b = %Zd\n", b);
    gmp_printf("c = %Zd\n", c);
    printf("b (ulong) = %lu\n", mpz_get_ui(b));  /* vers unsigned long */

    /* --- Libération (obligatoire) --- */
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);
}

/* Utilitaire : afficher un mpz_t avec label */
void afficher(const char *label, const mpz_t x) {
    gmp_printf("%s = %Zd\n", label, x);
}

/* Utilitaire : lire un mpz_t depuis argv */
void lire_arg(mpz_t x, const char *argv_val) {
    mpz_set_str(x, argv_val, 10);
}


/* =============================================================================
 * BLOC 2 — ARITHMÉTIQUE MODULAIRE DE BASE
 * ============================================================================= */

void exemple_arithmetique(void) {
    mpz_t a, b, n, r;
    mpz_init_set_str(a, "1234567890", 10);
    mpz_init_set_str(b, "9876543210", 10);
    mpz_init_set_str(n, "1000000007", 10);
    mpz_init(r);

    /* Addition */
    mpz_add(r, a, b);               /* r = a + b */
    afficher("a + b", r);

    mpz_add_ui(r, a, 1);            /* r = a + 1 */

    /* Soustraction */
    mpz_sub(r, a, b);               /* r = a - b (peut être négatif) */
    mpz_sub_ui(r, a, 1);            /* r = a - 1 */

    /* Multiplication */
    mpz_mul(r, a, b);               /* r = a * b */
    mpz_mul_ui(r, a, 3);            /* r = a * 3 */

    /* Modulo */
    mpz_mod(r, a, n);               /* r = a mod n (toujours positif) */
    afficher("a mod n", r);

    /* Division exacte (sans reste) */
    mpz_divexact(r, b, a);          /* r = b / a (exact) */

    /* Quotient et reste */
    mpz_t q;
    mpz_init(q);
    mpz_tdiv_qr(q, r, b, a);       /* q = b/a, r = b mod a */

    /* Valeur absolue */
    mpz_abs(r, a);                  /* r = |a| */

    /* Négatif */
    mpz_neg(r, a);                  /* r = -a */

    mpz_clear(a); mpz_clear(b); mpz_clear(n);
    mpz_clear(r); mpz_clear(q);
}

/* Comparaison */
void exemple_comparaison(void) {
    mpz_t a, b;
    mpz_init_set_ui(a, 5);
    mpz_init_set_ui(b, 10);

    int cmp = mpz_cmp(a, b);       /* < 0 si a<b, 0 si a=b, > 0 si a>b */
    if (cmp < 0) printf("a < b\n");
    if (cmp == 0) printf("a == b\n");
    if (cmp > 0) printf("a > b\n");

    /* Comparer avec un entier */
    if (mpz_cmp_ui(a, 0) > 0) printf("a > 0\n");

    /* Test de divisibilité */
    if (mpz_divisible_p(b, a)) printf("a divise b\n");

    mpz_clear(a); mpz_clear(b);
}


/* =============================================================================
 * BLOC 3 — INVERSE MODULAIRE
 * =============================================================================
 *
 * Trouver x tel que a*x ≡ 1 (mod n)
 * N'existe que si pgcd(a, n) = 1
 * Équivalent à pow(a, -1, n) en Python
 */

void exemple_inverse_modulaire(void) {
    mpz_t a, n, inv;
    mpz_init_set_ui(a, 3);
    mpz_init_set_ui(n, 11);
    mpz_init(inv);

    /* mpz_invert retourne 1 si l'inverse existe, 0 sinon */
    if (mpz_invert(inv, a, n)) {
        afficher("Inverse de 3 mod 11", inv);  /* doit être 4 : 3*4=12≡1 mod 11 */
    } else {
        printf("Pas d'inverse (a et n non premiers entre eux)\n");
    }

    /* Exemple sans inverse */
    mpz_set_ui(a, 6);
    mpz_set_ui(n, 9);  /* pgcd(6,9) = 3 ≠ 1 */
    if (!mpz_invert(inv, a, n)) {
        printf("6 n'a pas d'inverse mod 9\n");
    }

    mpz_clear(a); mpz_clear(n); mpz_clear(inv);
}

/* Calculer g^{-m} mod p (utile pour BSGS) */
void calculer_g_inv_m(mpz_t resultat, const mpz_t g,
                       unsigned long m, const mpz_t p) {
    mpz_t gm;
    mpz_init(gm);
    mpz_powm_ui(gm, g, m, p);     /* gm = g^m mod p */
    mpz_invert(resultat, gm, p);  /* resultat = (g^m)^{-1} mod p */
    mpz_clear(gm);
}


/* =============================================================================
 * BLOC 4 — PGCD ET ÉQUATIONS LINÉAIRES MODULAIRES
 * =============================================================================
 *
 * Résoudre ax ≡ b (mod n)
 * g = pgcd(a, n)
 * Si g ne divise pas b → 0 solution
 * Si g divise b        → exactement g solutions
 */

/* Calculer g = pgcd(a, b) */
void pgcd(mpz_t g, const mpz_t a, const mpz_t b) {
    mpz_gcd(g, a, b);
}

/* Combien de solutions ax ≡ b (mod n) ? */
int nombre_solutions(const mpz_t a, const mpz_t b, const mpz_t n) {
    mpz_t g;
    mpz_init(g);
    mpz_gcd(g, a, n);

    int nb;
    if (mpz_divisible_p(b, g))
        nb = (int)mpz_get_ui(g);
    else
        nb = 0;

    mpz_clear(g);
    return nb;
}

/* Trouver une solution x0 de ax ≡ b (mod n) */
/* Retourne le nombre de solutions (0 si aucune) */
int resoudre_lineaire(mpz_t x0, const mpz_t a, const mpz_t b, const mpz_t n) {
    mpz_t g, a_red, b_red, n_red, inv;
    mpz_init(g);    mpz_init(a_red);
    mpz_init(b_red); mpz_init(n_red); mpz_init(inv);

    mpz_gcd(g, a, n);

    if (!mpz_divisible_p(b, g)) {
        printf("Aucune solution\n");
        mpz_clear(g); mpz_clear(a_red); mpz_clear(b_red);
        mpz_clear(n_red); mpz_clear(inv);
        return 0;
    }

    /* Diviser tout par g */
    mpz_divexact(a_red, a, g);
    mpz_divexact(b_red, b, g);
    mpz_divexact(n_red, n, g);

    /* x0 = b_red * (a_red)^{-1} mod n_red */
    mpz_invert(inv, a_red, n_red);
    mpz_mul(x0, b_red, inv);
    mpz_mod(x0, x0, n_red);

    int nb = (int)mpz_get_ui(g);

    mpz_clear(g); mpz_clear(a_red); mpz_clear(b_red);
    mpz_clear(n_red); mpz_clear(inv);
    return nb;
}

void exemple_eq_lineaire(void) {
    /* Résoudre 6x ≡ 4 (mod 8) */
    mpz_t a, b, n, g, x0;
    mpz_init_set_ui(a, 6);
    mpz_init_set_ui(b, 4);
    mpz_init_set_ui(n, 8);
    mpz_init(g); mpz_init(x0);

    mpz_gcd(g, a, n);
    gmp_printf("pgcd(6, 8) = %Zd\n", g);  /* 2 */

    int nb = nombre_solutions(a, b, n);
    printf("Nombre de solutions : %d\n", nb);  /* 2 */

    resoudre_lineaire(x0, a, b, n);
    gmp_printf("x0 = %Zd\n", x0);  /* 2, l'autre solution est x0 + n/g = 6 */

    mpz_clear(a); mpz_clear(b); mpz_clear(n);
    mpz_clear(g); mpz_clear(x0);
}


/* =============================================================================
 * BLOC 5 — EXPONENTIATION MODULAIRE
 * =============================================================================
 *
 * Calcule base^exposant mod modulo
 * GMP utilise Square-and-Multiply en interne
 * Complexité : O((log exposant)^3)
 */

void exemple_exp_modulaire(void) {
    mpz_t base, expo, mod, resultat;
    mpz_init_set_str(base, "2", 10);
    mpz_init_set_str(expo, "100", 10);
    mpz_init_set_str(mod,  "997", 10);
    mpz_init(resultat);

    /* r = base^expo mod mod */
    mpz_powm(resultat, base, expo, mod);
    afficher("2^100 mod 997", resultat);

    /* Avec exposant unsigned long */
    mpz_powm_ui(resultat, base, 100UL, mod);
    afficher("2^100 mod 997 (ui)", resultat);

    mpz_clear(base); mpz_clear(expo);
    mpz_clear(mod);  mpz_clear(resultat);
}

/* Implémenter Square-and-Multiply manuellement (pour comprendre) */
void square_and_multiply(mpz_t resultat, const mpz_t base,
                          const mpz_t expo, const mpz_t mod) {
    /*
     * z = 1
     * Pour chaque bit de expo de poids fort au poids faible :
     *   z = z^2 mod n
     *   Si bit = 1 : z = z * base mod n
     */
    mpz_t z, x, j;
    mpz_init_set_ui(z, 1);
    mpz_init_set(x, base);
    mpz_init_set(j, expo);

    while (mpz_cmp_ui(j, 0) != 0) {
        /* Si j est impair (bit de poids faible = 1) */
        if (mpz_odd_p(j)) {
            mpz_mul(z, z, x);
            mpz_mod(z, z, mod);
        }
        /* x = x^2 mod n */
        mpz_mul(x, x, x);
        mpz_mod(x, x, mod);
        /* j = j / 2 */
        mpz_tdiv_q_ui(j, j, 2);
    }

    mpz_set(resultat, z);
    mpz_clear(z); mpz_clear(x); mpz_clear(j);
}


/* =============================================================================
 * BLOC 6 — BABY-STEP GIANT-STEP
 * =============================================================================
 *
 * Résoudre g^x ≡ beta (mod p)
 * Complexité : O(sqrt(p)) en temps et mémoire
 *
 * Principe :
 *   m = ceil(sqrt(p))
 *   x = i*m + j  avec 0 <= i,j <= m-1
 *
 *   g^{i*m + j} = beta
 *   g^j = beta * (g^{-m})^i
 *
 *   Phase 1 (Baby Steps) : calculer et stocker g^j pour j=0..m-1
 *   Phase 2 (Giant Steps) : calculer beta*(g^{-m})^i jusqu'à collision
 */

/* Version avec la table de hachage du TP (hashutils.h) */
unsigned long bsgs(const mpz_t g, const mpz_t beta, const mpz_t p) {

    /* Calculer m = floor(sqrt(p)) + 1 */
    mpz_t temp;
    mpz_init(temp);
    mpz_sqrt(temp, p);
    unsigned long m = mpz_get_ui(temp) + 1;
    mpz_clear(temp);

    printf("m = %lu\n", m);

    /* Allouer la table de hachage */
    unsigned int table_size = 2 * m + 1;
    HashTable ht = calloc(table_size, sizeof(Maillon *));
    if (!ht) { fprintf(stderr, "Erreur allocation\n"); exit(1); }

    /* ---- Phase 1 : Baby Steps ---- */
    /* Calculer g^i mod p et stocker (g^i, i) dans la table */
    mpz_t aux;
    mpz_init_set_ui(aux, 1);

    for (unsigned long i = 1; i < m; i++) {
        mpz_mul(aux, aux, g);
        mpz_mod(aux, aux, p);
        /* Stocker uniquement si la valeur tient dans unsigned long */
        hash_insert(&ht, mpz_get_ui(aux), i, table_size);
    }
    printf("Phase 1 OK\n");

    /* ---- Phase 2 : Giant Steps ---- */
    /* Calculer g^{-m} mod p */
    mpz_t ginvm;
    mpz_init(ginvm);
    mpz_powm_ui(ginvm, g, m, p);    /* g^m mod p */
    mpz_invert(ginvm, ginvm, p);    /* g^{-m} mod p */

    /* Parcourir beta*(g^{-m})^j jusqu'à collision */
    mpz_t z;
    mpz_init_set(z, beta);

    unsigned long logd = 0;
    unsigned long i_match;
    int trouve = 0;

    for (unsigned long j = 0; j <= m && !trouve; j++) {
        unsigned long z_val = mpz_get_ui(z);
        if (hash_find(&ht, z_val, &i_match, table_size)) {
            logd = j * m + i_match;   /* x = j*m + i */
            trouve = 1;
        }
        mpz_mul(z, z, ginvm);
        mpz_mod(z, z, p);
    }

    /* Libérer la mémoire */
    /* (libérer les listes chaînées si nécessaire) */
    free(ht);

    mpz_clear(aux); mpz_clear(ginvm); mpz_clear(z);

    return logd;
}

void exemple_bsgs(void) {
    mpz_t p, g, beta, verif;
    mpz_init_set_str(p,    "1934459", 10);
    mpz_init_set_str(g,    "762973",  10);
    mpz_init_set_str(beta, "1191663", 10);
    mpz_init(verif);

    unsigned long x = bsgs(g, beta, p);
    printf("Log discret x = %lu\n", x);

    /* Vérification : g^x mod p doit valoir beta */
    mpz_powm_ui(verif, g, x, p);
    gmp_printf("g^x mod p = %Zd\n", verif);
    gmp_printf("beta      = %Zd\n", beta);

    if (mpz_cmp(verif, beta) == 0)
        printf("Vérification : OK\n");
    else
        printf("Vérification : ERREUR\n");

    mpz_clear(p); mpz_clear(g);
    mpz_clear(beta); mpz_clear(verif);
}


/* =============================================================================
 * BLOC 7 — TABLE DE HACHAGE (hashutils.h)
 * =============================================================================
 *
 * Structure définie dans hashutils.h :
 *
 * typedef struct Maillon {
 *     unsigned long gj;    // valeur g^j mod p
 *     unsigned long j;     // exposant j
 *     struct Maillon *next;
 * } Maillon;
 *
 * typedef Maillon **HashTable;
 *
 * Fonctions disponibles :
 *   hash(z, table_size)           → index de hachage
 *   hash_insert(&ht, gj, j, size) → insérer (gj, j)
 *   hash_find(&ht, val, &j, size) → chercher val, retourne 1 si trouvé
 */

void exemple_hashtable(void) {
    int taille = 100;
    HashTable ht = calloc(taille, sizeof(Maillon *));

    /* Insérer des valeurs */
    hash_insert(&ht, 12345, 10, taille);
    hash_insert(&ht, 67890, 20, taille);
    hash_insert(&ht, 11111, 30, taille);

    /* Chercher une valeur */
    unsigned long j_trouve;
    if (hash_find(&ht, 67890, &j_trouve, taille)) {
        printf("67890 trouvé : j = %lu\n", j_trouve);  /* 20 */
    } else {
        printf("67890 non trouvé\n");
    }

    if (!hash_find(&ht, 99999, &j_trouve, taille)) {
        printf("99999 non trouvé\n");
    }

    free(ht);
}


/* =============================================================================
 * BLOC 8 — RECHERCHE NAÏVE DU LOGARITHME DISCRET
 * =============================================================================
 *
 * Énumération exhaustive de g^0, g^1, ..., g^{p-2}
 * Complexité : O(p) — uniquement pour petits p (tests/vérification)
 */

void recherche_naive(const mpz_t g, const mpz_t beta, const mpz_t p) {
    mpz_t b, x, p_1;
    mpz_init(b);
    mpz_init_set_ui(x, 0);
    mpz_init(p_1);
    mpz_sub_ui(p_1, p, 1);     /* p_1 = p - 1 */

    while (mpz_cmp(x, p_1) < 0) {
        mpz_powm(b, g, x, p);  /* b = g^x mod p */
        if (mpz_cmp(b, beta) == 0) {
            gmp_printf("Naïf : x = %Zd\n", x);
            break;
        }
        mpz_add_ui(x, x, 1);
    }

    mpz_clear(b); mpz_clear(x); mpz_clear(p_1);
}


/* =============================================================================
 * BLOC 9 — DIFFIE-HELLMAN
 * =============================================================================
 *
 * Paramètres publics : p premier, g générateur de (Z/pZ)*
 * Alice choisit x secret, envoie A = g^x mod p
 * Bob   choisit y secret, envoie B = g^y mod p
 * Secret partagé : K = g^{xy} mod p
 *
 * Sécurité basée sur CDH : connaissant A et B, calculer K est difficile
 */

void diffie_hellman(const mpz_t p, const mpz_t g,
                    unsigned long x, unsigned long y) {
    mpz_t A, B, K_alice, K_bob;
    mpz_init(A); mpz_init(B);
    mpz_init(K_alice); mpz_init(K_bob);

    /* Alice : A = g^x mod p */
    mpz_powm_ui(A, g, x, p);
    afficher("Alice envoie A = g^x", A);

    /* Bob : B = g^y mod p */
    mpz_powm_ui(B, g, y, p);
    afficher("Bob envoie B   = g^y", B);

    /* Alice : K = B^x mod p = g^{xy} mod p */
    mpz_powm_ui(K_alice, B, x, p);
    afficher("Secret Alice", K_alice);

    /* Bob : K = A^y mod p = g^{xy} mod p */
    mpz_powm_ui(K_bob, A, y, p);
    afficher("Secret Bob  ", K_bob);

    if (mpz_cmp(K_alice, K_bob) == 0)
        printf("Secrets identiques : OK\n");
    else
        printf("ERREUR : secrets différents\n");

    mpz_clear(A); mpz_clear(B);
    mpz_clear(K_alice); mpz_clear(K_bob);
}

void exemple_dh(void) {
    mpz_t p, g;
    mpz_init_set_ui(p, 11);
    mpz_init_set_ui(g, 2);

    printf("\n=== Diffie-Hellman (p=11, g=2, x=3, y=4) ===\n");
    diffie_hellman(p, g, 3, 4);

    mpz_clear(p); mpz_clear(g);
}


/* =============================================================================
 * BLOC 10 — ELGAMAL
 * =============================================================================
 *
 * Paramètres : p premier, a générateur, x secret, b = a^x mod p (public)
 *
 * Chiffrement de m avec k aléatoire :
 *   c1 = a^k mod p
 *   c2 = m * b^k mod p
 * Chiffré : (c1, c2)
 *
 * Déchiffrement :
 *   s = c1^x mod p = a^{kx} mod p
 *   m = c2 * s^{-1} mod p
 *
 * Propriété : même m avec k différents → chiffrés différents (probabiliste)
 * Inconvénient : chiffré 2 fois plus long que le message
 */

void elgamal_chiffrer(mpz_t c1, mpz_t c2,
                       const mpz_t m, const mpz_t a,
                       const mpz_t b, const mpz_t p,
                       unsigned long k) {
    mpz_t bk;
    mpz_init(bk);

    mpz_powm_ui(c1, a, k, p);     /* c1 = a^k mod p */
    mpz_powm_ui(bk, b, k, p);     /* bk = b^k mod p */
    mpz_mul(c2, m, bk);           /* c2 = m * b^k */
    mpz_mod(c2, c2, p);

    mpz_clear(bk);
}

void elgamal_dechiffrer(mpz_t m_dec,
                         const mpz_t c1, const mpz_t c2,
                         const mpz_t x, const mpz_t p) {
    mpz_t s, s_inv;
    mpz_init(s); mpz_init(s_inv);

    mpz_powm(s, c1, x, p);        /* s = c1^x mod p = a^{kx} */
    mpz_invert(s_inv, s, p);      /* s_inv = s^{-1} mod p */
    mpz_mul(m_dec, c2, s_inv);    /* m = c2 * s_inv */
    mpz_mod(m_dec, m_dec, p);

    mpz_clear(s); mpz_clear(s_inv);
}

void exemple_elgamal(void) {
    mpz_t p, a, x, b, m, c1, c2, m_dec;

    mpz_init_set_ui(p, 11);    /* premier */
    mpz_init_set_ui(a, 2);     /* générateur */
    mpz_init_set_ui(x, 3);     /* clé secrète */
    mpz_init(b);
    mpz_init_set_ui(m, 5);     /* message */
    mpz_init(c1); mpz_init(c2); mpz_init(m_dec);

    /* Clé publique b = a^x mod p */
    mpz_powm_ui(b, a, mpz_get_ui(x), p);
    afficher("Clé publique b = a^x", b);  /* 8 */

    /* Chiffrement avec k=2 */
    printf("\n=== ElGamal (p=11, a=2, x=3, m=5, k=2) ===\n");
    elgamal_chiffrer(c1, c2, m, a, b, p, 2);
    afficher("c1", c1);
    afficher("c2", c2);

    /* Déchiffrement */
    elgamal_dechiffrer(m_dec, c1, c2, x, p);
    afficher("m déchiffré", m_dec);  /* doit valoir 5 */

    mpz_clear(p); mpz_clear(a); mpz_clear(x); mpz_clear(b);
    mpz_clear(m); mpz_clear(c1); mpz_clear(c2); mpz_clear(m_dec);
}


/* =============================================================================
 * BLOC 11 — DSA (DIGITAL SIGNATURE ALGORITHM)
 * =============================================================================
 *
 * Paramètres publics :
 *   p : grand premier
 *   q : premier de 160 bits, q | (p-1)
 *   alpha : racine q-ième de l'unité dans (Z/pZ)*
 *   beta = alpha^a mod p : clé publique
 *   H : fonction de hachage (simulée ici par un entier)
 *
 * Clé secrète : a (1 <= a <= q-1)
 *
 * GÉNÉRATION de la signature de m :
 *   1. Calculer H(m)
 *   2. Choisir k aléatoire dans [1, q-1]  ← UNIQUE par signature
 *   3. gamma = (alpha^k mod p) mod q
 *   4. delta = (H(m) + a*gamma) * k^{-1} mod q
 *   Signature = (gamma, delta)
 *
 * VÉRIFICATION de (m, gamma, delta) :
 *   1. e1 = H(m) * delta^{-1} mod q
 *   2. e2 = gamma * delta^{-1} mod q
 *   3. v  = (alpha^e1 * beta^e2 mod p) mod q
 *   4. Accepter si v == gamma
 */

void dsa_signer(mpz_t gamma, mpz_t delta,
                const mpz_t Hm,
                const mpz_t a, const mpz_t alpha,
                const mpz_t p, const mpz_t q,
                unsigned long k) {
    mpz_t k_mpz, k_inv, tmp;
    mpz_init_set_ui(k_mpz, k);
    mpz_init(k_inv); mpz_init(tmp);

    /* gamma = (alpha^k mod p) mod q */
    mpz_powm(gamma, alpha, k_mpz, p);
    mpz_mod(gamma, gamma, q);

    /* delta = (H(m) + a*gamma) * k^{-1} mod q */
    mpz_mul(tmp, a, gamma);         /* tmp = a * gamma */
    mpz_add(tmp, Hm, tmp);          /* tmp = H(m) + a*gamma */
    mpz_mod(tmp, tmp, q);
    mpz_invert(k_inv, k_mpz, q);   /* k_inv = k^{-1} mod q */
    mpz_mul(delta, tmp, k_inv);
    mpz_mod(delta, delta, q);

    mpz_clear(k_mpz); mpz_clear(k_inv); mpz_clear(tmp);
}

int dsa_verifier(const mpz_t Hm,
                  const mpz_t gamma, const mpz_t delta,
                  const mpz_t alpha, const mpz_t beta,
                  const mpz_t p, const mpz_t q) {
    mpz_t d_inv, e1, e2, t1, t2, v;
    mpz_init(d_inv); mpz_init(e1); mpz_init(e2);
    mpz_init(t1);    mpz_init(t2); mpz_init(v);

    /* Vérifier 0 < gamma < q et 0 < delta < q */
    if (mpz_cmp_ui(gamma, 0) <= 0 || mpz_cmp(gamma, q) >= 0 ||
        mpz_cmp_ui(delta, 0) <= 0 || mpz_cmp(delta, q) >= 0) {
        mpz_clear(d_inv); mpz_clear(e1); mpz_clear(e2);
        mpz_clear(t1);    mpz_clear(t2); mpz_clear(v);
        return 0;
    }

    /* e1 = H(m) * delta^{-1} mod q */
    mpz_invert(d_inv, delta, q);
    mpz_mul(e1, Hm, d_inv);
    mpz_mod(e1, e1, q);

    /* e2 = gamma * delta^{-1} mod q */
    mpz_mul(e2, gamma, d_inv);
    mpz_mod(e2, e2, q);

    /* v = (alpha^e1 * beta^e2 mod p) mod q */
    mpz_powm(t1, alpha, e1, p);
    mpz_powm(t2, beta,  e2, p);
    mpz_mul(v, t1, t2);
    mpz_mod(v, v, p);
    mpz_mod(v, v, q);

    int ok = (mpz_cmp(v, gamma) == 0);

    mpz_clear(d_inv); mpz_clear(e1); mpz_clear(e2);
    mpz_clear(t1);    mpz_clear(t2); mpz_clear(v);
    return ok;
}

void exemple_dsa(void) {
    /*
     * Paramètres réduits pour l'exemple
     * En pratique : p=1024 bits, q=160 bits
     */
    mpz_t p, q, alpha, a, beta, Hm, gamma, delta;

    mpz_init_set_str(p,     "23",  10);
    mpz_init_set_str(q,     "11",  10);
    mpz_init_set_str(alpha, "2",   10);  /* alpha^q ≡ 1 mod p */
    mpz_init_set_str(a,     "3",   10);  /* clé secrète */
    mpz_init(beta);
    mpz_init_set_str(Hm,    "7",   10);  /* H(message) simulé */
    mpz_init(gamma); mpz_init(delta);

    /* Clé publique beta = alpha^a mod p */
    mpz_powm(beta, alpha, a, p);
    afficher("beta = alpha^a mod p", beta);

    printf("\n=== DSA Signature (k=4) ===\n");
    dsa_signer(gamma, delta, Hm, a, alpha, p, q, 4);
    afficher("gamma", gamma);
    afficher("delta", delta);

    printf("\n=== DSA Vérification ===\n");
    int ok = dsa_verifier(Hm, gamma, delta, alpha, beta, p, q);
    printf("Signature valide : %s\n", ok ? "OUI" : "NON");

    mpz_clear(p); mpz_clear(q); mpz_clear(alpha); mpz_clear(a);
    mpz_clear(beta); mpz_clear(Hm); mpz_clear(gamma); mpz_clear(delta);
}


/* =============================================================================
 * BLOC 12 — ATTAQUE : RÉUTILISATION DU NONCE DSA
 * =============================================================================
 *
 * Si le même k est utilisé pour deux signatures différentes :
 * → gamma est IDENTIQUE dans les deux signatures (signal d'alarme)
 *
 * On dispose de :
 *   Signature 1 : (m1, gamma, delta1)
 *   Signature 2 : (m2, gamma, delta2)
 *
 * On peut retrouver k :
 *   delta1 - delta2 = (H(m1) - H(m2)) * k^{-1} mod q
 *   k = (H(m1) - H(m2)) * (delta1 - delta2)^{-1} mod q
 *
 * Puis retrouver la clé secrète a :
 *   delta1 = (H(m1) + a*gamma) * k^{-1} mod q
 *   a = (delta1*k - H(m1)) * gamma^{-1} mod q
 */

void attaque_nonce_dsa(const mpz_t Hm1, const mpz_t Hm2,
                        const mpz_t delta1, const mpz_t delta2,
                        const mpz_t gamma, const mpz_t q) {
    mpz_t k, a_retrouve, num, den, tmp, g_inv;
    mpz_init(k);   mpz_init(a_retrouve);
    mpz_init(num); mpz_init(den);
    mpz_init(tmp); mpz_init(g_inv);

    printf("\n=== Attaque réutilisation nonce DSA ===\n");

    /* k = (H(m1) - H(m2)) * (delta1 - delta2)^{-1} mod q */
    mpz_sub(num, Hm1, Hm2);
    mpz_mod(num, num, q);          /* H(m1) - H(m2) mod q */

    mpz_sub(den, delta1, delta2);
    mpz_mod(den, den, q);          /* delta1 - delta2 mod q */
    mpz_invert(den, den, q);       /* (delta1 - delta2)^{-1} mod q */

    mpz_mul(k, num, den);
    mpz_mod(k, k, q);
    afficher("k retrouvé", k);

    /* a = (delta1*k - H(m1)) * gamma^{-1} mod q */
    mpz_mul(tmp, delta1, k);       /* delta1 * k */
    mpz_sub(tmp, tmp, Hm1);        /* delta1*k - H(m1) */
    mpz_mod(tmp, tmp, q);

    mpz_invert(g_inv, gamma, q);   /* gamma^{-1} mod q */
    mpz_mul(a_retrouve, tmp, g_inv);
    mpz_mod(a_retrouve, a_retrouve, q);
    afficher("Clé privée a retrouvée", a_retrouve);

    mpz_clear(k); mpz_clear(a_retrouve); mpz_clear(num);
    mpz_clear(den); mpz_clear(tmp); mpz_clear(g_inv);
}

void exemple_attaque_dsa(void) {
    /*
     * Scénario :
     * Alice signe deux messages avec le même k
     * L'attaquant observe les deux signatures et retrouve a
     */
    mpz_t p, q, alpha, a, beta;
    mpz_init_set_ui(p, 23);
    mpz_init_set_ui(q, 11);
    mpz_init_set_ui(alpha, 2);
    mpz_init_set_ui(a, 3);      /* clé secrète — inconnue de l'attaquant */
    mpz_init(beta);
    mpz_powm(beta, alpha, a, p);

    /* Alice signe m1 et m2 avec k=4 (erreur !) */
    unsigned long k = 4;
    mpz_t Hm1, Hm2, gamma, delta1, delta2;
    mpz_init_set_ui(Hm1, 7);
    mpz_init_set_ui(Hm2, 5);
    mpz_init(gamma); mpz_init(delta1); mpz_init(delta2);

    dsa_signer(gamma, delta1, Hm1, a, alpha, p, q, k);
    dsa_signer(gamma, delta2, Hm2, a, alpha, p, q, k); /* même k ! */

    printf("gamma identique dans les deux signatures : ");
    afficher("gamma", gamma);  /* signal d'alarme */

    /* L'attaquant retrouve k et a */
    attaque_nonce_dsa(Hm1, Hm2, delta1, delta2, gamma, q);

    mpz_clear(p); mpz_clear(q); mpz_clear(alpha); mpz_clear(a);
    mpz_clear(beta); mpz_clear(Hm1); mpz_clear(Hm2);
    mpz_clear(gamma); mpz_clear(delta1); mpz_clear(delta2);
}


/* =============================================================================
 * BLOC 13 — RSA MANUEL
 * =============================================================================
 *
 * Génération des clés :
 *   n = p * q
 *   phi(n) = (p-1)*(q-1)
 *   Choisir e tel que pgcd(e, phi(n)) = 1
 *   d = e^{-1} mod phi(n)
 *   Clé publique : (e, n)
 *   Clé privée   : (d, n)
 *
 * Chiffrement  : c = m^e mod n
 * Déchiffrement: m = c^d mod n
 */

void rsa_generer_cles(mpz_t n, mpz_t e, mpz_t d,
                       const mpz_t p, const mpz_t q) {
    mpz_t phi, p_1, q_1;
    mpz_init(phi); mpz_init(p_1); mpz_init(q_1);

    /* n = p * q */
    mpz_mul(n, p, q);

    /* phi(n) = (p-1)*(q-1) */
    mpz_sub_ui(p_1, p, 1);
    mpz_sub_ui(q_1, q, 1);
    mpz_mul(phi, p_1, q_1);

    /* Choisir e = 65537 (standard) — vérifier pgcd(e, phi) = 1 */
    mpz_set_ui(e, 65537);

    /* d = e^{-1} mod phi(n) */
    if (!mpz_invert(d, e, phi)) {
        fprintf(stderr, "Erreur : e non inversible mod phi(n)\n");
        exit(1);
    }

    mpz_clear(phi); mpz_clear(p_1); mpz_clear(q_1);
}

void rsa_chiffrer(mpz_t c, const mpz_t m, const mpz_t e, const mpz_t n) {
    mpz_powm(c, m, e, n);         /* c = m^e mod n */
}

void rsa_dechiffrer(mpz_t m, const mpz_t c, const mpz_t d, const mpz_t n) {
    mpz_powm(m, c, d, n);         /* m = c^d mod n */
}

void exemple_rsa(void) {
    /* Petits premiers pour l'exemple (en pratique : 1024+ bits) */
    mpz_t p, q, n, e, d, m, c, m_dec;
    mpz_init_set_ui(p, 61);
    mpz_init_set_ui(q, 53);
    mpz_init(n); mpz_init(e); mpz_init(d);
    mpz_init_set_ui(m, 42);   /* message */
    mpz_init(c); mpz_init(m_dec);

    printf("\n=== RSA (p=61, q=53) ===\n");
    rsa_generer_cles(n, e, d, p, q);

    afficher("n", n);
    afficher("e", e);
    afficher("d", d);

    rsa_chiffrer(c, m, e, n);
    afficher("Chiffré c = m^e mod n", c);

    rsa_dechiffrer(m_dec, c, d, n);
    afficher("Déchiffré m = c^d mod n", m_dec);  /* doit valoir 42 */

    mpz_clear(p); mpz_clear(q); mpz_clear(n); mpz_clear(e);
    mpz_clear(d); mpz_clear(m); mpz_clear(c); mpz_clear(m_dec);
}


/* =============================================================================
 * BLOC 14 — POHLIG-HELLMAN + CRT
 * =============================================================================
 *
 * Applicable quand l'ordre n du groupe est friable (petits facteurs premiers)
 * Si n = p1^e1 * p2^e2 * ... * pk^ek avec tous pi petits :
 *   - Résoudre x mod pi pour chaque facteur (recherche exhaustive petite)
 *   - Recombiner par le Théorème Chinois des Restes (CRT)
 *
 * CRT pour deux congruences :
 *   x ≡ a1 (mod m1)
 *   x ≡ a2 (mod m2)
 *   Solution : x = a1 + m1 * ((a2 - a1) * m1^{-1} mod m2)
 */

/* Résoudre g^x ≡ b (mod p) par énumération (ordre petit) */
unsigned long log_petit_groupe(const mpz_t g, const mpz_t b,
                                const mpz_t p, unsigned long ordre) {
    mpz_t val;
    mpz_init_set_ui(val, 1);

    for (unsigned long x = 0; x <= ordre; x++) {
        if (mpz_cmp(val, b) == 0) {
            mpz_clear(val);
            return x;
        }
        mpz_mul(val, val, g);
        mpz_mod(val, val, p);
    }
    mpz_clear(val);
    return 0;  /* pas trouvé */
}

/* CRT pour deux congruences x ≡ a1 (mod m1) et x ≡ a2 (mod m2) */
void crt_deux(mpz_t x, const mpz_t a1, const mpz_t m1,
              const mpz_t a2, const mpz_t m2) {
    mpz_t inv, diff, tmp;
    mpz_init(inv); mpz_init(diff); mpz_init(tmp);

    /* x = a1 + m1 * ((a2 - a1) * m1^{-1} mod m2) */
    mpz_invert(inv, m1, m2);
    mpz_sub(diff, a2, a1);
    mpz_mod(diff, diff, m2);
    mpz_mul(tmp, diff, inv);
    mpz_mod(tmp, tmp, m2);
    mpz_mul(tmp, tmp, m1);
    mpz_add(x, a1, tmp);

    mpz_clear(inv); mpz_clear(diff); mpz_clear(tmp);
}


/* =============================================================================
 * BLOC 15 — STRUCTURE TYPE D'UN PROGRAMME TP
 * =============================================================================
 *
 * Reprend exactement la structure de logdiscret.c du TP
 * Usage : ./logdiscret p g beta
 */

int main_type_tp(int argc, char *argv[]) {

    if (argc < 4) {
        fprintf(stderr, "Usage: %s p g beta\n", argv[0]);
        return 1;
    }

    /* 1. Déclarer et initialiser les variables GMP */
    mpz_t p, g, beta, aux, ginvm;
    mpz_init(p);
    mpz_init(g);
    mpz_init(beta);
    mpz_init(aux);
    mpz_init(ginvm);

    /* 2. Lire les paramètres depuis la ligne de commande */
    mpz_set_str(p,    argv[1], 10);
    mpz_set_str(g,    argv[2], 10);
    mpz_set_str(beta, argv[3], 10);

    gmp_printf("p    = %Zd\n", p);
    gmp_printf("g    = %Zd\n", g);
    gmp_printf("beta = %Zd\n", beta);

    /* 3. Calculer m = floor(sqrt(p)) */
    mpz_t temp;
    mpz_init(temp);
    mpz_sqrt(temp, p);
    unsigned long m = mpz_get_ui(temp) + 1;
    printf("m = %lu\n", m);
    mpz_clear(temp);

    /* 4. Allouer la table de hachage */
    unsigned int table_size = 2 * m + 1;
    HashTable ht = calloc(table_size, sizeof(Maillon *));
    if (!ht) { fprintf(stderr, "Erreur allocation\n"); return 1; }

    /* 5. Phase 1 — Baby Steps */
    printf("Init Phase 1\n");
    mpz_set_ui(aux, 1);
    for (unsigned long i = 1; i < m; i++) {
        mpz_mul(aux, aux, g);
        mpz_mod(aux, aux, p);
        hash_insert(&ht, mpz_get_ui(aux), i, table_size);
    }
    printf("Phase 1 OK\n");

    /* 6. Phase 2 — Giant Steps */
    printf("Init Phase 2\n");

    /* Calculer g^{-m} mod p */
    mpz_set_ui(aux, m);
    mpz_powm(ginvm, g, aux, p);    /* ginvm = g^m mod p */
    mpz_invert(ginvm, ginvm, p);   /* ginvm = g^{-m} mod p */

    /* Parcourir beta * (g^{-m})^j */
    mpz_t z;
    mpz_init_set(z, beta);

    unsigned long logd = 0;
    unsigned long i_match;
    int trouve = 0;

    for (unsigned long j = 0; j <= m && !trouve; j++) {
        if (hash_find(&ht, mpz_get_ui(z), &i_match, table_size)) {
            logd = j * m + i_match;
            trouve = 1;
        }
        mpz_mul(z, z, ginvm);
        mpz_mod(z, z, p);
    }

    printf("Log discret : %lu\n", logd);

    /* 7. Vérification */
    mpz_t verif;
    mpz_init(verif);
    mpz_powm_ui(verif, g, logd, p);
    gmp_printf("g^logd mod p = %Zd\n", verif);
    gmp_printf("beta         = %Zd\n", beta);
    if (mpz_cmp(verif, beta) == 0)
        printf("Vérification : OK\n");

    /* 8. Libérer la mémoire */
    free(ht);
    mpz_clear(p); mpz_clear(g); mpz_clear(beta);
    mpz_clear(aux); mpz_clear(ginvm); mpz_clear(z); mpz_clear(verif);

    return 0;
}


/* =============================================================================
 * BLOC 16 — TABLEAU RÉCAPITULATIF COMPLET DES FONCTIONS GMP
 * =============================================================================
 *
 * INITIALISATION
 * ──────────────
 * mpz_init(x)                       Initialiser x à 0 (obligatoire)
 * mpz_init_set_ui(x, val)           Initialiser et affecter unsigned long
 * mpz_init_set_si(x, val)           Initialiser et affecter signed long
 * mpz_init_set_str(x, "...", base)  Initialiser depuis string (base 10, 16...)
 * mpz_init_set(x, y)                Initialiser et copier y dans x
 * mpz_clear(x)                      Libérer la mémoire (obligatoire)
 *
 * AFFECTATION (après init)
 * ────────────────────────
 * mpz_set_ui(x, val)                Affecter unsigned long
 * mpz_set_si(x, val)                Affecter signed long
 * mpz_set_str(x, "...", base)       Affecter depuis string
 * mpz_set(x, y)                     Copier y dans x
 *
 * OPÉRATIONS ARITHMÉTIQUES
 * ────────────────────────
 * mpz_add(r, a, b)                  r = a + b
 * mpz_add_ui(r, a, v)               r = a + v
 * mpz_sub(r, a, b)                  r = a - b
 * mpz_sub_ui(r, a, v)               r = a - v
 * mpz_mul(r, a, b)                  r = a * b
 * mpz_mul_ui(r, a, v)               r = a * v
 * mpz_neg(r, a)                     r = -a
 * mpz_abs(r, a)                     r = |a|
 * mpz_tdiv_q(q, a, b)               q = a / b (quotient, tronqué vers 0)
 * mpz_tdiv_r(r, a, b)               r = a mod b (reste)
 * mpz_tdiv_q_ui(q, a, v)            q = a / v
 * mpz_mod(r, a, m)                  r = a mod m (toujours >= 0)
 * mpz_divexact(r, a, b)             r = a / b exact (sans reste)
 * mpz_divexact_ui(r, a, v)          r = a / v exact
 *
 * OPÉRATIONS MODULAIRES
 * ─────────────────────
 * mpz_powm(r, b, e, m)              r = b^e mod m
 * mpz_powm_ui(r, b, e, m)           r = b^e mod m (e = unsigned long)
 * mpz_invert(r, a, m)               r = a^{-1} mod m (retourne 0 si n'existe pas)
 *
 * COMPARAISON ET TESTS
 * ────────────────────
 * mpz_cmp(a, b)                     < 0, 0, ou > 0
 * mpz_cmp_ui(a, v)                  Comparer avec unsigned long
 * mpz_cmp_si(a, v)                  Comparer avec signed long
 * mpz_sgn(a)                        Signe : -1, 0, ou 1
 * mpz_odd_p(a)                      1 si a est impair
 * mpz_even_p(a)                     1 si a est pair
 * mpz_divisible_p(a, b)             1 si b divise a
 * mpz_divisible_ui_p(a, v)          1 si v divise a
 *
 * PGCD ET ARITHMÉTIQUE
 * ────────────────────
 * mpz_gcd(g, a, b)                  g = pgcd(a, b)
 * mpz_gcd_ui(g, a, v)               g = pgcd(a, v)
 * mpz_gcdext(g, s, t, a, b)         g = a*s + b*t (Bézout)
 * mpz_lcm(r, a, b)                  r = ppcm(a, b)
 *
 * RACINE ET PRIMALITÉ
 * ───────────────────
 * mpz_sqrt(r, a)                    r = floor(sqrt(a))
 * mpz_sqrtrem(r, rem, a)            r = floor(sqrt(a)), rem = a - r^2
 * mpz_probab_prime_p(p, reps)       Test primalité probabiliste
 *                                   retourne 0 (composé), 1 (prob premier), 2 (premier)
 *
 * CONVERSION
 * ──────────
 * mpz_get_ui(a)                     Retourne unsigned long (tronqué si trop grand)
 * mpz_get_si(a)                     Retourne signed long
 *
 * AFFICHAGE
 * ─────────
 * gmp_printf("... %Zd ...", x)      Afficher mpz_t (%Zd = entier GMP)
 * gmp_fprintf(f, "%Zd", x)          Écrire dans fichier
 * mpz_out_str(stdout, base, x)       Afficher en base donnée
 *
 * HASHUTILS.H (TP)
 * ────────────────
 * hash(z, size)                     Calculer l'index de hachage de z
 * hash_insert(&ht, gj, j, size)     Insérer (gj, j) dans la table
 * hash_find(&ht, val, &j, size)     Chercher val, retourne 1 si trouvé
 *
 * COMPILATION
 * ───────────
 * gcc fichier.c -o prog -lgmp -lm
 * ./prog p g beta
 */


/* =============================================================================
 * BLOC 17 — PIÈGES CLASSIQUES À L'EXAMEN
 * =============================================================================
 *
 * PIÈGE 1 — Oublier mpz_init
 *   ✗  mpz_t x;  mpz_set_ui(x, 5);         → comportement indéfini
 *   ✓  mpz_t x;  mpz_init(x);  mpz_set_ui(x, 5);
 *
 * PIÈGE 2 — Oublier mpz_clear → fuite mémoire
 *   Chaque mpz_init doit avoir un mpz_clear correspondant
 *
 * PIÈGE 3 — Ne pas vérifier le retour de mpz_invert
 *   ✗  mpz_invert(inv, a, n);               → silencieux si pas d'inverse
 *   ✓  if (!mpz_invert(inv, a, n)) { ... } → traiter l'erreur
 *
 * PIÈGE 4 — mpz_get_ui tronque les grands entiers
 *   Ne fonctionne que si la valeur tient dans unsigned long (64 bits max)
 *   Pour afficher : gmp_printf("%Zd", x)
 *
 * PIÈGE 5 — Exposant négatif dans mpz_powm
 *   GMP n'accepte pas les exposants négatifs dans mpz_powm
 *   ✗  mpz_powm(r, g, -m, p)               → erreur
 *   ✓  mpz_powm_ui(gm, g, m, p);           → g^m mod p
 *      mpz_invert(ginvm, gm, p);            → g^{-m} mod p
 *
 * PIÈGE 6 — Ordre des arguments dans hash_insert
 *   hash_insert(&ht, gj, j, size)
 *   Le premier argument est le POINTEUR sur la table : &ht
 *
 * PIÈGE 7 — Dans BSGS, commencer la Phase 1 à i=1 et non i=0
 *   g^0 = 1 peut créer une fausse collision
 *   Commencer à i=1 : for (i = 1; i < m; i++) ...
 *
 * PIÈGE 8 — Dans DSA, réutiliser k pour deux signatures
 *   gamma sera identique → l'attaquant peut retrouver k et a
 *   k doit être ALÉATOIRE et UNIQUE pour chaque signature
 *
 * PIÈGE 9 — Confondre ax ≡ b (mod n) et a^x ≡ b (mod p)
 *   Linéaire  : résolu par Euclide étendu en O(log n)   → FACILE
 *   Log discret: résolu par BSGS en O(sqrt(p))           → DIFFICILE
 *
 * PIÈGE 10 — Oublier free(ht) pour la table de hachage
 *   La table est allouée avec calloc, il faut libérer avec free
 *
 * PIÈGE 11 — mpz_mod vs mpz_tdiv_r
 *   mpz_mod(r, a, m)    : r toujours positif (comme % en mathématiques)
 *   mpz_tdiv_r(r, a, m) : r peut être négatif si a est négatif
 *   Toujours utiliser mpz_mod pour les calculs modulaires
 *
 * PIÈGE 12 — Dans ElGamal, le chiffré (c1, c2) est deux fois plus long que m
 *   C'est inhérent au schéma — pas une erreur
 */


/* =============================================================================
 * MAIN — LANCER TOUS LES EXEMPLES
 * ============================================================================= */

int main(int argc, char *argv[]) {

    printf("========================================\n");
    printf(" RÉFÉRENCE CRYPTOGRAPHIE C (GMP uniquement)\n");
    printf("========================================\n\n");

    /* Si des arguments sont passés → mode TP (logarithme discret) */
    if (argc == 4) {
        return main_type_tp(argc, argv);
    }

    /* Sinon → lancer les exemples */
    printf("--- BLOC 1 : Init et affectation GMP ---\n");
    exemple_init_et_affectation();

    printf("\n--- BLOC 2 : Arithmétique ---\n");
    exemple_arithmetique();

    printf("\n--- BLOC 3 : Inverse modulaire ---\n");
    exemple_inverse_modulaire();

    printf("\n--- BLOC 4 : Équation linéaire ax≡b(mod n) ---\n");
    exemple_eq_lineaire();

    printf("\n--- BLOC 5 : Exponentiation modulaire ---\n");
    exemple_exp_modulaire();

    printf("\n--- BLOC 6 : Baby-Step Giant-Step ---\n");
    exemple_bsgs();

    printf("\n--- BLOC 7 : Table de hachage ---\n");
    exemple_hashtable();

    printf("\n--- BLOC 9 : Diffie-Hellman ---\n");
    exemple_dh();

    printf("\n--- BLOC 10 : ElGamal ---\n");
    exemple_elgamal();

    printf("\n--- BLOC 11 : DSA ---\n");
    exemple_dsa();

    printf("\n--- BLOC 12 : Attaque nonce DSA ---\n");
    exemple_attaque_dsa();

    printf("\n--- BLOC 13 : RSA ---\n");
    exemple_rsa();

    return 0;
}