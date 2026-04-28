/*
 * =============================================================================
 * RÉFÉRENCE COMPLÈTE — CRYPTOGRAPHIE EN C
 * Bibliothèques : OpenSSL (libssl, libcrypto) + GMP (libgmp)
 * Compilation : 
 * gcc crypto_reference.c -o crypto_reference -lssl -lcrypto -lgmp -lm
 * =============================================================================
 *
 * SOMMAIRE :
 * BLOC  1 — Fonctions de hachage (SHA256, SHA512, SHA1, MD5)
 * BLOC  2 — HMAC
 * BLOC  3 — Génération aléatoire sécurisée
 * BLOC  4 — Opérations sur les grands entiers (GMP)
 * BLOC  5 — Exponentiation modulaire
 * BLOC  6 — Inverse modulaire
 * BLOC  7 — PGCD et équations linéaires modulaires
 * BLOC  8 — Baby-Step Giant-Step (logarithme discret)
 * BLOC  9 — Table de hachage pour BSGS (structure du TP)
 * BLOC 10 — Diffie-Hellman manuel avec GMP
 * BLOC 11 — ElGamal manuel avec GMP
 * BLOC 12 — DSA manuel avec GMP
 * BLOC 13 — RSA manuel avec GMP
 * BLOC 14 — AES-CBC avec OpenSSL
 * BLOC 15 — Chiffrement hybride RSA + AES
 * BLOC 16 — Signature RSA-PSS avec OpenSSL
 * BLOC 17 — Attaque réutilisation nonce DSA
 * BLOC 18 — Utilitaires : affichage, conversion hex
 * BLOC 19 — Pièges et erreurs classiques
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* OpenSSL */
#include <openssl/sha.h>
#include <openssl/md5.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/rsa.h>
#include <openssl/dsa.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/aes.h>

/* GMP */
#include <gmp.h>


/* =============================================================================
 * UTILITAIRES GÉNÉRAUX
 * ============================================================================= */

/* Afficher un tableau de bytes en hexadécimal */
void afficher_hex(const char *label, const unsigned char *data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; i++)
        printf("%02x", data[i]);
    printf("\n");
}

/* Afficher un grand entier GMP */
void afficher_mpz(const char *label, const mpz_t n) {
    gmp_printf("%s: %Zd\n", label, n);
}

/* Convertir un tableau de bytes en entier (petit exemple) */
unsigned long bytes_to_ulong(const unsigned char *b, size_t len) {
    unsigned long r = 0;
    for (size_t i = 0; i < len; i++)
        r = (r << 8) | b[i];
    return r;
}


/* =============================================================================
 * BLOC 1 — FONCTIONS DE HACHAGE
 * ============================================================================= */

/*
 * SHA-256 : produit 32 octets = 256 bits
 * SHA-512 : produit 64 octets = 512 bits
 * SHA-1   : produit 20 octets = 160 bits (obsolète)
 * MD5     : produit 16 octets = 128 bits (cassé)
 */

/* SHA-256 avec l'API bas niveau */
void sha256_simple(const unsigned char *message, size_t len,
                   unsigned char sortie[SHA256_DIGEST_LENGTH]) {
    SHA256(message, len, sortie);
    /* SHA256_DIGEST_LENGTH = 32 */
}

/* SHA-256 avec l'API EVP (recommandée) */
void sha256_evp(const unsigned char *message, size_t len,
                unsigned char *sortie, unsigned int *sortie_len) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
    EVP_DigestUpdate(ctx, message, len);
    EVP_DigestFinal_ex(ctx, sortie, sortie_len);
    EVP_MD_CTX_free(ctx);
}

/* SHA-256 en plusieurs fois (update) */
void sha256_incremental(const unsigned char *partie1, size_t len1,
                        const unsigned char *partie2, size_t len2,
                        unsigned char sortie[SHA256_DIGEST_LENGTH]) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, partie1, len1);
    SHA256_Update(&ctx, partie2, len2);
    SHA256_Final(sortie, &ctx);
}

/* SHA-512 */
void sha512_simple(const unsigned char *message, size_t len,
                   unsigned char sortie[SHA512_DIGEST_LENGTH]) {
    SHA512(message, len, sortie);
    /* SHA512_DIGEST_LENGTH = 64 */
}

/* SHA-1 */
void sha1_simple(const unsigned char *message, size_t len,
                 unsigned char sortie[SHA_DIGEST_LENGTH]) {
    SHA1(message, len, sortie);
    /* SHA_DIGEST_LENGTH = 20 */
}

/* MD5 */
void md5_simple(const unsigned char *message, size_t len,
                unsigned char sortie[MD5_DIGEST_LENGTH]) {
    MD5(message, len, sortie);
    /* MD5_DIGEST_LENGTH = 16 */
}

/* Hacher un fichier par blocs */
int hacher_fichier_sha256(const char *chemin,
                           unsigned char sortie[SHA256_DIGEST_LENGTH]) {
    FILE *f = fopen(chemin, "rb");
    if (!f) return -1;

    SHA256_CTX ctx;
    SHA256_Init(&ctx);

    unsigned char tampon[4096];
    size_t lu;
    while ((lu = fread(tampon, 1, sizeof(tampon), f)) > 0)
        SHA256_Update(&ctx, tampon, lu);

    SHA256_Final(sortie, &ctx);
    fclose(f);
    return 0;
}

/* Exemple d'utilisation du BLOC 1 */
void exemple_hachage(void) {
    unsigned char message[] = "Bonjour crypto";
    unsigned char h[SHA256_DIGEST_LENGTH];

    sha256_simple(message, strlen((char*)message), h);
    afficher_hex("SHA-256", h, SHA256_DIGEST_LENGTH);

    unsigned char h512[SHA512_DIGEST_LENGTH];
    sha512_simple(message, strlen((char*)message), h512);
    afficher_hex("SHA-512", h512, SHA512_DIGEST_LENGTH);
}


/* =============================================================================
 * BLOC 2 — HMAC
 * ============================================================================= */

/*
 * HMAC-SHA256 : authentification avec clé secrète partagée
 * Garantit intégrité + authentification (contrairement au simple hachage)
 */

void hmac_sha256(const unsigned char *cle, size_t cle_len,
                 const unsigned char *message, size_t msg_len,
                 unsigned char *sortie, unsigned int *sortie_len) {
    HMAC(EVP_sha256(), cle, (int)cle_len, message, msg_len,
         sortie, sortie_len);
}

/* Vérifier un HMAC (comparaison à temps constant pour éviter les attaques timing) */
int verifier_hmac(const unsigned char *cle, size_t cle_len,
                  const unsigned char *message, size_t msg_len,
                  const unsigned char *mac_recu, size_t mac_len) {
    unsigned char mac_calcule[EVP_MAX_MD_SIZE];
    unsigned int mac_calcule_len;
    HMAC(EVP_sha256(), cle, (int)cle_len, message, msg_len,
         mac_calcule, &mac_calcule_len);
    if (mac_calcule_len != mac_len) return 0;
    /* CRYPTO_memcmp : comparaison à temps constant */
    return (CRYPTO_memcmp(mac_calcule, mac_recu, mac_len) == 0);
}

void exemple_hmac(void) {
    unsigned char cle[] = "cle_secrete";
    unsigned char msg[] = "message a authentifier";
    unsigned char mac[EVP_MAX_MD_SIZE];
    unsigned int mac_len;

    hmac_sha256(cle, strlen((char*)cle), msg, strlen((char*)msg), mac, &mac_len);
    afficher_hex("HMAC-SHA256", mac, mac_len);

    int ok = verifier_hmac(cle, strlen((char*)cle), msg, strlen((char*)msg),
                            mac, mac_len);
    printf("HMAC valide: %s\n", ok ? "OUI" : "NON");
}


/* =============================================================================
 * BLOC 3 — GÉNÉRATION ALÉATOIRE SÉCURISÉE
 * ============================================================================= */

/* Générer des octets aléatoires cryptographiquement sûrs */
void generer_aleatoire(unsigned char *tampon, size_t len) {
    if (RAND_bytes(tampon, (int)len) != 1) {
        fprintf(stderr, "Erreur RAND_bytes\n");
        exit(1);
    }
}

/* Générer un IV pour AES-CBC */
void generer_iv(unsigned char iv[16]) {
    generer_aleatoire(iv, 16);
}

/* Générer une clé AES-128 */
void generer_cle_aes128(unsigned char cle[16]) {
    generer_aleatoire(cle, 16);
}

void exemple_aleatoire(void) {
    unsigned char iv[16], cle[16];
    generer_iv(iv);
    generer_cle_aes128(cle);
    afficher_hex("IV aléatoire ", iv, 16);
    afficher_hex("Clé aléatoire", cle, 16);
}


/* =============================================================================
 * BLOC 4 — GRANDS ENTIERS AVEC GMP
 * ============================================================================= */

/*
 * GMP : GNU Multiple Precision Arithmetic Library
 * Type principal : mpz_t (entier de précision arbitraire)
 *
 * RÈGLE : toujours mpz_init avant d'utiliser, mpz_clear après
 */

void exemple_gmp_bases(void) {
    mpz_t a, b, r;
    mpz_init(a);
    mpz_init(b);
    mpz_init(r);

    /* Assigner des valeurs */
    mpz_set_ui(a, 12345);                    /* depuis unsigned long */
    mpz_set_str(b, "99999999999999999", 10); /* depuis string, base 10 */

    /* Opérations de base */
    mpz_add(r, a, b);          /* r = a + b */
    mpz_sub(r, a, b);          /* r = a - b */
    mpz_mul(r, a, b);          /* r = a * b */
    mpz_tdiv_q(r, a, b);       /* r = a / b (quotient) */
    mpz_mod(r, a, b);          /* r = a mod b */

    /* Comparaison */
    if (mpz_cmp(a, b) < 0)    printf("a < b\n");
    if (mpz_cmp_ui(a, 0) > 0) printf("a > 0\n");

    /* Affichage */
    gmp_printf("a = %Zd\n", a);
    gmp_printf("b = %Zd\n", b);

    /* Libération */
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(r);
}

/* Initialiser plusieurs variables GMP en une fois */
void init_mpz(int n, ...) {
    /* Pattern courant dans les TPs : */
    /* mpz_t a, b, c;                 */
    /* mpz_init(a); mpz_init(b); mpz_init(c); */
}

/* Lire les paramètres depuis la ligne de commande */
void lire_params_cli(int argc, char *argv[],
                     mpz_t p, mpz_t g, mpz_t beta) {
    mpz_set_str(p,    argv[1], 10);
    mpz_set_str(g,    argv[2], 10);
    mpz_set_str(beta, argv[3], 10);
}


/* =============================================================================
 * BLOC 5 — EXPONENTIATION MODULAIRE
 * ============================================================================= */

/*
 * Calcul de base^exposant mod modulo
 * GMP fournit mpz_powm directement
 */

void exemple_exp_modulaire(void) {
    mpz_t base, expo, mod, resultat;
    mpz_init(base);
    mpz_init(expo);
    mpz_init(mod);
    mpz_init(resultat);

    mpz_set_str(base, "2",   10);
    mpz_set_str(expo, "100", 10);
    mpz_set_str(mod,  "997", 10);

    /* resultat = base^expo mod mod */
    mpz_powm(resultat, base, expo, mod);
    gmp_printf("2^100 mod 997 = %Zd\n", resultat);

    /* Avec un exposant unsigned long */
    mpz_powm_ui(resultat, base, 100UL, mod);

    /* Inverse modulaire : base^{-1} mod mod */
    /* Équivalent à pow(base, -1, mod) en Python */
    mpz_t inv;
    mpz_init(inv);
    if (mpz_invert(inv, base, mod)) {
        gmp_printf("Inverse de 2 mod 997 = %Zd\n", inv);
    } else {
        printf("Pas d'inverse (non premier entre eux)\n");
    }

    mpz_clear(base); mpz_clear(expo); mpz_clear(mod);
    mpz_clear(resultat); mpz_clear(inv);
}

/* Calculer g^{-m} mod p (utile pour BSGS) */
void calculer_g_inv_m(mpz_t resultat, const mpz_t g, unsigned long m,
                       const mpz_t p) {
    mpz_t g_m;
    mpz_init(g_m);
    mpz_powm_ui(g_m, g, m, p);      /* g_m = g^m mod p */
    mpz_invert(resultat, g_m, p);    /* resultat = g^{-m} mod p */
    mpz_clear(g_m);
}


/* =============================================================================
 * BLOC 6 — INVERSE MODULAIRE
 * ============================================================================= */

/*
 * Trouver x tel que a*x ≡ 1 (mod n)
 * Existe si et seulement si pgcd(a, n) = 1
 */

int inverse_modulaire(mpz_t resultat, const mpz_t a, const mpz_t n) {
    return mpz_invert(resultat, a, n);
    /* retourne 1 si l'inverse existe, 0 sinon */
    /* le résultat est dans resultat */
}

void exemple_inverse(void) {
    mpz_t a, n, inv;
    mpz_init_set_ui(a, 3);
    mpz_init_set_ui(n, 11);
    mpz_init(inv);

    if (inverse_modulaire(inv, a, n))
        gmp_printf("Inverse de 3 mod 11 = %Zd\n", inv);  /* 4 */

    mpz_clear(a); mpz_clear(n); mpz_clear(inv);
}


/* =============================================================================
 * BLOC 7 — PGCD ET ÉQUATIONS LINÉAIRES MODULAIRES
 * ============================================================================= */

/*
 * Résoudre ax ≡ b (mod n)
 * g = pgcd(a, n)
 * Si g ne divise pas b → aucune solution
 * Si g divise b → exactement g solutions
 */

void pgcd_gmp(mpz_t g, const mpz_t a, const mpz_t b) {
    mpz_gcd(g, a, b);
}

int nb_solutions_lineaire(const mpz_t a, const mpz_t b, const mpz_t n) {
    mpz_t g;
    mpz_init(g);
    mpz_gcd(g, a, n);

    int nb;
    if (mpz_divisible_p(b, g))   /* g divise b ? */
        nb = (int)mpz_get_ui(g);
    else
        nb = 0;

    mpz_clear(g);
    return nb;
}

/* Trouver une solution x0 de ax ≡ b (mod n) */
int resoudre_lineaire(mpz_t x0, const mpz_t a, const mpz_t b, const mpz_t n) {
    mpz_t g, a_red, b_red, n_red, inv;
    mpz_init(g); mpz_init(a_red); mpz_init(b_red);
    mpz_init(n_red); mpz_init(inv);

    mpz_gcd(g, a, n);

    if (!mpz_divisible_p(b, g)) {
        mpz_clear(g); mpz_clear(a_red); mpz_clear(b_red);
        mpz_clear(n_red); mpz_clear(inv);
        return 0;   /* aucune solution */
    }

    /* Diviser par g */
    mpz_divexact(a_red, a, g);
    mpz_divexact(b_red, b, g);
    mpz_divexact(n_red, n, g);

    /* x0 = b_red * a_red^{-1} mod n_red */
    mpz_invert(inv, a_red, n_red);
    mpz_mul(x0, b_red, inv);
    mpz_mod(x0, x0, n_red);

    mpz_clear(g); mpz_clear(a_red); mpz_clear(b_red);
    mpz_clear(n_red); mpz_clear(inv);
    return (int)mpz_get_ui(g);  /* nombre de solutions */
}

void exemple_eq_lineaire(void) {
    /* Résoudre 6x ≡ 4 (mod 8) */
    mpz_t a, b, n, x0, g;
    mpz_init_set_ui(a, 6);
    mpz_init_set_ui(b, 4);
    mpz_init_set_ui(n, 8);
    mpz_init(x0);
    mpz_init(g);

    mpz_gcd(g, a, n);
    gmp_printf("pgcd(6, 8) = %Zd\n", g);   /* 2 */

    int nb = nb_solutions_lineaire(a, b, n);
    printf("Nombre de solutions : %d\n", nb);

    resoudre_lineaire(x0, a, b, n);
    gmp_printf("x0 = %Zd\n", x0);           /* 2 */

    mpz_clear(a); mpz_clear(b); mpz_clear(n);
    mpz_clear(x0); mpz_clear(g);
}


/* =============================================================================
 * BLOC 8 — BABY-STEP GIANT-STEP
 * Structure du TP : table de hachage + listes chaînées
 * ============================================================================= */

/*
 * Résoudre g^x ≡ beta (mod p)
 * Complexité : O(sqrt(p)) en temps et mémoire
 *
 * Structure de données du TP :
 */

typedef struct Maillon {
    unsigned long gj;        /* valeur g^j mod p */
    unsigned long j;         /* exposant j */
    struct Maillon *next;
} Maillon;

typedef Maillon **HashTable;

/* Fonction de hachage : XOR des 32 bits de poids fort et faible */
unsigned int hash_bsgs(unsigned long z, unsigned int taille) {
    unsigned long z1 = z >> 32;
    unsigned long z2 = z & 0xFFFFFFFF;
    return (unsigned int)((z1 ^ z2) % taille);
}

/* Insérer dans la table */
void hash_inserer(HashTable *ht, unsigned long gj, unsigned long j,
                  unsigned int taille) {
    unsigned int idx = hash_bsgs(gj, taille);
    Maillon *m = calloc(1, sizeof(Maillon));
    m->gj   = gj;
    m->j    = j;
    m->next = (*ht)[idx];
    (*ht)[idx] = m;
}

/* Chercher dans la table */
int hash_chercher(HashTable *ht, unsigned long val, unsigned long *j_out,
                  unsigned int taille) {
    unsigned int idx = hash_bsgs(val, taille);
    Maillon *cur = (*ht)[idx];
    while (cur != NULL) {
        if (cur->gj == val) {
            *j_out = cur->j;
            return 1;
        }
        cur = cur->next;
    }
    return 0;
}

/* Libérer la table */
void hash_liberer(HashTable ht, unsigned int taille) {
    for (unsigned int i = 0; i < taille; i++) {
        Maillon *cur = ht[i];
        while (cur) {
            Maillon *next = cur->next;
            free(cur);
            cur = next;
        }
    }
    free(ht);
}

/* BSGS complet avec GMP (pour grands entiers) */
unsigned long bsgs_gmp(const mpz_t g, const mpz_t beta, const mpz_t p) {
    mpz_t racine;
    mpz_init(racine);
    mpz_sqrt(racine, p);
    unsigned long m = mpz_get_ui(racine) + 1;
    mpz_clear(racine);

    unsigned int taille = 2 * m + 1;
    HashTable ht = calloc(taille, sizeof(Maillon *));

    /* Phase 1 — Baby Steps : stocker g^j pour j = 1..m */
    mpz_t aux;
    mpz_init_set_ui(aux, 1);
    for (unsigned long j = 1; j < m; j++) {
        mpz_mul(aux, aux, g);
        mpz_mod(aux, aux, p);
        hash_inserer(&ht, mpz_get_ui(aux), j, taille);
    }

    /* Phase 2 — Giant Steps */
    mpz_t ginvm, z;
    mpz_init(ginvm);
    mpz_init_set(z, beta);

    /* Calculer g^{-m} mod p */
    mpz_set_ui(aux, m);
    mpz_powm(ginvm, g, aux, p);
    mpz_invert(ginvm, ginvm, p);

    unsigned long x = 0;
    unsigned long j_match;
    for (unsigned long i = 0; i <= m; i++) {
        unsigned long z_val = mpz_get_ui(z);
        if (hash_chercher(&ht, z_val, &j_match, taille)) {
            x = i * m + j_match;
            break;
        }
        mpz_mul(z, z, ginvm);
        mpz_mod(z, z, p);
    }

    hash_liberer(ht, taille);
    mpz_clear(aux); mpz_clear(ginvm); mpz_clear(z);
    return x;
}

/* Recherche naïve (pour petits p, pour tester) */
unsigned long recherche_naive(const mpz_t g, const mpz_t beta,
                               const mpz_t p) {
    mpz_t b, x, p_1;
    mpz_init(b);
    mpz_init_set_ui(x, 0);
    mpz_init(p_1);
    mpz_sub_ui(p_1, p, 1);

    while (mpz_cmp(x, p_1) < 0) {
        mpz_powm(b, g, x, p);
        if (mpz_cmp(b, beta) == 0) {
            unsigned long res = mpz_get_ui(x);
            mpz_clear(b); mpz_clear(x); mpz_clear(p_1);
            return res;
        }
        mpz_add_ui(x, x, 1);
    }
    mpz_clear(b); mpz_clear(x); mpz_clear(p_1);
    return 0;
}

void exemple_bsgs(void) {
    mpz_t p, g, beta;
    mpz_init_set_str(p,    "1934459", 10);
    mpz_init_set_str(g,    "762973",  10);
    mpz_init_set_str(beta, "1191663", 10);

    unsigned long x = bsgs_gmp(g, beta, p);
    printf("Log discret x = %lu\n", x);

    /* Vérification */
    mpz_t verif;
    mpz_init(verif);
    mpz_powm_ui(verif, g, x, p);
    gmp_printf("g^x mod p = %Zd\n", verif);
    gmp_printf("beta      = %Zd\n", beta);

    mpz_clear(p); mpz_clear(g); mpz_clear(beta); mpz_clear(verif);
}


/* =============================================================================
 * BLOC 9 — ÉCHANGE DIFFIE-HELLMAN AVEC GMP
 * ============================================================================= */

/*
 * Protocole DH :
 * Paramètres publics : p, g
 * Alice choisit x, envoie g^x mod p
 * Bob   choisit y, envoie g^y mod p
 * Secret partagé : K = g^{xy} mod p
 */

void diffie_hellman_demo(const mpz_t p, const mpz_t g,
                          unsigned long x, unsigned long y) {
    mpz_t A, B, K_alice, K_bob;
    mpz_init(A); mpz_init(B);
    mpz_init(K_alice); mpz_init(K_bob);

    /* Alice calcule A = g^x mod p */
    mpz_powm_ui(A, g, x, p);

    /* Bob calcule B = g^y mod p */
    mpz_powm_ui(B, g, y, p);

    gmp_printf("Alice envoie A = g^x = %Zd\n", A);
    gmp_printf("Bob   envoie B = g^y = %Zd\n", B);

    /* Alice : K = B^x mod p */
    mpz_powm_ui(K_alice, B, x, p);

    /* Bob : K = A^y mod p */
    mpz_powm_ui(K_bob, A, y, p);

    gmp_printf("Secret Alice : %Zd\n", K_alice);
    gmp_printf("Secret Bob   : %Zd\n", K_bob);

    if (mpz_cmp(K_alice, K_bob) == 0)
        printf("Secrets identiques : OK\n");

    mpz_clear(A); mpz_clear(B);
    mpz_clear(K_alice); mpz_clear(K_bob);
}

void exemple_dh(void) {
    mpz_t p, g;
    mpz_init_set_ui(p, 11);
    mpz_init_set_ui(g, 2);
    diffie_hellman_demo(p, g, 3, 4);
    mpz_clear(p); mpz_clear(g);
}


/* =============================================================================
 * BLOC 10 — ELGAMAL AVEC GMP
 * ============================================================================= */

/*
 * Paramètres : p premier, a générateur, x secret, b = a^x mod p
 * Chiffrement de m avec k aléatoire :
 *   c1 = a^k mod p
 *   c2 = m * b^k mod p
 * Déchiffrement :
 *   m = c2 * c1^{-x} mod p
 */

void elgamal_chiffrer(mpz_t c1, mpz_t c2,
                       const mpz_t m,
                       const mpz_t a, const mpz_t b, const mpz_t p,
                       unsigned long k) {
    mpz_t bk;
    mpz_init(bk);

    mpz_powm_ui(c1, a, k, p);       /* c1 = a^k mod p */
    mpz_powm_ui(bk, b, k, p);       /* bk = b^k mod p */
    mpz_mul(c2, m, bk);             /* c2 = m * b^k */
    mpz_mod(c2, c2, p);             /* c2 mod p */

    mpz_clear(bk);
}

void elgamal_dechiffrer(mpz_t m_dec,
                         const mpz_t c1, const mpz_t c2,
                         const mpz_t x, const mpz_t p) {
    mpz_t s, s_inv;
    mpz_init(s); mpz_init(s_inv);

    mpz_powm(s, c1, x, p);          /* s = c1^x mod p */
    mpz_invert(s_inv, s, p);        /* s_inv = s^{-1} mod p */
    mpz_mul(m_dec, c2, s_inv);      /* m = c2 * s_inv */
    mpz_mod(m_dec, m_dec, p);

    mpz_clear(s); mpz_clear(s_inv);
}

void exemple_elgamal(void) {
    mpz_t p, a, x, b, m, c1, c2, m_dec;

    mpz_init_set_ui(p, 11);     /* premier */
    mpz_init_set_ui(a, 2);      /* générateur */
    mpz_init_set_ui(x, 3);      /* clé secrète */
    mpz_init(b);
    mpz_init_set_ui(m, 5);      /* message */
    mpz_init(c1); mpz_init(c2); mpz_init(m_dec);

    mpz_powm_ui(b, a, 3, p);   /* b = a^x mod p = 2^3 mod 11 = 8 */

    elgamal_chiffrer(c1, c2, m, a, b, p, 2);    /* k=2 */
    gmp_printf("Chiffré : c1=%Zd, c2=%Zd\n", c1, c2);

    elgamal_dechiffrer(m_dec, c1, c2, x, p);
    gmp_printf("Déchiffré : m=%Zd\n", m_dec);   /* doit être 5 */

    mpz_clear(p); mpz_clear(a); mpz_clear(x); mpz_clear(b);
    mpz_clear(m); mpz_clear(c1); mpz_clear(c2); mpz_clear(m_dec);
}


/* =============================================================================
 * BLOC 11 — DSA MANUEL AVEC GMP
 * ============================================================================= */

/*
 * Paramètres : p, q (q divise p-1), alpha, beta = alpha^a mod p, H
 * Clé secrète : a
 * Signature de m :
 *   gamma = (alpha^k mod p) mod q
 *   delta = (H(m) + a*gamma) * k^{-1} mod q
 * Vérification :
 *   e1 = H(m) * delta^{-1} mod q
 *   e2 = gamma * delta^{-1} mod q
 *   v  = (alpha^e1 * beta^e2 mod p) mod q
 *   Accepter si v == gamma
 */

void dsa_signer(mpz_t gamma, mpz_t delta,
                const mpz_t Hm,
                const mpz_t a, const mpz_t alpha,
                const mpz_t p, const mpz_t q,
                unsigned long k) {
    mpz_t k_mpz, k_inv, tmp;
    mpz_init(k_mpz); mpz_init(k_inv); mpz_init(tmp);

    mpz_set_ui(k_mpz, k);

    /* gamma = (alpha^k mod p) mod q */
    mpz_powm(gamma, alpha, k_mpz, p);
    mpz_mod(gamma, gamma, q);

    /* delta = (H(m) + a*gamma) * k^{-1} mod q */
    mpz_mul(tmp, a, gamma);        /* tmp = a * gamma */
    mpz_add(tmp, Hm, tmp);         /* tmp = H(m) + a*gamma */
    mpz_mod(tmp, tmp, q);
    mpz_invert(k_inv, k_mpz, q);  /* k_inv = k^{-1} mod q */
    mpz_mul(delta, tmp, k_inv);    /* delta = tmp * k_inv */
    mpz_mod(delta, delta, q);

    mpz_clear(k_mpz); mpz_clear(k_inv); mpz_clear(tmp);
}

int dsa_verifier(const mpz_t Hm, const mpz_t gamma, const mpz_t delta,
                  const mpz_t alpha, const mpz_t beta,
                  const mpz_t p, const mpz_t q) {
    mpz_t d_inv, e1, e2, t1, t2, v;
    mpz_init(d_inv); mpz_init(e1); mpz_init(e2);
    mpz_init(t1); mpz_init(t2); mpz_init(v);

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
    mpz_clear(t1); mpz_clear(t2); mpz_clear(v);
    return ok;
}


/* =============================================================================
 * BLOC 12 — ATTAQUE : RÉUTILISATION DU NONCE DSA
 * ============================================================================= */

/*
 * Si le même k est utilisé pour deux signatures (gamma est identique) :
 *   k = (H(m1) - H(m2)) * (delta1 - delta2)^{-1} mod q
 *   a = (delta1*k - H(m1)) * gamma^{-1} mod q
 */

void retrouver_cle_dsa(mpz_t a_retrouve,
                        const mpz_t Hm1, const mpz_t Hm2,
                        const mpz_t delta1, const mpz_t delta2,
                        const mpz_t gamma, const mpz_t q) {
    mpz_t k, num, den, tmp;
    mpz_init(k); mpz_init(num); mpz_init(den); mpz_init(tmp);

    /* k = (H(m1) - H(m2)) * (delta1 - delta2)^{-1} mod q */
    mpz_sub(num, Hm1, Hm2);
    mpz_mod(num, num, q);
    mpz_sub(den, delta1, delta2);
    mpz_mod(den, den, q);
    mpz_invert(den, den, q);
    mpz_mul(k, num, den);
    mpz_mod(k, k, q);

    /* a = (delta1*k - H(m1)) * gamma^{-1} mod q */
    mpz_mul(tmp, delta1, k);
    mpz_sub(tmp, tmp, Hm1);
    mpz_mod(tmp, tmp, q);
    mpz_t g_inv;
    mpz_init(g_inv);
    mpz_invert(g_inv, gamma, q);
    mpz_mul(a_retrouve, tmp, g_inv);
    mpz_mod(a_retrouve, a_retrouve, q);

    gmp_printf("k retrouvé : %Zd\n", k);
    gmp_printf("a retrouvé : %Zd\n", a_retrouve);

    mpz_clear(k); mpz_clear(num); mpz_clear(den);
    mpz_clear(tmp); mpz_clear(g_inv);
}


/* =============================================================================
 * BLOC 13 — AES-CBC AVEC OPENSSL
 * ============================================================================= */

/*
 * AES-128-CBC :
 *   Clé : 16 octets
 *   IV  : 16 octets (doit être aléatoire pour chaque chiffrement)
 *   Entrée : doit être un multiple de 16 (avec padding PKCS7)
 */

/* Chiffrer avec AES-128-CBC */
int aes_cbc_chiffrer(const unsigned char *clair,  int clair_len,
                      const unsigned char *cle,    const unsigned char *iv,
                      unsigned char *chiffre,      int *chiffre_len) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;

    int len, total = 0;

    EVP_EncryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, cle, iv);
    EVP_EncryptUpdate(ctx, chiffre, &len, clair, clair_len);
    total += len;
    EVP_EncryptFinal_ex(ctx, chiffre + total, &len);
    total += len;

    *chiffre_len = total;
    EVP_CIPHER_CTX_free(ctx);
    return 0;
}

/* Déchiffrer avec AES-128-CBC */
int aes_cbc_dechiffrer(const unsigned char *chiffre, int chiffre_len,
                        const unsigned char *cle,     const unsigned char *iv,
                        unsigned char *clair,          int *clair_len) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;

    int len, total = 0;

    EVP_DecryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, cle, iv);
    EVP_DecryptUpdate(ctx, clair, &len, chiffre, chiffre_len);
    total += len;
    EVP_DecryptFinal_ex(ctx, clair + total, &len);
    total += len;

    *clair_len = total;
    EVP_CIPHER_CTX_free(ctx);
    return 0;
}

void exemple_aes_cbc(void) {
    unsigned char cle[16], iv[16];
    generer_aleatoire(cle, 16);
    generer_aleatoire(iv,  16);

    unsigned char message[] = "Message secret de 32 caracteres!";
    unsigned char chiffre[256];
    unsigned char clair[256];
    int chiffre_len, clair_len;

    aes_cbc_chiffrer(message, strlen((char*)message),
                     cle, iv, chiffre, &chiffre_len);
    afficher_hex("Chiffré", chiffre, chiffre_len);

    aes_cbc_dechiffrer(chiffre, chiffre_len,
                       cle, iv, clair, &clair_len);
    clair[clair_len] = '\0';
    printf("Déchiffré: %s\n", clair);
}


/* =============================================================================
 * BLOC 14 — RSA AVEC OPENSSL (EVP)
 * ============================================================================= */

/* Générer une paire de clés RSA */
EVP_PKEY *generer_cles_rsa(int taille_bits) {
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, NULL);
    EVP_PKEY_keygen_init(ctx);
    EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, taille_bits);

    EVP_PKEY *cle = NULL;
    EVP_PKEY_keygen(ctx, &cle);
    EVP_PKEY_CTX_free(ctx);
    return cle;
}

/* Chiffrer avec RSA-OAEP */
int rsa_oaep_chiffrer(const unsigned char *clair, size_t clair_len,
                       EVP_PKEY *cle_pub,
                       unsigned char *chiffre, size_t *chiffre_len) {
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(cle_pub, NULL);
    EVP_PKEY_encrypt_init(ctx);
    EVP_PKEY_CTX_set_rsa_padding(ctx, RSA_PKCS1_OAEP_PADDING);
    EVP_PKEY_CTX_set_rsa_oaep_md(ctx, EVP_sha256());

    int ret = EVP_PKEY_encrypt(ctx, chiffre, chiffre_len, clair, clair_len);
    EVP_PKEY_CTX_free(ctx);
    return ret;
}

/* Déchiffrer avec RSA-OAEP */
int rsa_oaep_dechiffrer(const unsigned char *chiffre, size_t chiffre_len,
                         EVP_PKEY *cle_priv,
                         unsigned char *clair, size_t *clair_len) {
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(cle_priv, NULL);
    EVP_PKEY_decrypt_init(ctx);
    EVP_PKEY_CTX_set_rsa_padding(ctx, RSA_PKCS1_OAEP_PADDING);
    EVP_PKEY_CTX_set_rsa_oaep_md(ctx, EVP_sha256());

    int ret = EVP_PKEY_decrypt(ctx, clair, clair_len, chiffre, chiffre_len);
    EVP_PKEY_CTX_free(ctx);
    return ret;
}

/* Signer avec RSA-PSS */
int rsa_pss_signer(const unsigned char *message, size_t msg_len,
                    EVP_PKEY *cle_priv,
                    unsigned char *signature, size_t *sig_len) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_PKEY_CTX *pkey_ctx = NULL;

    EVP_DigestSignInit(ctx, &pkey_ctx, EVP_sha256(), NULL, cle_priv);
    EVP_PKEY_CTX_set_rsa_padding(pkey_ctx, RSA_PKCS1_PSS_PADDING);
    EVP_PKEY_CTX_set_rsa_pss_saltlen(pkey_ctx, RSA_PSS_SALTLEN_DIGEST);

    EVP_DigestSignUpdate(ctx, message, msg_len);
    int ret = EVP_DigestSignFinal(ctx, signature, sig_len);

    EVP_MD_CTX_free(ctx);
    return ret;
}

/* Vérifier une signature RSA-PSS */
int rsa_pss_verifier(const unsigned char *message, size_t msg_len,
                      const unsigned char *signature, size_t sig_len,
                      EVP_PKEY *cle_pub) {
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_PKEY_CTX *pkey_ctx = NULL;

    EVP_DigestVerifyInit(ctx, &pkey_ctx, EVP_sha256(), NULL, cle_pub);
    EVP_PKEY_CTX_set_rsa_padding(pkey_ctx, RSA_PKCS1_PSS_PADDING);
    EVP_PKEY_CTX_set_rsa_pss_saltlen(pkey_ctx, RSA_PSS_SALTLEN_DIGEST);

    EVP_DigestVerifyUpdate(ctx, message, msg_len);
    int ret = EVP_DigestVerifyFinal(ctx, signature, sig_len);

    EVP_MD_CTX_free(ctx);
    return (ret == 1);  /* 1 = signature valide */
}


/* =============================================================================
 * BLOC 15 — MAIN DE RÉFÉRENCE (STRUCTURE TYPE TP)
 * ============================================================================= */

int main(int argc, char *argv[]) {

    printf("=== Exemples de fonctions cryptographiques ===\n\n");

    /* Hachage */
    printf("--- BLOC 1 : Hachage ---\n");
    exemple_hachage();

    /* HMAC */
    printf("\n--- BLOC 2 : HMAC ---\n");
    exemple_hmac();

    /* Aléatoire */
    printf("\n--- BLOC 3 : Aléatoire ---\n");
    exemple_aleatoire();

    /* GMP bases */
    printf("\n--- BLOC 4 : GMP ---\n");
    exemple_gmp_bases();

    /* Exponentiation modulaire */
    printf("\n--- BLOC 5 : Exponentiation modulaire ---\n");
    exemple_exp_modulaire();

    /* Inverse modulaire */
    printf("\n--- BLOC 6 : Inverse modulaire ---\n");
    exemple_inverse();

    /* Équation linéaire */
    printf("\n--- BLOC 7 : Équation linéaire ax≡b(mod n) ---\n");
    exemple_eq_lineaire();

    /* BSGS */
    printf("\n--- BLOC 8 : Baby-Step Giant-Step ---\n");
    exemple_bsgs();

    /* Diffie-Hellman */
    printf("\n--- BLOC 9 : Diffie-Hellman ---\n");
    exemple_dh();

    /* ElGamal */
    printf("\n--- BLOC 10 : ElGamal ---\n");
    exemple_elgamal();

    /* AES-CBC */
    printf("\n--- BLOC 11 : AES-CBC ---\n");
    exemple_aes_cbc();

    return 0;
}


/*
 * =============================================================================
 * PIÈGES CLASSIQUES EN C
 * =============================================================================
 *
 * 1. TOUJOURS mpz_init avant d'utiliser une variable GMP
 *    et mpz_clear à la fin pour éviter les fuites mémoire
 *
 * 2. mpz_invert retourne 0 si l'inverse n'existe pas
 *    (quand pgcd(a,n) ≠ 1) — toujours vérifier le retour
 *
 * 3. mpz_get_ui tronque si la valeur dépasse unsigned long
 *    Pour les grands entiers, utiliser mpz_export ou gmp_printf
 *
 * 4. EVP_CIPHER_CTX_new() et EVP_MD_CTX_new() allouent de la mémoire
 *    Toujours appeler EVP_CIPHER_CTX_free() et EVP_MD_CTX_free() après
 *
 * 5. Le tampon de sortie du chiffrement AES doit être plus grand que l'entrée
 *    (au moins clair_len + 16 pour le padding)
 *
 * 6. CRYPTO_memcmp pour comparer des MACs/signatures
 *    Ne jamais utiliser memcmp (vulnérable aux attaques timing)
 *
 * 7. Dans BSGS : la table de hachage doit être libérée après utilisation
 *    pour éviter les fuites mémoire
 *
 * 8. mpz_powm avec exposant négatif : utiliser mpz_invert séparément
 *    GMP ne supporte pas directement pow(g, -m, p)
 *    → mpz_powm_ui(r, g, m, p); mpz_invert(r, r, p);
 *
 * =============================================================================
 * COMPILATION
 * =============================================================================
 *
 * Avec OpenSSL et GMP :
 *   gcc crypto_reference.c -o crypto_ref -lssl -lcrypto -lgmp -lm
 *
 * Avec seulement GMP (pour les exercices DLP/DH/ElGamal) :
 *   gcc logdiscret.c -o logdiscret -lgmp -lm
 *
 * Avec seulement OpenSSL (pour hachage/AES/RSA) :
 *   gcc signature.c -o signature -lssl -lcrypto
 *
 */


/* =============================================================================
 * BLOC 16 — CBC-MAC MANUEL
 * Standard FIPS PUB 113 — Calcul d'un MAC par chiffrement CBC
 * ============================================================================= */

/*
 * c0 = 0 (IV fixé à zéro)
 * ci = E_K(mi XOR c_{i-1})
 * MAC = cn (dernier bloc chiffré)
 *
 * Le message doit être un multiple de 16 octets (taille bloc AES)
 * Toute modification d'un bloc invalide le MAC final
 */

void cbc_mac(const unsigned char *message, int nb_blocs,
             const unsigned char *cle,
             unsigned char mac_sortie[16]) {

    unsigned char c_prec[16];
    unsigned char bloc_xor[16];
    unsigned char bloc_chiffre[16];
    int chiffre_len;

    /* c0 = 0 */
    memset(c_prec, 0, 16);

    for (int i = 0; i < nb_blocs; i++) {
        /* XOR du bloc courant avec le résultat précédent */
        for (int j = 0; j < 16; j++)
            bloc_xor[j] = message[i * 16 + j] ^ c_prec[j];

        /* Chiffrer avec AES-ECB (un seul bloc) */
        EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
        EVP_EncryptInit_ex(ctx, EVP_aes_128_ecb(), NULL, cle, NULL);
        EVP_CIPHER_CTX_set_padding(ctx, 0);   /* pas de padding — bloc complet */
        EVP_EncryptUpdate(ctx, bloc_chiffre, &chiffre_len, bloc_xor, 16);
        EVP_CIPHER_CTX_free(ctx);

        /* Le résultat devient c_prec pour la prochaine itération */
        memcpy(c_prec, bloc_chiffre, 16);
    }

    /* MAC = dernier bloc chiffré */
    memcpy(mac_sortie, c_prec, 16);
}

/* Vérifier un CBC-MAC */
int verifier_cbc_mac(const unsigned char *message, int nb_blocs,
                      const unsigned char *cle,
                      const unsigned char mac_recu[16]) {
    unsigned char mac_calcule[16];
    cbc_mac(message, nb_blocs, cle, mac_calcule);
    return (CRYPTO_memcmp(mac_calcule, mac_recu, 16) == 0);
}

void exemple_cbc_mac(void) {
    /* Message de 32 octets = 2 blocs AES */
    unsigned char message[32] = "Bonjour, ceci est un message ok!";
    unsigned char cle[16];
    unsigned char mac[16];

    generer_aleatoire(cle, 16);
    cbc_mac(message, 2, cle, mac);
    afficher_hex("CBC-MAC", mac, 16);

    int ok = verifier_cbc_mac(message, 2, cle, mac);
    printf("MAC valide: %s\n", ok ? "OUI" : "NON");

    /* Modifier un octet et vérifier que le MAC change */
    message[0] ^= 0x01;
    ok = verifier_cbc_mac(message, 2, cle, mac);
    printf("MAC après modification: %s\n", ok ? "OUI" : "NON");
}


/* =============================================================================
 * BLOC 17 — POHLIG-HELLMAN (STRUCTURE DE L'ALGORITHME)
 * ============================================================================= */

/*
 * Si l'ordre n du groupe se décompose en petits facteurs premiers
 * n = p1^e1 * p2^e2 * ... * pk^ek
 * On résout x mod pi^ei pour chaque facteur
 * puis on reconstruit x par le Théorème Chinois des Restes (CRT)
 *
 * Complexité : O(sqrt(q_max)) où q_max est le plus grand facteur premier de n
 */

/* Résoudre g^x ≡ b (mod p) dans un sous-groupe d'ordre q_i (petit) */
unsigned long log_discret_petit_groupe(const mpz_t g, const mpz_t b,
                                        const mpz_t p, unsigned long ordre) {
    /* Recherche exhaustive — ordre petit donc faisable */
    mpz_t val, b_mod;
    mpz_init_set_ui(val, 1);
    mpz_init(b_mod);
    mpz_mod(b_mod, b, p);

    for (unsigned long x = 0; x < ordre; x++) {
        if (mpz_cmp(val, b_mod) == 0) {
            mpz_clear(val); mpz_clear(b_mod);
            return x;
        }
        mpz_mul(val, val, g);
        mpz_mod(val, val, p);
    }
    mpz_clear(val); mpz_clear(b_mod);
    return 0;
}

/* Théorème Chinois des Restes : trouver x tel que x ≡ ai (mod mi) */
void crt_deux(mpz_t resultat,
              const mpz_t a1, const mpz_t m1,
              const mpz_t a2, const mpz_t m2) {
    /* x = a1 + m1 * ((a2 - a1) * m1^{-1} mod m2) */
    mpz_t inv, diff, tmp;
    mpz_init(inv); mpz_init(diff); mpz_init(tmp);

    mpz_invert(inv, m1, m2);
    mpz_sub(diff, a2, a1);
    mpz_mod(diff, diff, m2);
    mpz_mul(tmp, diff, inv);
    mpz_mod(tmp, tmp, m2);
    mpz_mul(tmp, tmp, m1);
    mpz_add(resultat, a1, tmp);

    mpz_clear(inv); mpz_clear(diff); mpz_clear(tmp);
}


/* =============================================================================
 * BLOC 18 — ENCODAGE / DÉCODAGE BASE64 EN C
 * ============================================================================= */

/*
 * OpenSSL fournit BIO pour l'encodage base64
 * Utile pour sérialiser des signatures ou des clés
 */

/* Encoder en base64 */
char *encoder_base64_openssl(const unsigned char *donnees, size_t len,
                              size_t *sortie_len) {
    BIO *b64 = BIO_new(BIO_f_base64());
    BIO *mem = BIO_new(BIO_s_mem());
    BIO_push(b64, mem);
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);   /* pas de retour à la ligne */

    BIO_write(b64, donnees, (int)len);
    BIO_flush(b64);

    BUF_MEM *bptr;
    BIO_get_mem_ptr(mem, &bptr);

    char *resultat = malloc(bptr->length + 1);
    memcpy(resultat, bptr->data, bptr->length);
    resultat[bptr->length] = '\0';
    *sortie_len = bptr->length;

    BIO_free_all(b64);
    return resultat;
}

/* Décoder depuis base64 */
unsigned char *decoder_base64_openssl(const char *b64_str, size_t *sortie_len) {
    size_t len = strlen(b64_str);
    unsigned char *resultat = malloc(len);

    BIO *b64 = BIO_new(BIO_f_base64());
    BIO *mem = BIO_new_mem_buf(b64_str, (int)len);
    BIO_push(b64, mem);
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);

    *sortie_len = BIO_read(b64, resultat, (int)len);
    BIO_free_all(b64);
    return resultat;
}

/* Version simple avec EVP */
int base64_encoder_evp(const unsigned char *src, size_t src_len,
                        char *dest, size_t dest_len) {
    return EVP_EncodeBlock((unsigned char *)dest, src, (int)src_len);
}

int base64_decoder_evp(const char *src, unsigned char *dest) {
    return EVP_DecodeBlock(dest, (unsigned char *)src, (int)strlen(src));
}


/* =============================================================================
 * BLOC 19 — GESTION DES ERREURS OPENSSL
 * ============================================================================= */

/* Afficher les erreurs OpenSSL */
void afficher_erreurs_openssl(void) {
    unsigned long err;
    char buf[256];
    while ((err = ERR_get_error()) != 0) {
        ERR_error_string_n(err, buf, sizeof(buf));
        fprintf(stderr, "OpenSSL error: %s\n", buf);
    }
}

/* Vérifier le résultat d'une opération OpenSSL */
void verifier_openssl(int ret, const char *contexte) {
    if (ret <= 0) {
        fprintf(stderr, "Erreur dans %s\n", contexte);
        afficher_erreurs_openssl();
        exit(1);
    }
}


/* =============================================================================
 * BLOC 20 — LECTURE/ÉCRITURE DE CLÉS PEM
 * ============================================================================= */

/* Sauvegarder une clé privée RSA en PEM */
void sauver_cle_privee_pem(EVP_PKEY *cle, const char *chemin) {
    FILE *f = fopen(chemin, "w");
    if (!f) { perror("fopen"); return; }
    PEM_write_PrivateKey(f, cle, NULL, NULL, 0, NULL, NULL);
    fclose(f);
}

/* Sauvegarder une clé publique RSA en PEM */
void sauver_cle_publique_pem(EVP_PKEY *cle, const char *chemin) {
    FILE *f = fopen(chemin, "w");
    if (!f) { perror("fopen"); return; }
    PEM_write_PUBKEY(f, cle);
    fclose(f);
}

/* Charger une clé privée depuis un fichier PEM */
EVP_PKEY *charger_cle_privee_pem(const char *chemin) {
    FILE *f = fopen(chemin, "r");
    if (!f) { perror("fopen"); return NULL; }
    EVP_PKEY *cle = PEM_read_PrivateKey(f, NULL, NULL, NULL);
    fclose(f);
    return cle;
}

/* Charger une clé publique depuis un fichier PEM */
EVP_PKEY *charger_cle_publique_pem(const char *chemin) {
    FILE *f = fopen(chemin, "r");
    if (!f) { perror("fopen"); return NULL; }
    EVP_PKEY *cle = PEM_read_PUBKEY(f, NULL, NULL, NULL);
    fclose(f);
    return cle;
}


/* =============================================================================
 * BLOC 21 — SCÉNARIO COMPLET : ALICE SIGNE ET CHIFFRE POUR BOB
 * ============================================================================= */

/*
 * 1. Alice génère sa paire de clés RSA
 * 2. Bob génère sa paire de clés RSA
 * 3. Alice : signe le message avec sa clé privée (RSA-PSS)
 * 4. Alice : chiffre (message + signature) avec AES-CBC
 * 5. Alice : chiffre la clé de session avec la clé publique de Bob (RSA-OAEP)
 * 6. Bob : déchiffre la clé de session
 * 7. Bob : déchiffre le message
 * 8. Bob : vérifie la signature d'Alice
 */

void scenario_alice_bob(void) {
    printf("\n=== Scénario complet : Alice → Bob ===\n");

    /* Génération des clés */
    EVP_PKEY *cle_alice = generer_cles_rsa(2048);
    EVP_PKEY *cle_bob   = generer_cles_rsa(2048);

    unsigned char message[] = "Message confidentiel d'Alice a Bob";
    size_t msg_len = strlen((char*)message);

    /* --- ALICE : Signer --- */
    unsigned char signature[512];
    size_t sig_len = sizeof(signature);
    rsa_pss_signer(message, msg_len, cle_alice, signature, &sig_len);
    printf("Alice a signé le message (%zu octets)\n", sig_len);

    /* --- ALICE : Chiffrer avec AES-CBC --- */
    unsigned char cle_session[16], iv[16];
    generer_aleatoire(cle_session, 16);
    generer_aleatoire(iv, 16);

    /* Payload = message + separateur + signature */
    unsigned char sep[] = "||SEP||";
    size_t sep_len = strlen((char*)sep);
    size_t payload_len = msg_len + sep_len + sig_len;
    unsigned char *payload = malloc(payload_len);
    memcpy(payload, message, msg_len);
    memcpy(payload + msg_len, sep, sep_len);
    memcpy(payload + msg_len + sep_len, signature, sig_len);

    unsigned char chiffre[4096];
    int chiffre_len;
    aes_cbc_chiffrer(payload, (int)payload_len, cle_session, iv,
                     chiffre, &chiffre_len);
    printf("Message chiffré (%d octets)\n", chiffre_len);

    /* --- ALICE : Chiffrer la clé de session avec la clé publique de Bob --- */
    unsigned char cle_chiffree[512];
    size_t cle_chiffree_len = sizeof(cle_chiffree);
    rsa_oaep_chiffrer(cle_session, 16, cle_bob,
                      cle_chiffree, &cle_chiffree_len);
    printf("Clé de session chiffrée (%zu octets)\n", cle_chiffree_len);

    /* === Transmission : (cle_chiffree, cle_chiffree_len, iv, chiffre, chiffre_len) === */

    /* --- BOB : Déchiffrer la clé de session --- */
    unsigned char cle_session_recue[16];
    size_t cle_session_recue_len = sizeof(cle_session_recue);
    rsa_oaep_dechiffrer(cle_chiffree, cle_chiffree_len, cle_bob,
                        cle_session_recue, &cle_session_recue_len);

    /* --- BOB : Déchiffrer le message --- */
    unsigned char payload_recu[4096];
    int payload_recu_len;
    aes_cbc_dechiffrer(chiffre, chiffre_len, cle_session_recue, iv,
                       payload_recu, &payload_recu_len);

    /* --- BOB : Séparer message et signature --- */
    unsigned char *sep_ptr = (unsigned char*)memmem(payload_recu,
                                                     payload_recu_len,
                                                     sep, sep_len);
    if (!sep_ptr) { printf("Séparateur non trouvé\n"); goto fin; }

    size_t msg_recu_len = sep_ptr - payload_recu;
    unsigned char *sig_recue = sep_ptr + sep_len;
    size_t sig_recue_len = payload_recu_len - msg_recu_len - sep_len;

    /* Afficher le message reçu */
    printf("Message reçu: %.*s\n", (int)msg_recu_len, payload_recu);

    /* --- BOB : Vérifier la signature d'Alice --- */
    int valide = rsa_pss_verifier(payload_recu, msg_recu_len,
                                   sig_recue, sig_recue_len, cle_alice);
    printf("Signature Alice: %s\n", valide ? "VALIDE ✓" : "INVALIDE ✗");

fin:
    free(payload);
    EVP_PKEY_free(cle_alice);
    EVP_PKEY_free(cle_bob);
}


/* =============================================================================
 * BLOC 22 — TABLEAU RÉCAPITULATIF DES FONCTIONS GMP
 * ============================================================================= */

/*
 * INITIALISATION ET AFFECTATION
 * ─────────────────────────────
 * mpz_init(x)                  Initialiser x (obligatoire avant toute utilisation)
 * mpz_init_set_ui(x, val)      Initialiser et affecter un unsigned long
 * mpz_init_set_str(x, "...",10) Initialiser depuis une chaîne en base 10
 * mpz_init_set(x, y)           Initialiser x et copier y dedans
 * mpz_set_ui(x, val)           Affecter un unsigned long
 * mpz_set_str(x, "...", 10)    Affecter depuis une chaîne
 * mpz_set(x, y)                Copier y dans x
 * mpz_clear(x)                 Libérer la mémoire (obligatoire après utilisation)
 *
 * OPÉRATIONS ARITHMÉTIQUES
 * ────────────────────────
 * mpz_add(r, a, b)             r = a + b
 * mpz_add_ui(r, a, v)          r = a + v (unsigned long)
 * mpz_sub(r, a, b)             r = a - b
 * mpz_sub_ui(r, a, v)          r = a - v
 * mpz_mul(r, a, b)             r = a * b
 * mpz_mul_ui(r, a, v)          r = a * v
 * mpz_neg(r, a)                r = -a
 * mpz_abs(r, a)                r = |a|
 * mpz_tdiv_q(q, a, b)          q = a / b (quotient, tronqué vers zéro)
 * mpz_tdiv_r(r, a, b)          r = a mod b (reste)
 * mpz_mod(r, a, m)             r = a mod m (toujours positif)
 * mpz_divexact(r, a, b)        r = a / b (exact, sans reste)
 *
 * OPÉRATIONS MODULAIRES
 * ─────────────────────
 * mpz_powm(r, b, e, m)         r = b^e mod m
 * mpz_powm_ui(r, b, e, m)      r = b^e mod m (e unsigned long)
 * mpz_invert(r, a, m)          r = a^{-1} mod m (retourne 0 si pas d'inverse)
 * mpz_mod(r, a, m)             r = a mod m
 *
 * COMPARAISON
 * ───────────
 * mpz_cmp(a, b)                < 0 si a<b, 0 si a=b, > 0 si a>b
 * mpz_cmp_ui(a, v)             Comparer a avec unsigned long v
 * mpz_sgn(a)                   Signe de a : -1, 0, ou 1
 * mpz_divisible_p(a, b)        1 si b divise a, 0 sinon
 * mpz_divisible_ui_p(a, v)     1 si v divise a, 0 sinon
 *
 * PGCD ET ARITHMÉTIQUE
 * ────────────────────
 * mpz_gcd(g, a, b)             g = pgcd(a, b)
 * mpz_gcdext(g, s, t, a, b)    g = a*s + b*t (Euclide étendu)
 * mpz_lcm(r, a, b)             r = ppcm(a, b)
 *
 * RACINE ET TESTS
 * ───────────────
 * mpz_sqrt(r, a)               r = floor(sqrt(a))
 * mpz_probab_prime_p(p, reps)  Test de primalité probabiliste (retourne 0, 1, ou 2)
 *
 * CONVERSION
 * ──────────
 * mpz_get_ui(a)                Retourne la valeur comme unsigned long (tronqué si grand)
 * mpz_get_si(a)                Retourne la valeur comme signed long
 *
 * AFFICHAGE
 * ─────────
 * gmp_printf("valeur: %Zd\n", a)   Afficher un mpz_t
 * gmp_fprintf(f, "%Zd\n", a)       Écrire dans un fichier
 * mpz_out_str(stdout, 10, a)        Afficher en base 10
 *
 * =============================================================================
 * TABLEAU RÉCAPITULATIF DES FONCTIONS OPENSSL
 * =============================================================================
 *
 * HACHAGE
 * ───────
 * SHA256(msg, len, out)                    SHA-256 direct
 * SHA512(msg, len, out)                    SHA-512 direct
 * SHA1(msg, len, out)                      SHA-1 direct (obsolète)
 * MD5(msg, len, out)                       MD5 direct (cassé)
 * EVP_MD_CTX_new()                         Créer contexte de hachage
 * EVP_DigestInit_ex(ctx, EVP_sha256(), N)  Initialiser avec SHA-256
 * EVP_DigestUpdate(ctx, data, len)         Mettre à jour
 * EVP_DigestFinal_ex(ctx, out, &out_len)   Finaliser
 * EVP_MD_CTX_free(ctx)                     Libérer
 *
 * HMAC
 * ────
 * HMAC(EVP_sha256(), key, klen, msg, mlen, out, &out_len)  HMAC-SHA256 direct
 *
 * ALÉATOIRE
 * ─────────
 * RAND_bytes(buf, len)                     Générer des octets aléatoires sûrs
 *
 * AES-CBC
 * ───────
 * EVP_CIPHER_CTX_new()                     Créer contexte
 * EVP_EncryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, key, iv)  Init chiffrement
 * EVP_EncryptUpdate(ctx, out, &len, in, in_len)               Chiffrer
 * EVP_EncryptFinal_ex(ctx, out+len, &len)                     Finaliser (padding)
 * EVP_DecryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, key, iv)  Init déchiffrement
 * EVP_DecryptUpdate(ctx, out, &len, in, in_len)               Déchiffrer
 * EVP_DecryptFinal_ex(ctx, out+len, &len)                     Finaliser (unpadding)
 * EVP_CIPHER_CTX_free(ctx)                                    Libérer
 *
 * RSA OAEP
 * ────────
 * EVP_PKEY_CTX_new(key, NULL)
 * EVP_PKEY_encrypt_init(ctx)
 * EVP_PKEY_CTX_set_rsa_padding(ctx, RSA_PKCS1_OAEP_PADDING)
 * EVP_PKEY_CTX_set_rsa_oaep_md(ctx, EVP_sha256())
 * EVP_PKEY_encrypt(ctx, out, &out_len, in, in_len)
 *
 * RSA PSS
 * ───────
 * EVP_DigestSignInit(ctx, &pkey_ctx, EVP_sha256(), NULL, key)
 * EVP_PKEY_CTX_set_rsa_padding(pkey_ctx, RSA_PKCS1_PSS_PADDING)
 * EVP_DigestSignUpdate(ctx, msg, len)
 * EVP_DigestSignFinal(ctx, sig, &sig_len)
 * EVP_DigestVerifyInit / EVP_DigestVerifyFinal
 *
 * GÉNÉRATION RSA
 * ──────────────
 * EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, NULL)
 * EVP_PKEY_keygen_init(ctx)
 * EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, 2048)
 * EVP_PKEY_keygen(ctx, &key)
 *
 * COMPARAISON SÉCURISÉE
 * ─────────────────────
 * CRYPTO_memcmp(a, b, len)     Comparaison à temps constant (pour MAC/signatures)
 *                               NE PAS utiliser memcmp pour comparer des MACs
 */


/* =============================================================================
 * BLOC 23 — PIÈGES ET ERREURS CLASSIQUES À L'EXAMEN
 * ============================================================================= */

/*
 * PIÈGE 1 — Oublier mpz_init avant d'utiliser une variable GMP
 *   ✗ mpz_t x;  mpz_set_ui(x, 5);           // comportement indéfini
 *   ✓ mpz_t x;  mpz_init(x);  mpz_set_ui(x, 5);
 *
 * PIÈGE 2 — Oublier mpz_clear → fuite mémoire
 *   Toujours mpz_clear pour chaque mpz_init
 *
 * PIÈGE 3 — mpz_invert retourne 0 si pas d'inverse
 *   Toujours vérifier :
 *   if (!mpz_invert(inv, a, n)) { printf("Pas d'inverse\n"); }
 *
 * PIÈGE 4 — mpz_get_ui tronque les grands entiers
 *   Ne fonctionne correctement que si la valeur tient dans unsigned long
 *   Pour les grands entiers : utiliser gmp_printf("%Zd", x) ou mpz_out_str
 *
 * PIÈGE 5 — Exposant négatif dans GMP
 *   mpz_powm ne supporte pas les exposants négatifs directement
 *   ✗ mpz_powm(r, g, -m, p)
 *   ✓ mpz_powm_ui(gm, g, m, p);  mpz_invert(r, gm, p);  // g^{-m} mod p
 *
 * PIÈGE 6 — Taille du tampon de chiffrement AES
 *   Le chiffré est plus long que le clair (padding PKCS7 ajoute jusqu'à 16 octets)
 *   Prévoir : unsigned char chiffre[clair_len + 16];
 *
 * PIÈGE 7 — Réutiliser un EVP_CIPHER_CTX
 *   Ne jamais réutiliser le même contexte pour chiffrer ET déchiffrer
 *   Toujours créer un nouveau contexte avec EVP_CIPHER_CTX_new()
 *
 * PIÈGE 8 — Comparer des MACs avec memcmp
 *   memcmp s'arrête au premier octet différent → vulnérable aux attaques timing
 *   Toujours utiliser CRYPTO_memcmp(a, b, len)
 *
 * PIÈGE 9 — Dans BSGS, ne pas calculer g^{-m} correctement
 *   ✗ mpz_powm_ui(ginvm, g, -m, p)         // exposant négatif interdit
 *   ✓ mpz_powm_ui(gm, g, m, p);            // d'abord g^m
 *     mpz_invert(ginvm, gm, p);            // puis l'inverse
 *
 * PIÈGE 10 — Table de hachage BSGS : ne pas libérer la mémoire
 *   Toujours libérer les listes chaînées et le tableau après utilisation
 *
 * PIÈGE 11 — Dans CBC-MAC, ne pas mettre le padding à zéro sur l'IV initial
 *   L'IV de CBC-MAC est toujours 0x00...00 (contrairement à AES-CBC standard)
 *
 * PIÈGE 12 — Confondre taille en bits et taille en octets
 *   AES-128 = 128 bits = 16 octets
 *   SHA-256 = 256 bits = 32 octets = SHA256_DIGEST_LENGTH
 *   SHA-512 = 512 bits = 64 octets = SHA512_DIGEST_LENGTH
 */


/* =============================================================================
 * BLOC 24 — STRUCTURE COMPLÈTE D'UN PROGRAMME TP
 * ============================================================================= */

/*
 * Structure type d'un programme de TP sur le logarithme discret
 * (reprend exactement la structure du fichier logdiscret.c du TP)
 */

/*
int main_type_tp(int argc, char *argv[]) {

    // 1. Vérifier les arguments
    if (argc < 4) {
        fprintf(stderr, "Usage: %s p g beta\n", argv[0]);
        return 1;
    }

    // 2. Initialiser les variables GMP
    mpz_t p, g, beta;
    mpz_init(p); mpz_init(g); mpz_init(beta);

    // 3. Lire les paramètres depuis la ligne de commande
    mpz_set_str(p,    argv[1], 10);
    mpz_set_str(g,    argv[2], 10);
    mpz_set_str(beta, argv[3], 10);

    // 4. Afficher les paramètres
    gmp_printf("p    = %Zd\n", p);
    gmp_printf("g    = %Zd\n", g);
    gmp_printf("beta = %Zd\n", beta);

    // 5. Calculer la racine carrée de p pour déterminer m
    mpz_t temp;
    mpz_init(temp);
    mpz_sqrt(temp, p);
    unsigned long m = mpz_get_ui(temp) + 1;
    printf("m = %lu\n", m);
    mpz_clear(temp);

    // 6. Allouer la table de hachage
    unsigned int table_size = 2 * m + 1;
    HashTable ht = calloc(table_size, sizeof(Maillon *));

    // 7. Phase 1 — Baby Steps
    mpz_t aux;
    mpz_init_set_ui(aux, 1);
    for (unsigned long j = 1; j < m; j++) {
        mpz_mul(aux, aux, g);
        mpz_mod(aux, aux, p);
        hash_inserer(&ht, mpz_get_ui(aux), j, table_size);
    }

    // 8. Phase 2 — Giant Steps
    mpz_t ginvm, z;
    mpz_init(ginvm); mpz_init_set(z, beta);

    // Calculer g^{-m} mod p
    mpz_powm_ui(aux, g, m, p);      // aux = g^m mod p
    mpz_invert(ginvm, aux, p);      // ginvm = g^{-m} mod p

    unsigned long logd = 0;
    unsigned long j_match;
    for (unsigned long i = 0; i <= m; i++) {
        if (hash_chercher(&ht, mpz_get_ui(z), &j_match, table_size)) {
            logd = i * m + j_match;
            break;
        }
        mpz_mul(z, z, ginvm);
        mpz_mod(z, z, p);
    }

    printf("Log discret = %lu\n", logd);

    // 9. Vérification
    mpz_t verif;
    mpz_init(verif);
    mpz_powm_ui(verif, g, logd, p);
    int ok = (mpz_cmp(verif, beta) == 0);
    printf("Vérification: %s\n", ok ? "OK" : "ERREUR");

    // 10. Libérer la mémoire
    hash_liberer(ht, table_size);
    mpz_clear(p); mpz_clear(g); mpz_clear(beta);
    mpz_clear(aux); mpz_clear(ginvm); mpz_clear(z); mpz_clear(verif);

    return 0;
}
*/