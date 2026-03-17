#include <stdio.h>
#include <gmp.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include "hashutils.h"


gmp_randstate_t etat;


void naivesearch(mpz_t beta, mpz_t g, mpz_t p)
{
    /* Recherche de façon exhaustive l'entier expo tel que */
    /* beta = g^expo modulo p */
    /* affiche la valeur de expo */

    mpz_t b, p_1, x;
    mpz_init(b);
    mpz_init(p_1);
    mpz_init(x);

    mpz_sub_ui(p_1, p, 1);
    mpz_set_ui(x,0);
    
    while(mpz_cmp(x,p_1) < 0){
        mpz_powm(b,g,x,p);
        if (mpz_cmp(b,beta) == 0){
            gmp_printf("Valeur de expo : %Zd\n", x);
            break;
        }
        mpz_add_ui(x,x,1);
    }

    // mpz_clear(b);
    // mpz_clear(p_1);
    // mpz_clear(x);
 
}


int main(int argc, char *argv[]) {

    int logok;
    mpz_t p,g,beta,aux,ginvm,k;
    unsigned long logd,m, i, j_match, table_size;
    HashTable ht;



    // A COMPLETER initialisation de p g et beta
    // en fonction des parametres de la ligne de commande

    
    mpz_init(p);
    mpz_init(g);
    mpz_init(beta);

    mpz_set_str(p,argv[1],10);
    mpz_set_str(g,argv[2],10);
    mpz_set_str(beta,argv[3],10);

    mpz_init(aux);
    mpz_init(ginvm);
    mpz_init(k);

    gmp_printf("p = %Zd\n",p);
    gmp_printf("gen = %Zd\n",g);
    gmp_printf("beta = %Zd\n",beta);

    // PARTIE 1 : recherche naïve
    // naivesearch(beta,g,p);



    // PARTIE 2 : ici la partie recherche de l'exposant utilisant l'algo 
    // vu en cours, commentez la PARTIE1

    // Calcul de la partie entière de la racine carré de p
    // a completer
    mpz_t temp;
    mpz_init(temp);

    mpz_sqrt(temp, p);
    m = mpz_get_ui(temp);
    printf("racine de p = %lu\n", m);
    
    // Allocation de la table de hachage
    // table un peu plus grande pour limiter les collisions 
    table_size = 2 * m + 1;
    ht = calloc(table_size, sizeof(Maillon *));
    if (ht == NULL) {
        fprintf(stderr, "Erreur: allocation de la table de hachage.\n");
        exit(1);
    }


    // Calcul des g^j et stockage dans la table de hachage 
    printf("Init Phase 1\n");
    // variable aux utilisée pour calculer g^j mod p
    mpz_set_ui(aux,1);
    for (i = 1; i < m ; i++)
    {
        // calcul de g^i % p
        // insertion dans la table de hachage
        mpz_mul(aux, aux, g);
        mpz_mod(aux, aux, p);
        hash_insert(ht, aux, i, table_size);

    }
    printf("Phase 1 OK\n");

    printf("Init Phase 2\n");
    mpz_set_ui(aux,m);
    // calcul de g^m mod p 
    mpz_powm(ginvm,g,aux,p);
    // calcul de g^(-m) mod p 
    mpz_invert(ginvm,ginvm,p);

    // Rercherche des g^(-mi) dans la table de hachage
    // jusqu'à trouver un g^j tel que g^(-mi)=g^j
    // le log discret recherché vaut alors i*m+j
    while (i < m) {
        unsigned long j;
        if (hash_find(ht, current, &j, table_size)) {  // CORRECTION : ht, &j
            logd = i * m + j;
            break;
        }
        mpz_mul(current, current, ginvm);
        mpz_mod(current, current, p);
        i++;
    }

    printf("Log discret : %lu\n",logd);
    mpz_clear(aux);
    mpz_clear(p);
    mpz_clear(g);
    mpz_clear(k);
    mpz_clear(beta);
    mpz_clear(ginvm);
}
