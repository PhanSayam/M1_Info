#include <stdio.h>
#include <time.h>
#include <gmp.h>
#include <stdlib.h>

// gcc miller_rabin.c -o miller_rabin -lgmp

int main (int argc, char *argv[]){

    int k;                        //parametre pour la probabilite
    mpz_t n;                      // entier a tester
    mpz_t nmu;                    // pour le calcul de n-1
    mpz_t n_3;                    // n-3
    mpz_t t;                      // pour la decomposition n-1 = 2^s*t
    gmp_randstate_t state;        // variable nécessaire pour le générateur aléatoire

    mpz_t a;
    mpz_t at;
    mpz_t gcd;
    unsigned long decalage;
    unsigned long s;
    unsigned long valeur_2;
    int premier;

    // on teste si le nombre de parametres d'appels est correct
    if (argc < 3)
    {
        printf ("Usage : ./miller_rabin k n\n");
        exit (-1);
    }

    // initialisation du générateur aléatoire 
    gmp_randinit_default (state);
    gmp_randseed_ui (state, time (NULL));

    // récupération de la valeur de t sur la ligne de commande
    k = atoi (argv[1]);

    // récupération de la valeur de n sur la ligne de commande  
    mpz_init (n);
    mpz_set_str (n, argv[2], 16);

    // calcul de n-1
    mpz_init (nmu);
    mpz_sub_ui (nmu, n, 1);

    mpz_init(t);
    s = 0;
    decalage = 1;  //sera utilisé pour mpz_mul_2exp, pour trouver s et t tel que n-1=2^s*t

    /*A COMPLETER, la variable state est à utiliser dans toute fonction générant de l'aléatoire.*/
    // void mpz_gcd (mpz_t rop, const mpz_t op1, const mpz_t op2)

    // t = n-1
    mpz_set(t,nmu);

    // mpz_mul_2exp (mpz_t rop, const mpz_t op1, mp_bitcnt_t op2) qui réalise l'opération rop ← (op1 << op2), le type mp_bitcnt_t correspond au type unsigned long.
    while(mpz_tstbit(t,0)==1){
        s++;
        mpz_mul_2exp(t, t, decalage);
    }

    mpz_init(a);
    mpz_init(n_3);
    valeur_2 = 2;

    mpz_sub_ui(n_3,n,3);
    mpz_urandomm(a, state, n_3);
    mpz_add_ui(a, a, valeur_2);

    mpz_init(at);
    mpz_init(gcd);

    for(int i=0;i<k;i++){
        mpz_gcd(gcd,a,n);
        while(mpz_cmp_ui(gcd,(signed long int) 1) != 0){
            mpz_urandomm(a, state, n_3);
            mpz_add_ui(a, a, valeur_2);
        }
        mpz_powm(at, a, t, n);

        if(mpz_cmp_ui(at, (unsigned long int) 1)!=0 && mpz_cmp(at, nmu)!=0){
            premier = 0;
            for (int i=0; i < s ; i++){
                mpz_powm_ui(at,at,valeur_2,n);
                if (mpz_cmp(at, nmu)==0){
                    premier = 1;
                }
            }            
            if (!premier) {
                mpz_clear(n);
                mpz_clear(nmu);
                mpz_clear(n_3);
                mpz_clear(t);
                mpz_clear(a);
                mpz_clear(at);
                mpz_clear(gcd);
                printf("\n0\n");
                return 0;
            }
        }
        mpz_clear(n);
        mpz_clear(nmu);
        mpz_clear(n_3);
        mpz_clear(t);
        mpz_clear(a);
        mpz_clear(at);
        mpz_clear(gcd);

        printf("\n1\n");
        return 1;
        
    }
}
