#include <stdio.h>
#include <time.h>
#include <gmp.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  int taille; //taille de l'entier n
  gmp_randstate_t state; // variable nécessaire pour le générateur aléatoire
  mpz_t p,q,n,phin,p1,q1, e,d,gcd;
  int test_p_premier, test_q_premier;

  mpz_init(p);
  mpz_init(q);
  mpz_init(p1);
  mpz_init(q1);
  mpz_init(n);
  mpz_init(phin);

  //PARTIE 2
  mpz_init(e);
  mpz_init(d);
  mpz_init(gcd);

 // récupération de la valeur de taille sur la ligne de commande
    taille = atoi(argv[1]);
 // initialisation du générateur aléatoire 
    gmp_randinit_default(state);
    gmp_randseed_ui(state,time(NULL));

/*A COMPLETER, la variable state est à utiliser dans toute fonction générant de l'aléatoire.*/

 // générer aléatoirement deux entiers premiers p et q de taille EXACTEMENT taille/2
    mpz_urandomb(p, state, taille/2);
    mpz_setbit(p, taille/2-1);

    mpz_urandomb(q, state, taille/2);
    mpz_setbit(q, taille/2-1);

 // Test si premier, returns 1 si quasi sur premier sinon 0 si pas premier
    test_p_premier = mpz_probab_prime_p(p, 25);
    test_q_premier = mpz_probab_prime_p(q, 25);    

    while(test_p_premier == 0){
        mpz_urandomb(p, state, taille/2 -1);
        mpz_setbit(p, taille/2-1);
        test_p_premier = mpz_probab_prime_p(p, 25);
    }   

    while(q==p){
        mpz_urandomb(q, state, taille/2);
        mpz_setbit(q, taille/2-1);
        while(test_q_premier == 0){
            mpz_urandomb(q, state, taille/2 -1);
            mpz_setbit(q, taille/2-1);
            test_q_premier = mpz_probab_prime_p(q, 25);
        }
    }
    

    mpz_mul(n, p, q);

    mpz_sub_ui(p1,p,1);
    mpz_sub_ui(q1,q,1);

    mpz_mul(phin,p1,q1); 
    
    gmp_printf("p : %Zd\n", p);
    gmp_printf("q : %Zd\n", q);
    gmp_printf("n : %Zd\n", n);
    gmp_printf("phi(n) : %Zd\n", phin);

// engendrer un entier e aléatoire plus petit que φ(n) et premier avec φ(n)
    mpz_urandomm(e, state, phin);
    mpz_gcd(gcd,e,phin);
    while(mpz_cmp_ui(gcd,1)>0){
        mpz_urandomm(e, state, phin);
        mpz_gcd(gcd,e,phin);
    }

    //Déduisez-en l'entier d tel que ed≡1modφ(n)
    mpz_invert(d,e,phin);

    gmp_printf("e : %Zd\n", e);
    gmp_printf("d : %Zd\n", d);    
    
    // PARTIE 3

    mpz_t m,c,temp;
    mpz_init(m);
    mpz_init(c);
    mpz_init(temp);


    mpz_urandomm(m, state, n);
    mpz_powm(c,m,e,n);
    gmp_printf("c : %Zd\n", c);    
    mpz_powm(temp,c,d,n);
    gmp_printf("Inverse : %Zd\n", temp);

    //gmp_printf("Comparaison : %Zd\n", mpz_cmp(c,temp));
    gmp_printf("m : %Zd\n", m);  


    // CLEAR

    mpz_clear(p);
    mpz_clear(q);
    mpz_clear(n);
    mpz_clear(p1);
    mpz_clear(q1);
    mpz_clear(phin);

    mpz_clear(e);
    mpz_clear(d);

    mpz_clear(m);
    mpz_clear(c);
    mpz_clear(temp);
}
