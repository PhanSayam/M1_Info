#include <stdio.h>
#include <time.h>
#include <gmp.h>
#include <stdlib.h>

int solovay_strassen(int t, mpz_t n, gmp_randstate_t state){
   mpz_t a,n_minus, temp, nminus1div2;
   mpz_t jacobi_modn;

   mpz_init(a);
   mpz_init(n_minus);
   mpz_init(temp);
   mpz_init(nminus1div2);
   mpz_init(jacobi_modn);

   mpz_sub_ui(n_minus,n,1);
   mpz_fdiv_q_ui(nminus1div2, n_minus,2);

   for(int i=1;i<t+1;i++){
      // tire a aléatoirement
      //changer
      mpz_urandomm(a, state, n_minus); 
      // a <- a+2
      mpz_add_ui(a, a, 2); 

      // faire le pgcd ici 
      // jacobi de (a/n)
      int jacobi = mpz_jacobi(a, n);

      if(jacobi == 0){ // Pas besoin de ce cas a ce moment la 
         mpz_clears(a,n_minus,temp,nminus1div2,jacobi_modn);
         return 0;
      }else if(jacobi == -1){
         mpz_set(jacobi_modn, n_minus);
      }
      else {
         mpz_set_ui(jacobi_modn, 1);
      }

      mpz_powm(temp, a, nminus1div2, n);
      if(mpz_cmp(temp, jacobi_modn)!=0){
         mpz_clears(a,n_minus,temp,nminus1div2,jacobi_modn);
         return 0;
      }
   }
   mpz_clear(a);
   mpz_clear(n_minus);
   mpz_clear(temp);
   mpz_clear(nminus1div2);
   mpz_clear(jacobi_modn);
   
   return 1;
}

int main(int argc, char *argv[])
{
  int t; //parametre pour la probabilite
  mpz_t n; // entier a tester
  gmp_randstate_t state; // variable nécessaire pour le générateur aléatoire

 // on teste si le nombre de parametres d'appels est correct
    if (argc < 3)
    {
        printf("Usage : ./solovay_strassen t n\n");
        exit(-1);
    }
 // récupération de la valeur de t sur la ligne de commande
    t = atoi(argv[1]);
 // récupération de la valeur de n sur la ligne de commande  
    mpz_init(n);
    mpz_set_str(n,argv[2],16); 
 // initialisation du générateur aléatoire 
    gmp_randinit_default(state);
    gmp_randseed_ui(state,time(NULL));

    gmp_printf("n : %Zd\n",n);

/*A COMPLETER, la variable state est à utiliser dans toute fonction générant de l'aléatoire.*/
   printf("%d\n",solovay_strassen(t, n, state));
}