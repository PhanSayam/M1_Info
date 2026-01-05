#include <stdio.h>
#include <time.h>
#include <gmp.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{

    mpz_t e,p,q,n,phin,p1,q1, d;

    
    mpz_init(e);
    mpz_init(p);
    mpz_init(q);
    mpz_init(p1);
    mpz_init(q1);
    mpz_init(n);
    mpz_init(phin);

    mpz_init(d);
    
    mpz_set_str(e, "0003", 10);
    mpz_set_str(p, "4aa55829181056994b47e8c26e3ed27780892a2679901510ab2769bcec3ea77f098a03d28be3c7834978d92ba57f74f19aff", 16);
    mpz_set_str(q, "f4197a54665c00d21df5ca59a6d8c1632b2c781e29284573d10dfcd0d06c251f858fcf5b86914a9858157a727c2e62e2fdadb", 16);

    gmp_printf("p : %Zd\n", p);
    gmp_printf("q : %Zd\n", q);
    gmp_printf("e : %Zd\n", e);

    mpz_mul(n,p,q);
    
    gmp_printf("n: %Zd\n",n);

    mpz_sub_ui(p1,p,1);
    mpz_sub_ui(q1,q,1);

    mpz_mul(phin,p1,q1);    

    gmp_printf("phi(n): %Zd\n",phin);

    mpz_invert(d,e,phin);

    gmp_printf("Clée secrète d : %Zd\n", d);  


    mpz_t message_chiffre, m;
    //String temp;

    mpz_init(message_chiffre);
    mpz_init(m);


    mpz_set_str(message_chiffre, "16b92f99d4cfcd5513e5cf5d0a1d5803a43bea28c0", 16);

    mpz_powm(m,message_chiffre,d,n);
    gmp_printf("Inverse : %Zx\n", m);

    /*
    for(int i=0;i<m.length();i++){
        temp = m[0+i:i];

    }d
        */

        //r2.html
        //http://veron.univ-tln.fr/M2/TP_RSA/r2.html

    mpz_clear(p);
    mpz_clear(q);
    mpz_clear(n);
    mpz_clear(p1);
    mpz_clear(q1);
    mpz_clear(phin);
    mpz_clear(d);


}