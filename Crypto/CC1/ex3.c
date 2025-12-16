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

    mpz_set_ui(e, 3);

    mpz_mul(n, p, q);

    mpz_t p1, q1;
    mpz_inits(p1, q1, NULL);
    mpz_sub_ui(p1, p, 1);
    mpz_sub_ui(q1, q, 1);
    mpz_mul(phi, p1, q1);

    mpz_invert(d, e, phi);
    gmp_printf("d = %Zx\n", d);

    // Exemple déchiffrement :
    // mpz_set_str(C, "CRYPTOTEX...", 16);
    // mpz_powm(M, C, d, n);
    // gmp_printf("Message M = %Zx\n", M);

    mpz_clears(p,q,n,phi,e,d,C,M,p1,q1,NULL);
    return 0;
}
