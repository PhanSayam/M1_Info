#include <stdio.h>
#include <gmp.h>

int main() {
    mpz_t n, p, q, r;
    mpz_inits(n, p, q, r, NULL);

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

    mpz_sqrt(p, n);

    while (1) {
        mpz_mod(r, n, p);
        if (mpz_cmp_ui(r, 0) == 0) break;
        mpz_add_ui(p, p, 1);
    }

    mpz_div(q, n, p);

    printf("p = "); mpz_out_str(stdout, 16, p);
    printf("\nq = "); mpz_out_str(stdout, 16, q);
    printf("\n");

    mpz_clears(n, p, q, r, NULL);
    return 0;
}
