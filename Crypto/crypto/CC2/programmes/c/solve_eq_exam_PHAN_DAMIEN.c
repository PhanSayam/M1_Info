#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

// Algorithme de Tonelli-Shanks pour trouver r tel que r^2 ≡ a (mod p)
// pour p = 1 mod 4
void tonelli_shanks(mpz_t r, mpz_t a, mpz_t p) {
    mpz_t q, s, z, m, c, t, temp, b, exp;
    mpz_init(q); mpz_init(s); mpz_init(z); mpz_init(m);
    mpz_init(c); mpz_init(t); mpz_init(temp); mpz_init(b); mpz_init(exp);
    
    // Trouve Q et S tels que p-1 = Q * 2^S avec Q impair
    mpz_sub_ui(q, p, 1);
    mpz_set_ui(s, 0);
    while (mpz_even_p(q)) {
        mpz_fdiv_q_ui(q, q, 2);
        mpz_add_ui(s, s, 1);
    }
    
    // Trouve un non-résidu quadratique z
    // exp = (p-1)/2
    mpz_sub_ui(exp, p, 1);
    mpz_fdiv_q_ui(exp, exp, 2);

    mpz_set_ui(z, 2);
    mpz_powm(temp, z, exp, p);
    while (mpz_cmp_ui(temp, 1) == 0) {
        mpz_add_ui(z, z, 1);
        mpz_powm(temp, z, exp, p);
    }
    
    mpz_set(m, s);
    mpz_powm(c, z, q, p);  // c = z^Q mod p
    mpz_powm(t, a, q, p);  // t = a^Q mod p
    
    // exp = (Q+1)/2
    mpz_add_ui(exp, q, 1);
    mpz_fdiv_q_ui(exp, exp, 2);
    mpz_powm(r, a, exp, p);  // r = a^((Q+1)/2) mod p
    
    while (mpz_cmp_ui(t, 1) != 0) {
        // Trouve i tel que t^(2^i) ≡ 1 (mod p)
        mpz_set(temp, t);
        unsigned long i = 1;
        mpz_powm_ui(temp, temp, 2, p);
        while (mpz_cmp_ui(temp, 1) != 0) {
            mpz_powm_ui(temp, temp, 2, p);
            i++;
        }
        
        // b = c^(2^(m-i-1)) mod p
        mpz_sub_ui(temp, m, i);
        mpz_sub_ui(temp, temp, 1);
        mpz_ui_pow_ui(exp, 2, mpz_get_ui(temp));
        mpz_powm(b, c, exp, p);
        
        mpz_mul(r, r, b);
        mpz_mod(r, r, p);
        
        mpz_powm_ui(c, b, 2, p);
        
        mpz_mul(t, t, c);
        mpz_mod(t, t, p);
        
        mpz_set_ui(m, i);
    }
    
    mpz_clear(q); mpz_clear(s); mpz_clear(z); mpz_clear(m);
    mpz_clear(c); mpz_clear(t); mpz_clear(temp); mpz_clear(b); mpz_clear(exp);
}

/**
 * Calcule une racine carrée de a modulo p (p premier)
 * 
 * @param r: sortie - racine carrée (r^2 ≡ a mod p)
 * @param a: entrée - nombre dont on cherche la racine
 * @param p: entrée - module premier
 */

void square_root(mpz_t r, mpz_t a, mpz_t p) {
    mpz_t a_mod, p_mod_4, exp;
    mpz_init(a_mod);
    mpz_init(p_mod_4);
    mpz_init(exp);
    
    // Réduit a modulo p
    mpz_mod(a_mod, a, p);
        
    // Vérifie si p ≡ 3 (mod 4)
    mpz_mod_ui(p_mod_4, p, 4);
    
    if (mpz_cmp_ui(p_mod_4, 3) == 0) {
        // Cas simple: p ≡ 3 (mod 4)
        // r = a^((p+1)/4) mod p
        mpz_add_ui(exp, p, 1);
        mpz_fdiv_q_ui(exp, exp, 4);
        mpz_powm(r, a_mod, exp, p);
    } else {
        // Cas général: utilise l'algorithme de Tonelli-Shanks
        tonelli_shanks(r, a_mod, p);
    }
    
    mpz_clear(a_mod);
    mpz_clear(p_mod_4);
    mpz_clear(exp);
}

void generate_example(mpz_t a, mpz_t p, mpz_t q, mpz_t n, gmp_randstate_t state)
{
    mpz_t aux1, aux2, exp1, exp2;

    mpz_init(aux1);
    mpz_init(aux2);
    mpz_init(exp1);
    mpz_init(exp2);

    // Generation aleatoire de 2 entiers premiers 
    // p et q de 512 bits
    mpz_rrandomb(p,state,512);
    mpz_nextprime(p,p);
    mpz_nextprime(q,p);
    // Calcul de n = p*q
    mpz_mul(n,p,q);
    // recherche de a tant que a n'est pas un QR mod p et mod q
    mpz_urandomm(a,state,n);
    // exp1 = (p-1)/2
    mpz_sub_ui(exp1, p, 1);
    mpz_fdiv_q_ui(exp1, exp1, 2);    
    // aux1 = a^exp mod p
    mpz_powm(aux1, a, exp1, p);
    // exp2 = (q-1)/2
    mpz_sub_ui(exp2, q, 1);
    mpz_fdiv_q_ui(exp2, exp2, 2);    
    // aux2 = a^exp mod p
    mpz_powm(aux2, a, exp2, q);
    while ((mpz_cmp_ui(aux1,1)!=0) || (mpz_cmp_ui(aux2,1)!=0))
    {
        mpz_urandomm(a,state,n);
        mpz_powm(aux1, a, exp1, p);
        mpz_powm(aux2, a, exp2, q);
    }
    mpz_clear(aux1); mpz_clear(aux2); mpz_clear(exp1); mpz_clear(exp2);

}

int main() {
    mpz_t a, p, q, n;
    mpz_t temp, r1, r2, r3, r4;
    gmp_randstate_t state;
    int i, score;

    // RAJOUTER ICI LES VARIABLES SUPPLEMENTAIRES DONT VOUS AUREZ EVENTUELLEMENT BESOIN
    mpz_t x1,x2,z1,z2;
    mpz_t eea, inverse_p, inverse_q;
    mpz_t y1, y2, val1, val2;

    mpz_init(a); mpz_init(p); mpz_init(q); mpz_init(n);
    mpz_init(temp); mpz_init(r1); mpz_init(r2); mpz_init(r3); mpz_init(r4);

    // RAJOUTER ICI LES INITIALISATIONS NECESSAIRES SI BESOIN
    mpz_init(x1); mpz_init(x2); mpz_init(z1); mpz_init(z2); 
    mpz_init(eea); mpz_init(inverse_p); mpz_init(inverse_q); 
    mpz_init(y1);mpz_init(y2); mpz_init(val1);mpz_init(val2);

    // Initialisation du generateur aleatoire
    gmp_randinit_default(state);
    
    score = 0;
    for (i = 1; i <= 10; i++)
    {
        // La fonction ci-dessous génère un jeu de paramètres a, p, q, et n. p et q sont deux entiers
        // premiers et a est toujours un résidu quadratique modulo n
        generate_example(a,p,q,n,state);   
        gmp_printf("p = %Zd\n", p);
        gmp_printf("q = %Zd\n", q);
        gmp_printf("n = p*q = %Zd\n", n);
        gmp_printf("a = %Zd\n\n", a);
       
        // COMPLETER ICI LE CODE 
        // LES RACINES CARREES DE A DEVRONT ETRE STOCKEES DANS
        // LES VARIABLES R1, R2, R3 et R4
        // FAITES ATTENTION A NE PAS MODIFIER DANS VOTRE CODE LES VARIABLES A ET N
        /************************************************************************************/
        square_root(x1,a,p);
        square_root(z1,a,q);
    
        //mpz_neg(x2, x1);
        //mpz_neg(z2, z1);
        mpz_sub(temp, p, x1);
        mpz_mod(x2, temp, p);
    
        mpz_sub(temp, q, z1);
        mpz_mod(z2, temp, q);

        // void mpz_gcdext (mpz_t g, mpz_t s, mpz_t t, const mpz_t a, const mpz_t b)
        mpz_gcdext(eea, inverse_p, inverse_q, p, q);

        // Function: void mpz_mul (mpz_t rop, const mpz_t op1, const mpz_t op2)
        // void mpz_add (mpz_t rop, const mpz_t op1, const mpz_t op2)

        mpz_mod(temp, inverse_q, p);
        mpz_mul(y1, temp, q);
        mpz_mod(y1, y1, n);

        mpz_mod(temp, inverse_p, q);
        mpz_mul(y2, temp, p);
        mpz_mod(y2, y2, n);

        mpz_mul(val1,x1,y1);
        mpz_mul(val2,z1,y2);
        mpz_add(r1,val1, val2);
        mpz_mod(r1, r1, n);
        mpz_neg(r2, r1);
        mpz_mod(r2, r2, n);
        
        mpz_mul(val1,x1,y1);
        mpz_mul(val2,z2,y2);
        mpz_add(r3,val1, val2);
        mpz_mod(r3, r3, n);
        mpz_neg(r4, r3);
        mpz_mod(r4, r4, n);
        /***********************************************************************************/
        // FIN DE LA PARTIE A COMPLETER

        // PHASE DE VERIFICATION
        // NE RIEN TOUCHER CI-DESSOUS
        mpz_powm_ui(temp, r1, 2, n);
        if (mpz_cmp(temp,a)==0)
            score++;
        mpz_powm_ui(temp, r2, 2, n);
        if (mpz_cmp(temp,a)==0)
            score++;
        mpz_powm_ui(temp, r3, 2, n);
        if (mpz_cmp(temp,a)==0)
            score++;
        mpz_powm_ui(temp, r4, 2, n);
        if (mpz_cmp(temp,a)==0)
            score++;
    }
    printf("Taux de réussite : %f%%\n",100*(((float)score)/40));
    mpz_clear(a); mpz_clear(p); mpz_clear(q); mpz_clear(n);
    mpz_clear(temp); mpz_clear(r1); mpz_clear(r2); mpz_clear(r3); mpz_clear(r4);
    return 0;
}