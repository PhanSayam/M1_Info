typedef struct Maillon {
    unsigned long gj;       // valeur g^j mod p pour p un entier d'au plus 54 bits
    unsigned long j;         // exposant j
    struct Maillon *next;
} Maillon;

#include <math.h>
#include <stdlib.h>

typedef Maillon **HashTable;
// une table de hachage associe à une entree k,
// une liste de Maillons, i.e si h est de type
// Hashtable, alors h[k] est une liste de maillons
// h est donc une liste de liste de maillons

/* ---------- Outils hash ---------- */

static unsigned int hash(unsigned long z, unsigned int table_size) {

    unsigned long  h ; 
    
    // calcule le hash de z en faisant un xor entre les 32 bits de poids fort de z
    // et les 32 bits de poids faible de z
    // retourne la valeur obtenue modulo table_size

    unsigned long z1;
    unsigned long z2;
    unsigned long puissance;

    puissance = pow(2,32)-1;

    z1 = z >> 32;
    z2 = z & puissance;

    h = (z1 ^ z2) % table_size;

    return h;
}


static int hash_insert(HashTable *ht, unsigned long gj, unsigned long j, unsigned int size) {
    unsigned int idx; 

    // ici ht est un pointeur sur une table de hachage
    // donc *ht est une table de hachage
    // donc (*ht)[idx] est la liste des maillons (eventuellement vide) que l'on trouve à l'entree idx de cette table.

    // cette fonction insère dans la table de hachage le maillon constitue de gj et de j
    // ce maillon est inséré dans la liste située en position idx dans la table
    // où idx est le hashé de gj
 
    Maillon *e ;
    
    // on calcule idx à partir de gj
    idx = hash(gj,size);

   // On créé un nouveau maillon e
   // et on l'insère en position idx    
   // dans la table
 
    //A COMPLETER
    e = calloc(1,sizeof(Maillon));
    e-> gj = gj;
    e-> j  = j;
    e-> next = (*ht)[idx];

    (*ht)[idx] = e;   
 
    return 1;
}

static int hash_find(HashTable *ht, const unsigned long val, unsigned long *j_out, unsigned int size) {
 
    // recherche si dans la table de hachage pointée par ht 
    // à l'entrée correspondant à la valeur val on trouve
    // un maillon dont le membre gj vaut val
    // Si c'est le cas la fonction renvoie 1 et remplit *j_out
    // avec  la valeur j du maillon sinon la fonction renvoie 0   
        
    unsigned int idx;

    Maillon *cur;
    
    idx = hash(val, size);

    // A COMPLETER
    int onContinue = 0;

    cur = calloc(1,sizeof(Maillon));
    cur = (*ht)[idx];
    
    while ((cur != NULL) && !(onContinue)){  
        if (cur->gj == val){
            onContinue = 1;
            *j_out = (cur->j);
        }
        cur = cur->next;
    }    
    return onContinue;
}
