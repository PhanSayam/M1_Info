#include <stdio.h>
#include <stdint.h>
#include "hashutils.h"

void affiche(HashTable *ht, int idx)
{
    // Affiche la liste chainee située en position idx

    Maillon *aux;

    aux = (*ht)[idx];
    while (aux != NULL){
        printf("%lx:%lu ",aux->gj,aux->j);
        aux = aux->next;
    }
    printf("\n");
}

int main(void)
{
  uint64_t test1,test2, j;
  uint32_t res;
  HashTable dico;


  test1 = 0xffffffffffffffff;
  test2 = 0x0aaaaaaa0a1a1a1a;

  res = hash(test1,(1 << 31));
  if (res == 0)
    printf("Test 1 hash ok\n");
  else
  {
    printf("Tes 1 hash failed\n");
    return 0;
  }
  res = hash(test2,(1 << 31));
  if (res == 11579568)
    printf("Test 2 hash ok\n");
  else
  {
    printf("Tes 2 hash failed\n");
    return 0;
  }
  // A decommenter pour tester la fonction hash_insert
  // on cree un dictionnaire à 10 entrees
  dico = calloc(10,sizeof(Maillon*));
  // on remplit arbitrairement la table de hachage*/
  test1 = 0x0000001000000013;
  hash_insert(&dico,test1,100,10);
  test1 = 0x00000a0000000a03;
  hash_insert(&dico,test1,120,10);
  test1 = 0xabcdef13abcdef10;
  hash_insert(&dico,test1,130,10);
  // le hash des entiers 0x0000001000000013
  // 0x00000a0000000a03 et 0xabcdef13abcdef10
  // vaut 3, on a donc en fait remplit l'entrée 3
  // de la table de hachage
  // la fonction affiche doit afficher 
  // abcdef13abcdef10:130 a0000000a03:120 1000000013:100
  affiche(&dico,3);



  // A decommenter pour tester la fonction hash_find
  test1 = 0x00000a0000000a03;
  if (hash_find(&dico,test1,&j,10)==1)
   printf("%lx est associée à la valeur j = %ld\n",test1,j);
  else 
   printf("%lx n'est pas dans la table\n",test1);
  test1 = 0x7b;
  if (hash_find(&dico,test1,&j,10)==1)
   printf("%lx est associée à la valeur j = %ld\n",test1,j);
  else 
   printf("%lx n'est pas dans la table\n",test1);
}