def tohex(n):
# retourne la repr'esentation hexad'ecimale
# sur 2 chiffres d'un entier < 256
# fonction utile pour l'affichage des etats	
	if n < 16:
		return '0'+hex(n)[-1] 
	else:
		return hex(n)[2:]

def affiche(L):
# pour afficher le contenu des états
# en hexadecimal. La liste L contient 4
# sous listes, chacune d'entre elle represente
# une ligne d'un etat pour l'AES.	
	print(list(map(tohex,L[0])))
	print(list(map(tohex,L[1])))
	print(list(map(tohex,L[2])))
	print(list(map(tohex,L[3])))
	print()

def convert_to_state(message):
# renvoie un message ou une cl'e sous forme d''etat
# un message est une suite de caract'eres
# on remplit le tableau avec les codes ascii correspondants	
# le message fait 16 caracteres (128 bits)
# les 4 premiers constituent la premiere colonne du tableau
# et ainsi de suite.
	state = [0]*4
	for i in range(4):
		state[i] = [ord(message[0:4][i]),ord(message[4:8][i]),ord(message[8:12][i]),ord(message[12:16][i])]
	return state

def multbyalpha(x):
# multiplication par alpha
# dans le corps F_2^8	
	y = x << 1
	if (y & (1 << 8)):
		y = y ^ polynome
	return y

def multbygen(x):
# pour l'AES \alpha+1 est un générateur 
# cette fonction realise la multiplication
# d'un element x par alpha+1
	return (multbyalpha(x)^x)

def mult(a,b):
# renvoie l'élément y = a*b dans F_2^8
# en se servant de la table des log en base g
# o'u g est le g'en'erateur de F_2^8
# regarder la fonction construit_F_2_8()
# pour la construction de la table des log	
	if (a == 0) or (b == 0):
		return 0
	idxa = log_gen[a]
	idxb = log_gen[b]
	return gen[(idxa+idxb) % 255]

def construit_F_2_8():
# le corps est construit en 
# en utilisant le fait que alpha+1
# est un generateur	
# on construit la table log_t t.q
# log_t[k] = i si l'entier k represente
# l'element (alpha+1)^i, cette table n'est
# pas definie en 0. Pour la construire, on
# part de l'entier 1 et on multiplie successivement
# par alpha+1. On construit en parallele la liste "table"
# tel que table[i] = entier représentant (alpha+1)^i
	table = [1]
	log_t = [0]*256
	log_t[0] = -1
	log_t[1] = 0
	for i in range(1,255):
		aux = multbygen(table[i-1])
		table = table + [aux]
		log_t[aux] = i
	table += [1]
	return table, log_t

def inv(x):
# renvoie y = x^(-1) dans F_2^8
# en se servant de la table des log en base g 
# on suppose que cette fonction est toujours
# appelee avec x non nul	
	idx = log_gen[x]
	return gen[255-idx]

def transforme(W):
# tranforme le tableau des cl'es interm'ediaires
# repr'esent'e par 44 colonnes de taille 4 par une liste L
# compos'e de 4 sous-listes de taille 44.	
# fonction utilis'ee 'a la derniere 'etape de l'algorithme
# de g'en'eration de cl'es
	L=[0]*4
	for i in range(4):
		L[i] = [0]*44
		for j in range(44):
			L[i][j] = W[j][i]
	return L

def gen_cles(k):
# algorithmes de generation des cles intermediaires	
# a partir de la cle initiale
	RC = [0,1]
	for i in range(2,11):
		RC = RC + [multbyalpha(RC[i-1])]
	W_ = [0]*44
	for i in range(44):
		W_[i] = [0]*4
	cle_convert = convert_to_state(k)
	for j in range(4):
		for i in range(4):
			W_[i][j] = cle_convert[j][i]
	for i in range(4,44):
		temp = W_[i-1]
		if (i % 4) == 0 :
			temp = list(map(S,temp[1:]+[temp[0]]))
			temp[0] ^= RC[i//4]
		for j in range(4):
			W_[i][j] = W_[i-4][j] ^ temp[j]
	return transforme(W_)

def S(x):
# calcul de la fonction S intervenant dans
# l'AES	
	if x == 0:
		y = 0
	else:
		y = inv(x)
	result = 0
	# multiplication par la matrice
	# et ajout du vecteur constant
	for i in range(8):
		result = result ^ (
			(
				((y >> i) & 1)^((y >>((i+4) % 8)) & 1)^
			((y >> ((i+5) % 8)) & 1)^((y >> ((i+6) % 8)) & 1)^((y >> ((i+7) % 8)) & 1)^((c >> i) & 1)
				) 
			<< i)
	return result

def SubBytes(etat):
# renvoie dans state le tableau etat 
# apr'es application de la transformation S
# ceci fait appel `a la fonction S disponible
# dans ce script.
# Chaque element x du tableau etat est
# remplac'e par S(x)
	state = []
	for i in etat:
		temp = []
		for j in i : 
			temp.append(SBOX[j])
		state = state + [list(temp)]
	return state

def invSubBytes(etat):
	state = []
	for i in etat:
		temp = []
		for j in i : 
			temp.append(InvSBOX[j])
		state = state + [list(temp)]
	return state


def ShiftRows(etat):
# renvoie dans state le tableau etat 
# apr`es application de la transformation ShiftRows		
	state = [etat[0]]
	state = state + [etat[1][1:]+[etat[1][0]]]
	state = state + [etat[2][2:]+etat[2][0:2]]
	state = state + [etat[3][3:]+etat[3][0:3]]
	return state

def invShifRows(etat):
	state = [etat[0]]
	state = state + [[etat[1][3]]+etat[1][0:3]]
	state = state + [etat[2][2:4]+etat[2][0:2]]
	state = state + [etat[3][1:]+[etat[3][0]]]
	return state

def MixColumns(etat):
# renvoie dans state le tableau etat 
# apr'es application de la transformation MixColumns
# cette tranformation revient 'a multiplier chaque colonne
# de etat par la matrice mix_column	
# attention il s'agit d'une multiplication dans F_2^8.
# Chaque 'el'ement de la matrice est un 'el'ement de F_2^8
# et chaque colonne de etat est consid'er'e comme un vecteur
# de F_2^8.
	state = []
	for i in range(4):
		aux = []
		for j in range(4):
			somme = 0
			for k in range(4):
				somme = somme ^ mult(matrix_mix_columns[i][k], etat[k][j])
			aux = aux + [somme]
		state = state + [aux]
	return state 

def invMixColums(etat):
	state = []
	for i in range(4):
		aux = []
		for j in range(4):
			somme = 0
			for k in range(4):
				somme = somme ^ mult(matrix_invmix_columns[i][k], etat[k][j])
			aux = aux + [somme]
		state = state + [aux]
	return state 

def AddRoundKey(etat,tour):
# ajoute a etat la cle correspond au tour
# numero "tour"	
	state = []
	K = [0,0,0,0]
	for i in range(4):
		K[i] = W[i][4*tour:4*(tour+1)]
	for i in range(4):
		aux = []
		for j in range(4):
			aux = aux + [etat[i][j] ^ K[i][j]]
		state = state + [aux]
	return state

def invAddRoundKey(etat,K):
	state = []
	for i in range(4):
		aux = []
		for j in range(4):
			aux = aux + [etat[i][j] ^ K[i][j]]
		state = state + [aux]
	return state

# Debut du programme

polynome = 0b100011011 
# polynome x^8+x^4+x^3+x+1 pour generer F_2^8
# il n'est pas primitif
# donc alpha sa racine n'est pas un generateur
# par contre on peut montrer que alpha+1 est un generateur
c = 0b01100011 # constante pour la cr'eation des cl'es de tour
# l'op'eration mixcolumns correspond 'a une multiplication matricielle
# attention la matrice ci-dessous est constitu'ee d''elements de F_2^8
# elle correspond a la matrice :
# |alpha		alpha+1		1			1		|
# |1			alpha		alpha+1		1		|	
# |1			1			alpha		alpha+1	|
# |alpha+1		1			1			alpha	|
matrix_mix_columns = [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]
matrix_invmix_columns = [[0xe,0xb,0xd,0x9],[0x9,0xe,0xb,0xd],[0xd,0x9,0xe,0xb],[0xb,0xd,0x9,0xe]]
gen, log_gen = construit_F_2_8()

clair = "J'adore vraiment la cryptographie,bien plus que le developpement"
cle =   "Ceci est une cle"
IV = "1234567890abcdef1234567890abcdef"

message1 = clair[:16]
message2 = clair[16:32]
message3 = clair[32:48]
message4 = clair[48:]

# Deroulement de l'AES
W = gen_cles(cle)
# affichage du clair sous forme d'etat

etat=convert_to_state(clair)
print("clair = ")
affiche(etat)


SBOX = []
InvSBOX = [0]*256

for i in range(256):
	SBOX.append(S(i))
	InvSBOX[SBOX[i]] = i


etat1=convert_to_state(message1)
etat2=convert_to_state(message2)
etat3=convert_to_state(message3)
etat4=convert_to_state(message4)
print("clair = ")
affiche(etat1)
affiche(etat2)
affiche(etat3)
affiche(etat4)

IV_state = convert_to_state(IV)

def AES(etat):
	etat=AddRoundKey(etat,0)
	for i in range(1,10):
		etat = SubBytes(etat)
		etat = ShiftRows(etat)
		etat = MixColumns(etat)
		etat = AddRoundKey(etat,i)
	etat = SubBytes(etat)
	etat = ShiftRows(etat)
	etat = AddRoundKey(etat,10)
	print("crypto = ")
	affiche(etat)

#m1 XOR IV
etat1=invAddRoundKey(etat1,IV_state)
#AES sur bloc1
c1 = AES(etat1)
c1

#m2 XOR c1
etat2 = invAddRoundKey(message2,c1)
c2 = AES(etat2)
c2
#m3 XOR c2
etat3 = invAddRoundKey(message3,c2)
c3 = AES(etat3)
c3
#m4 XOR c3
etat4 = invAddRoundKey(message4,c3)
c4 = AES(etat4)
c4




