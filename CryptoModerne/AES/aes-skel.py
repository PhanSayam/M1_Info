def convert_to_state(message):
# renvoie un message ou une cl'e sous forme d''etat
# un etat est un tableau 4x4 qui contient les codes ascii
# des lettres du message 
# la fonction affiche (cf. ligne 64) se charge de la conversion
# en hexadecimal pour l'affichage.
	state = [0]*4
	# A COMPLETER
	for i in range(4):
		state[i] = [0]*4
	
	for column in range(4):
		for line in range(4):
			state[line][column] = ord(message[4*column+line])
 
	return state



def multbyalpha(x):
# A COMPLETER
# fontion de multiplication par alpha
# dans F_2^8
	# p = x^8 + x^4 + x^3 + x + 1
	p = 0b100011011
	y = x << 1
	if y >= 256:
		y = y ^ p
	return y


def multbygen(x):
# pour l'AES \alpha+1 est un générateur 
	return (multbyalpha(x)^x)

def mult(a,b):
# A COMPLETER
# renvoie l'élément y = a*b dans F_2^8
# en se servant de la table des log en base g
# o'u g est le g'en'erateur de F_2^8
# regarder la fonction construit_F_2_8()
# pour la table des log
	if (a == 0) or (b == 0):
		return 0
	# A COMPLETER
	# m = polynome.bit_length()-1
	m = 8
	table, log_t = construit_F_2_8()
 	
	exp_a = log_t[a]
	exp_b = log_t[b]

	somme = (exp_a + exp_b) % ((1 << m) - 1)
	y = table[somme]
	return y



def inv(x):
# A COMPLETER
# renvoie y = x^(-1) dans F_2^8
# en se servant de la table des log en base g 
# on suppose que cette fonction est toujours
# appelee avec x non nul
# on rappelle que (alpha+1) est un generateur
# donc x est de la forme (alpha+1)^j, j etant donne
# par la table log_t. 
# on rappelle de plus que (alpha+1)^7 = 1.
	table, log_t = construit_F_2_8()
	j = log_t[x]
	y = table[255-j]
	return y


def transforme(W):
# tranforme le tableau des cl'es interm'ediaires
# repr'esent'e par 44 colonnes de taille 4 par une liste L
# compos'e de 4 sous-listes de taille 44.
	L=[0]*4
	for i in range(4):
		L[i] = [0]*44
		for j in range(44):
			L[i][j] = W[j][i]
	return L

def tohex(n):
# retourne la repr'esentation hexad'ecimale
# sur 2 chiffres d'un entier < 256
	if n < 16:
		return '0'+hex(n)[-1] 
	else:
		return hex(n)[2:]

def affiche(L):
# pour afficher les états
	print(list(map(tohex,L[0])))
	print(list(map(tohex,L[1])))
	print(list(map(tohex,L[2])))
	print(list(map(tohex,L[3])))
	print()


def construit_F_2_8():
#A COMPLETER
# le corps est construit en 
# en utilisant le fait que alpha+1
# est un generateur
# dans table[i] on met l'entier qui represente (alpha+1)^i
# il faut donc partir de 1 et utiliser la fonction multbygen
# afin de calculer les puissances successives de apha+1 pour obtenir
# les entiers qui représentent ces valeurs.
# dans log_t[i] on met j si i est l'entier qui represente (alpha+1)^j
# log_t[0] est indéfini
	j = 1
	table = [1]
	log_t = [0]*256
	log_t[1] = 0
	for i in range(1,256):
		table.append(multbygen(table[i-1]))
		j = multbygen(j)
		log_t[j] = i
           # A COMPLETER
	return table, log_t

def S(x):
#veritable tranformation S de l'AES
	if x == 0:
		y = 0
	else:
		y = inv(x)
	result = 0
	for i in range(8):
		result = result ^ (
			(
				((y >> i) & 1)^((y >>((i+4) % 8)) & 1)^
			((y >> ((i+5) % 8)) & 1)^((y >> ((i+6) % 8)) & 1)^((y >> ((i+7) % 8)) & 1)^((c >> i) & 1)
				) 
			<< i)
	return result

def gen_cles(k):
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


def SubBytes(etat):
# renvoie dans state le tableau etat 
# après application de la transformation S
# ceci fait appel à la fonction S disponible
# dans ce script
# il faut donc remplacer chaque element x de etat
# par S(x)
	state = [0]*4
	# A COMPLETER
	for i in range(4):
		state[i] = [0]*4
	
	for i in range(len(etat)) :
		for j in range(len(etat)) :
			state[i][j] = S(etat[i][j])
	return state

def ShiftRows(etat):
# renvoie dans state le tableau etat 
# après application de la transformation ShiftRows	
	state = [etat[0]]
	# A COMPLETER
	for i in range(1,4):
		state = state + [etat[i][i:]+etat[i][:i]]
	return state

def MixColumns(etat):
# renvoie dans state le tableau etat 
# apr'es application de la transformation MixColumns
# cette tranformation revient 'a multiplier chaque colonne
# de etat par la matrice mix_column	
# attention il s'agit d'une multiplication dans F_2^8.
# Chaque 'el'ement de la matrice est un 'el'ement de F_2^8
# et chaque colonne de etat est consid'er'e comme un vecteur
# de F_2^8, voir section 4.2.3 page 12 de aes-standard.pdf
# et le transparent 7 de aes-exemple.pdf
	state = []
	# A COMPLETER
	for i in range(4):
		aux = []
		for j in range(4):
			somme = 0
			for k in range(4):
				somme = somme ^ mult(matrix_mix_columns[i][k],etat[k][j])
			aux = aux + [somme]
		state = state + [aux]
	
	return state 

def AddRoundKey(etat,tour):
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

polynome = 0b100011011 
# polynome x^8+x^4+x^3+x+1 pour generer F_2^8
# il n'est pas primitif
# donc alpha sa racine n'est pas un generateur
# par contre on peut montrer que alpha+1 est un generateur
c = 0b01100011 # constante pour la cr'eation des cl'es de tour
# l'op'eration mixcolumns correspond 'a une multiplication matricielle
# attention la matrice ci-dessous est constitu'ee d''elements de F_2^8
# elle correspond a la matrice :
# |alpha		alpha+1		1			1	|
# |1			alpha		alpha+1		1		|	
# |1			1			alpha		alpha+1	|
# |alpha+1		1			1			alpha	|
matrix_mix_columns = [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]
gen, log_gen = construit_F_2_8()

cle = "Thats my Kung Fu"
clair = "Two One Nine Two"

W = gen_cles(cle)
etat=convert_to_state(clair)
affiche(etat)

# Deroulement de l'AES
etat=AddRoundKey(etat,0)
for i in range(1,10):
	etat = SubBytes(etat)
	etat = ShiftRows(etat)
	etat = MixColumns(etat)
	etat = AddRoundKey(etat,i)
etat = SubBytes(etat)
etat = ShiftRows(etat)
etat = AddRoundKey(etat,10)
# affichage du cryptogramme
affiche(etat)


print(multbyalpha(2))
print(multbyalpha(128))
print(multbyalpha(253))

print(multbygen(2))
print(multbygen(128))
print(multbygen(253))

print(mult(2,64))
print(mult(128,2))
print(mult(253,4))

print(gen)

print(log_gen)

print(inv(2))
print(inv(128))
print(inv(253))