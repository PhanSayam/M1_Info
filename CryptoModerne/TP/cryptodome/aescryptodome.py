import binascii

from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad,unpad
from base64 import b64encode

def tohex(n):
	if n < 16:
		return '0'+hex(n)[-1] 
	else:
		return hex(n)[2:]

def affiche(L):
	print(list(map(tohex,L[0])))
	print(list(map(tohex,L[1])))
	print(list(map(tohex,L[2])))
	print(list(map(tohex,L[3])))
	print()

def convert_to_state(message):
	state = [0]*4
	for i in range(4):
		state[i] = [message[0:4][i],message[4:8][i],message[8:12][i],message[12:16][i]]
	return state

#Chiffrement ECB simple K fixé
clair = b"Un message clair"
cle =   b"Ceci est une cle"
# A COMPLETER POUr LA PARTIE 1

# cypher = AES.new(cle, AES.MODE_ECB)
# crypto = cypher.encrypt(clair)
# etat = convert_to_state(crypto)
# affiche(etat)


# A COMPLETER POUR LA PARTIE 2
#Chiffrement CBC, longueur message multiple de 16
# IV fix'e donn'e sous forme d'une chaine hexa
clair = b"J'adore vraiment la cryptographie,bien plus que le developpement"
IV = "1234567890abcdef1234567890abcdef"

IV_byte = binascii.unhexlify(IV)
cipher = AES.new(cle, AES.MODE_CBC, IV_byte)

crypto = cipher.encrypt(clair)
etat = convert_to_state(crypto)
etat2 = convert_to_state(crypto[16:33])
etat3 = convert_to_state(crypto[32:49])
etat4 = convert_to_state(crypto[48::])
# affiche(etat)
# affiche(etat2)
# affiche(etat3)
# affiche(etat4)


# A COMPLETER POUR LA PARTIE 3
#chiffrement CBC, longueur du texte quelconque
# IV fix'e donn'e sous forme d'une chaine hexa
clair = b"voici un texte dont la longueur n'a aucune raison d'etre un multiple de 16 et qui necessite donc du padding."
IV = "1234567890abcdef1234567890abcdef"

cipher = AES.new(cle, AES.MODE_CBC, IV_byte)
message_pad = pad(clair, AES.block_size)
ct = cipher.encrypt(message_pad)

cipher_decrypt = AES.new(cle, AES.MODE_CBC, IV_byte)
decrypt = cipher_decrypt.decrypt(ct)
pt = unpad(decrypt, AES.block_size)
# print("message : ", pt)


# A COMPLETER POUR LA PARTIE 4
#chiffrement CBC, longueur du texte quelconque
# IV g'en'er'e par la fonction encrypt
# Alice envoie IV concat'en'e avec le crypto
clair = b"voici un texte dont la longueur n'a aucune raison d'etre un multiple de 16 et qui necessite donc du padding."
cipher = AES.new(cle, AES.MODE_CBC, iv=None)
IV_gen = cipher.iv
ct = cipher.encrypt(pad(clair, AES.block_size))
ct_iv = IV_gen+ct

iv = ct_iv[:16]
message_crypte = ct_iv[16::]
cipher_decrypt = AES.new(cle, AES.MODE_CBC, iv)
decrypt = cipher_decrypt.decrypt(message_crypte)
pt = unpad(decrypt, AES.block_size)
# print("message : ", pt)


# A COMPLETER POUR LA PARTIE 5
#chiffrement CBC, longueur du texte quelconque
# IV g'en'er'e par la fonction encrypt
# Alice envoie IV chiffr'e en mode ECB concat'en'e avec le crypto
clair = b"voici un texte dont la longueur n'a aucune raison d'etre un multiple de 16 et qui necessite donc du padding."

cipher = AES.new(cle, AES.MODE_CBC, iv=None)
cipher_ECB = AES.new(cle, AES.MODE_ECB)

IV_gen = cipher.iv

ct = cipher.encrypt(pad(clair, AES.block_size)) #CBC
ct_ecb = cipher_ECB.encrypt(IV_gen) #ECB

ct_iv = ct_ecb + ct



cipher_decrypt_iv = AES.new(cle, AES.MODE_ECB)
iv = cipher_decrypt_iv.decrypt(ct_iv[:16])

message_crypte = ct_iv[16::]
cipher_decrypt = AES.new(cle, AES.MODE_CBC, iv)
decrypt = cipher_decrypt.decrypt(message_crypte)
pt = unpad(decrypt, AES.block_size)
print("message : ", pt)