from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Hash import SHA3_256
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad,unpad
from Cryptodome.Random import get_random_bytes
from base64 import b64decode, b64encode

password = b'oxxfcdbk'

# message64 = "eNoOJ+8PtUR2m3p9Pihfl28nK3/bIHYAaG0IKOf86+E0K0n+t/5VghwWEBIKAJW28kWJZ3DbB8XPOdKHA68RhwXgwDGAYbW3GwA6G4e97/dYnKSGfvzNg14ykJFOqX8UhYgIXW4vVRLEKZ1JE2OI0mqnyIb8tZRgZjJAyX77i8XJw6gdm2BjFURu1CdNOTyZRvc8cz2uR0poc0eGABOu4g=="
# message = b64decode(message64)

# salt = message[:16]
# iv_chiffre = message[16:32]
# clair_chiffre = message[32:]

# cle = PBKDF2(password, salt, 16, count=100000, hmac_hash_module=SHA3_256)
# cipher_ECB = AES.new(cle, AES.MODE_ECB)
# IV = cipher_ECB.decrypt(iv_chiffre)

# cipher = AES.new(cle, AES.MODE_CBC, iv=IV)
# message_final = cipher.decrypt(clair_chiffre)
# final = unpad(message_final, AES.block_size)
# # print(final)



message_a_envoyer = b"Je pense enfin avoir compris le fonctionnement du protocole PBKDF2, c'est en fait assez simple."

message = "UxLVXPNCy3JPmHNB5+lnIrSJmb0e3Lb3i7K2VbHukSJ2KxNOjLzHvmTsR27xWm6O"
sel = message[:16]
IV = message[16:32]
crypto = message[32:]


salt = get_random_bytes(16)
key = PBKDF2(password, salt, 16, count=100000, hmac_hash_module=SHA3_256)

cipher_CBC = AES.new(key, AES.MODE_CBC, iv=None)
IV = cipher_CBC.iv
message_CBC = cipher_CBC.encrypt(pad(message_a_envoyer,AES.block_size))

cipher_ECB = AES.new(key, AES.MODE_ECB)
IV_ECB = cipher_ECB.encrypt(IV)

envoi = b64encode(salt+IV_ECB+message_CBC)

print(envoi)
 