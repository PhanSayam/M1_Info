# =============================================================================
# RÉFÉRENCE COMPLÈTE — CRYPTOGRAPHIE AVEC PYCRYPTODOME
# DS Final : Chiffrement public, Log discret, Signature, PKI, Hachage, RSA-OAEP
# =============================================================================

# Installation : pip install pycryptodome
# Import général selon les modules

# =============================================================================
# BLOC 1 — FONCTIONS DE HACHAGE (SHA, MD5...)
# =============================================================================

from Cryptodome.Hash import SHA256, SHA512, SHA1, MD5, SHA3_256, HMAC

# --- Hachage simple ---
def hacher_message(message: bytes) -> str:
    h = SHA256.new(message)
    return h.hexdigest()          # retourne la représentation hexadécimale

# Exemple d'utilisation
h = SHA256.new(b"mon message")
print("SHA256 hexdigest :", h.hexdigest())
print("SHA256 digest    :", h.digest())       # bytes bruts
print("SHA256 taille    :", h.digest_size)    # 32 octets = 256 bits

# --- Autres fonctions de hachage ---
h1 = SHA512.new(b"message")     # 512 bits
h2 = SHA1.new(b"message")       # 160 bits (obsolète)
h3 = MD5.new(b"message")        # 128 bits (cassé)
h4 = SHA3_256.new(b"message")   # 256 bits (construction éponge, sûr)

# --- Hacher un fichier par blocs ---
def hacher_fichier(chemin: str) -> str:
    h = SHA256.new()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(4096), b""):
            h.update(bloc)
    return h.hexdigest()

# --- Mettre à jour un haché en plusieurs fois ---
h = SHA256.new()
h.update(b"premiere partie")
h.update(b"deuxieme partie")
print(h.hexdigest())             # équivalent à SHA256(b"premiere partiedeuxieme partie")


# =============================================================================
# BLOC 2 — HMAC (Message Authentication Code avec hachage)
# =============================================================================

from Cryptodome.Hash import HMAC, SHA256

# --- Créer un HMAC ---
def creer_mac(cle: bytes, message: bytes) -> bytes:
    mac = HMAC.new(cle, message, digestmod=SHA256)
    return mac.digest()

# --- Vérifier un HMAC ---
def verifier_mac(cle: bytes, message: bytes, mac_recu: bytes) -> bool:
    mac = HMAC.new(cle, message, digestmod=SHA256)
    try:
        mac.verify(mac_recu)
        return True
    except ValueError:
        return False

# Exemple complet
cle_mac = b"cle_secrete_partagee"
message = b"message a authentifier"
mac = creer_mac(cle_mac, message)
print("MAC valide :", verifier_mac(cle_mac, message, mac))


# =============================================================================
# BLOC 3 — GÉNÉRATION DE NOMBRES ALÉATOIRES
# =============================================================================

from Cryptodome.Random import get_random_bytes
from Cryptodome.Random.random import randint

# --- Octets aléatoires (pour IV, sel, nonce...) ---
iv  = get_random_bytes(16)    # 16 octets = 128 bits
sel = get_random_bytes(16)

# --- Entier aléatoire ---
k = randint(1, 100)


# =============================================================================
# BLOC 4 — DÉRIVATION DE CLÉ : PBKDF2
# =============================================================================

from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Hash import SHA256, SHA3_256
from Cryptodome.Random import get_random_bytes

# --- Dériver une clé depuis un mot de passe ---
def deriver_cle(mot_de_passe: str, sel: bytes = None, taille: int = 16) -> tuple:
    if sel is None:
        sel = get_random_bytes(16)
    cle = PBKDF2(
        mot_de_passe.encode(),
        sel,
        dkLen=taille,           # 16 octets = AES-128, 32 octets = AES-256
        count=100000,           # nombre d'itérations (plus = plus sûr)
        hmac_hash_module=SHA256
    )
    return cle, sel

# Avec SHA3-256 (comme dans le TP)
def deriver_cle_sha3(mot_de_passe: str, sel: bytes) -> bytes:
    return PBKDF2(
        mot_de_passe.encode(),
        sel,
        dkLen=16,
        count=100000,
        hmac_hash_module=SHA3_256
    )

# Exemple
cle, sel = deriver_cle("oxxfcdbk")
print("Clé dérivée (hex):", cle.hex())


# =============================================================================
# BLOC 5 — CHIFFREMENT SYMÉTRIQUE AES
# =============================================================================

from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
from Cryptodome.Random import get_random_bytes

# --- MODE CBC (le plus courant dans les TPs) ---

def chiffrer_aes_cbc(message: bytes, cle: bytes) -> tuple:
    iv = get_random_bytes(16)
    cipher = AES.new(cle, AES.MODE_CBC, iv)
    chiffre = cipher.encrypt(pad(message, AES.block_size))
    return iv, chiffre

def dechiffrer_aes_cbc(chiffre: bytes, cle: bytes, iv: bytes) -> bytes:
    cipher = AES.new(cle, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(chiffre), AES.block_size)

# --- MODE ECB (dangereux — chaque bloc indépendant) ---

def chiffrer_aes_ecb(message: bytes, cle: bytes) -> bytes:
    cipher = AES.new(cle, AES.MODE_ECB)
    return cipher.encrypt(pad(message, AES.block_size))

def dechiffrer_aes_ecb(chiffre: bytes, cle: bytes) -> bytes:
    cipher = AES.new(cle, AES.MODE_ECB)
    return unpad(cipher.decrypt(chiffre), AES.block_size)

# --- MODE GCM (authentifié — chiffrement + intégrité) ---

def chiffrer_aes_gcm(message: bytes, cle: bytes) -> tuple:
    cipher = AES.new(cle, AES.MODE_GCM)
    chiffre, tag = cipher.encrypt_and_digest(message)
    return cipher.nonce, chiffre, tag       # nonce joue le rôle de l'IV

def dechiffrer_aes_gcm(chiffre: bytes, cle: bytes, nonce: bytes, tag: bytes) -> bytes:
    cipher = AES.new(cle, AES.MODE_GCM, nonce=nonce)
    try:
        return cipher.decrypt_and_verify(chiffre, tag)
    except ValueError:
        raise ValueError("Authentification échouée — message altéré")

# Exemple complet CBC
cle_sym = get_random_bytes(16)
message = b"Message secret a chiffrer"
iv, chiffre = chiffrer_aes_cbc(message, cle_sym)
clair = dechiffrer_aes_cbc(chiffre, cle_sym, iv)
print("Déchiffré CBC:", clair.decode())


# =============================================================================
# BLOC 6 — PROTOCOLE COMPLET : PBKDF2 + AES-CBC (format TP)
# Format : sel | IV | chiffré  encodé en base64
# =============================================================================

import base64
from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Hash import SHA3_256
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
from Cryptodome.Random import get_random_bytes

MOT_DE_PASSE = b"oxxfcdbk"

def chiffrer_message_complet(message: str) -> str:
    sel     = get_random_bytes(16)
    iv      = get_random_bytes(16)
    cle     = PBKDF2(MOT_DE_PASSE, sel, dkLen=16, count=100000, hmac_hash_module=SHA3_256)
    cipher  = AES.new(cle, AES.MODE_CBC, iv)
    chiffre = cipher.encrypt(pad(message.encode(), AES.block_size))
    return base64.b64encode(sel + iv + chiffre).decode()

def dechiffrer_message_complet(message_b64: str) -> str:
    donnees = base64.b64decode(message_b64)
    sel     = donnees[:16]
    iv      = donnees[16:32]
    chiffre = donnees[32:]
    cle     = PBKDF2(MOT_DE_PASSE, sel, dkLen=16, count=100000, hmac_hash_module=SHA3_256)
    cipher  = AES.new(cle, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(chiffre), AES.block_size).decode()

# Test
cryptogramme = chiffrer_message_complet("Message de test")
print("Chiffré  :", cryptogramme)
print("Déchiffré:", dechiffrer_message_complet(cryptogramme))


# =============================================================================
# BLOC 7 — RSA : GÉNÉRATION, CHIFFREMENT, DÉCHIFFREMENT
# =============================================================================

from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP
from Cryptodome.Hash import SHA256

# --- Générer une paire de clés RSA ---
def generer_cles_rsa(taille: int = 2048) -> tuple:
    cle = RSA.generate(taille)
    cle_privee = cle
    cle_publique = cle.publickey()
    return cle_privee, cle_publique

# --- Exporter / Importer les clés ---
def exporter_cle_pem(cle) -> str:
    return cle.export_key().decode()

def importer_cle_pem(pem: str):
    return RSA.import_key(pem.encode())

# --- Construire une clé depuis ses paramètres (n, e, d, p, q) ---
def construire_cle_rsa_privee(n, e, d, p, q):
    return RSA.construct((n, e, d, p, q))

def construire_cle_rsa_publique(n, e):
    return RSA.construct((n, e))

# --- Chiffrement RSA-OAEP ---
def chiffrer_rsa_oaep(message: bytes, cle_publique) -> bytes:
    cipher = PKCS1_OAEP.new(cle_publique, hashAlgo=SHA256)
    return cipher.encrypt(message)

def dechiffrer_rsa_oaep(chiffre: bytes, cle_privee) -> bytes:
    cipher = PKCS1_OAEP.new(cle_privee, hashAlgo=SHA256)
    return cipher.decrypt(chiffre)

# --- Accéder aux paramètres d'une clé RSA ---
cle_privee, cle_publique = generer_cles_rsa(2048)
print("n =", cle_privee.n)
print("e =", cle_privee.e)
print("d =", cle_privee.d)
print("p =", cle_privee.p)
print("q =", cle_privee.q)

# Exemple chiffrement/déchiffrement OAEP
message_rsa = b"Cle de session AES"
chiffre_rsa = chiffrer_rsa_oaep(message_rsa, cle_publique)
clair_rsa   = dechiffrer_rsa_oaep(chiffre_rsa, cle_privee)
print("Déchiffré RSA-OAEP:", clair_rsa.decode())


# =============================================================================
# BLOC 8 — CHIFFREMENT HYBRIDE COMPLET
# RSA-OAEP pour la clé de session + AES-CBC pour le message
# =============================================================================

from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP, AES
from Cryptodome.Util.Padding import pad, unpad
from Cryptodome.Random import get_random_bytes

def chiffrement_hybride(message: bytes, cle_publique_rsa) -> tuple:
    cle_session = get_random_bytes(16)                         # clé AES-128
    iv          = get_random_bytes(16)

    # Chiffrer la clé de session avec RSA-OAEP
    cipher_rsa  = PKCS1_OAEP.new(cle_publique_rsa)
    cle_chiffree = cipher_rsa.encrypt(cle_session)

    # Chiffrer le message avec AES-CBC
    cipher_aes  = AES.new(cle_session, AES.MODE_CBC, iv)
    msg_chiffre = cipher_aes.encrypt(pad(message, AES.block_size))

    return cle_chiffree, iv, msg_chiffre

def dechiffrement_hybride(cle_chiffree: bytes, iv: bytes,
                           msg_chiffre: bytes, cle_privee_rsa) -> bytes:
    # Déchiffrer la clé de session
    cipher_rsa  = PKCS1_OAEP.new(cle_privee_rsa)
    cle_session = cipher_rsa.decrypt(cle_chiffree)

    # Déchiffrer le message
    cipher_aes = AES.new(cle_session, AES.MODE_CBC, iv)
    return unpad(cipher_aes.decrypt(msg_chiffre), AES.block_size)

# Exemple
cle_priv, cle_pub = generer_cles_rsa()
cle_chiffree, iv, msg_chiffre = chiffrement_hybride(b"Message long", cle_pub)
clair = dechiffrement_hybride(cle_chiffree, iv, msg_chiffre, cle_priv)
print("Hybride déchiffré:", clair.decode())


# =============================================================================
# BLOC 9 — SIGNATURE RSA-PSS (PKCS#1 v2.1)
# =============================================================================

from Cryptodome.PublicKey import RSA
from Cryptodome.Signature import pss
from Cryptodome.Hash import SHA256
import base64

# --- Signer avec RSA-PSS ---
def signer_rsa_pss(message: bytes, cle_privee) -> bytes:
    h = SHA256.new(message)
    signer = pss.new(cle_privee)
    return signer.sign(h)

# --- Vérifier une signature RSA-PSS ---
def verifier_rsa_pss(message: bytes, signature: bytes, cle_publique) -> bool:
    h = SHA256.new(message)
    verifier = pss.new(cle_publique)
    try:
        verifier.verify(h, signature)
        return True
    except ValueError:
        return False

# --- Vérifier depuis une signature encodée en base64 ---
def verifier_rsa_pss_base64(message: bytes, signature_b64: str, cle_publique) -> bool:
    signature = base64.b64decode(signature_b64)     # TOUJOURS décoder avant verify
    return verifier_rsa_pss(message, signature, cle_publique)

# --- Construire clé publique depuis (n, e) et vérifier ---
def verifier_depuis_parametres(message: bytes, signature_b64: str, n: int, e: int) -> bool:
    cle_pub = RSA.construct((n, e))
    return verifier_rsa_pss_base64(message, signature_b64, cle_pub)

# Exemple complet
cle_priv, cle_pub = generer_cles_rsa()
message = b"Je signe mon message avec RSA-PSS"
sig = signer_rsa_pss(message, cle_priv)
print("Signature valide:", verifier_rsa_pss(message, sig, cle_pub))
print("Message altéré  :", verifier_rsa_pss(b"autre message", sig, cle_pub))


# =============================================================================
# BLOC 10 — SIGNATURE DSA (FIPS-186-3)
# =============================================================================

from Cryptodome.PublicKey import DSA
from Cryptodome.Signature import DSS
from Cryptodome.Hash import SHA256

# --- Générer une clé DSA ---
def generer_cles_dsa(taille: int = 2048):
    cle = DSA.generate(taille)
    return cle, cle.publickey()

# --- Afficher les paramètres DSA ---
def afficher_parametres_dsa(cle):
    print("p =", cle.p)
    print("q =", cle.q)
    print("g =", cle.g)
    print("y =", cle.y)   # clé publique
    print("x =", cle.x)   # clé secrète (seulement sur clé privée)

# --- Signer avec DSA ---
def signer_dsa(message: bytes, cle_privee) -> bytes:
    h = SHA256.new(message)
    signer = DSS.new(cle_privee, 'fips-186-3')
    return signer.sign(h)

# --- Vérifier une signature DSA ---
def verifier_dsa(message: bytes, signature: bytes, cle_publique) -> bool:
    h = SHA256.new(message)
    verifier = DSS.new(cle_publique, 'fips-186-3')
    try:
        verifier.verify(h, signature)
        return True
    except ValueError:
        return False

# --- Construire une clé DSA depuis les paramètres (y, g, p, q, x) ---
def construire_cle_dsa(y, g, p, q, x):
    # True = clé privée (contient x), False = clé publique
    return DSA.construct((y, g, p, q, x), True)

def construire_cle_dsa_publique(y, g, p, q):
    return DSA.construct((y, g, p, q))

# Exemple avec les paramètres du TP
p = 21982318881223953334999182554595205538084193111543983730717487230159656932130365772494500760478003874043989336097133871512269471794698526917932847071284383115197223243991455774591726638907923888559045204552869730263688779378724036878702423372434562085196107573359236549708848433307877439684487333137001552089475693382062813912658959065457933236641716667110579652948619119011472256705612423828546393299364509385751525362561194583423496421553302746619706995129788028971287954909729561068836933318116155092057614516956183967544334676331280353621897557453824386780830506535276889588232798882128170031921251985394979100821
q = 15392901590359449378224604923394484367524403031577922083259055488969
g = 11187023668015798624512678976363575189270549188959571448425107948872770202123075192276701018382428322979338623329688525893435052245163111435758174858625837738544367374591733535590712266726864610275124071131054593427360326708225788640155606184176326771982180936150158946803774935771991898866937540609692433636253252341549220178114133040662444962563263188995439086184529814302594682178980913265939839867652533490264013366498383505792092983664560235983709082956762313727424247841958875604284479303964689337173041083295320832237624580777499906337067007927293170236364923986492968506345287726154016043094897023082214489199
y = 9765301154172541585270791518435323087213897112848515903966377315268169472807557150056089477383034973538737969059210466047552347257919141154295678629762780665562526437265490113009795272149754967022959188602434633964951583593004816173511710006199148655683377385368258265945887211143381481705529597355166083992883893774533192245503860827400738313569940817504354311258701457268662037002336590856379348426145607101497727791112789026128949395821470615577291508357024235097023053792068876118033902583600783384451324902562490970806413587924213065058485538529569849050600040995102206047153502637499293641931433872494196055859
x = 11875430977526496396795733469820081394048450164566056799676184013339

message_dsa = b'Je signe mon premier message avec DSA'

# Construire la clé et signer
cle_dsa = construire_cle_dsa(y, g, p, q, x)
signature_dsa = signer_dsa(message_dsa, cle_dsa)

# Vérifier
cle_pub_dsa = construire_cle_dsa_publique(y, g, p, q)
print("DSA valide:", verifier_dsa(message_dsa, signature_dsa, cle_pub_dsa))


# =============================================================================
# BLOC 11 — ENCODAGE BASE64
# =============================================================================

import base64

# --- Encoder en base64 ---
def encoder_base64(donnees: bytes) -> str:
    return base64.b64encode(donnees).decode()

# --- Décoder depuis base64 ---
def decoder_base64(texte: str) -> bytes:
    return base64.b64decode(texte)

# Exemple
donnees = b"donnees binaires"
encoded = encoder_base64(donnees)
decoded = decoder_base64(encoded)
print("Base64:", encoded)
print("Décodé:", decoded)


# =============================================================================
# BLOC 12 — OPÉRATIONS MODULAIRES UTILES
# =============================================================================

# --- Inverse modulaire ---
def inverse_mod(a: int, n: int) -> int:
    return pow(a, -1, n)          # Python 3.8+

# --- Exponentiation modulaire ---
def exp_mod(base: int, exposant: int, modulo: int) -> int:
    return pow(base, exposant, modulo)

# --- PGCD et algorithme d'Euclide ---
import math
def pgcd(a: int, b: int) -> int:
    return math.gcd(a, b)

# --- Nombre de solutions de ax ≡ b (mod n) ---
def nb_solutions_lineaire(a: int, b: int, n: int) -> int:
    g = math.gcd(a, n)
    if b % g != 0:
        return 0
    return g

# --- Résoudre ax ≡ b (mod n) --- retourne toutes les solutions
def resoudre_lineaire(a: int, b: int, n: int):
    g = math.gcd(a, n)
    if b % g != 0:
        return []
    a_, b_, n_ = a // g, b // g, n // g
    x0 = (b_ * pow(a_, -1, n_)) % n_
    return [x0 + k * n_ for k in range(g)]

# Exemple : 6x ≡ 4 (mod 8)
solutions = resoudre_lineaire(6, 4, 8)
print("Solutions de 6x≡4(mod 8):", solutions)   # [2, 6]


# =============================================================================
# BLOC 13 — LOGARITHME DISCRET : BABY-STEP GIANT-STEP
# =============================================================================

import math

def baby_step_giant_step(g: int, b: int, p: int) -> int:
    m = int(math.ceil(math.sqrt(p)))

    # Phase 1 — Baby Steps : calculer g^j pour j = 0..m-1
    table = {}
    val = 1
    for j in range(m):
        table[val] = j
        val = (val * g) % p

    # Phase 2 — Giant Steps : calculer b * (g^{-m})^i pour i = 0..m-1
    g_inv_m = pow(g, -m, p)    # g^{-m} mod p
    z = b
    for i in range(m + 1):
        if z in table:
            x = i * m + table[z]
            # Vérification
            if pow(g, x, p) == b:
                return x
        z = (z * g_inv_m) % p

    return None    # pas trouvé

# Exemple du TP
p = 1934459
g = 762973
b = 1191663
x = baby_step_giant_step(g, b, p)
print(f"Log discret: g^{x} ≡ {b} (mod {p})")
print(f"Vérification: {pow(g, x, p)} == {b}: {pow(g, x, p) == b}")


# =============================================================================
# BLOC 14 — DIFFIE-HELLMAN MANUEL
# =============================================================================

import math
from Cryptodome.Random.random import randint

def diffie_hellman(p: int, g: int) -> tuple:
    # Chaque partie génère son exposant secret
    x = randint(2, p - 2)    # exposant secret d'Alice
    y = randint(2, p - 2)    # exposant secret de Bob

    # Chaque partie calcule sa valeur publique
    A = pow(g, x, p)         # Alice envoie A = g^x mod p
    B = pow(g, y, p)         # Bob envoie B = g^y mod p

    # Chaque partie calcule le secret partagé
    K_alice = pow(B, x, p)   # Alice calcule B^x = g^{xy}
    K_bob   = pow(A, y, p)   # Bob calcule A^y = g^{xy}

    assert K_alice == K_bob, "Erreur : les secrets ne correspondent pas"
    return K_alice

# Exemple avec petits paramètres
p_dh = 11
g_dh = 2
K = diffie_hellman(p_dh, g_dh)
print("Secret DH partagé:", K)


# =============================================================================
# BLOC 15 — ELGAMAL MANUEL
# =============================================================================

from Cryptodome.Random.random import randint

def elgamal_chiffrer(m: int, p: int, a: int, b: int) -> tuple:
    # b = a^x mod p (clé publique)
    k = randint(1, p - 2)
    c1 = pow(a, k, p)
    c2 = (m * pow(b, k, p)) % p
    return c1, c2

def elgamal_dechiffrer(c1: int, c2: int, p: int, x: int) -> int:
    # x = clé secrète
    s = pow(c1, x, p)           # s = c1^x = a^{kx}
    s_inv = pow(s, -1, p)       # inverse de s mod p
    return (c2 * s_inv) % p

# Exemple
p_eg = 11
a_eg = 2                         # générateur
x_eg = 3                         # clé secrète
b_eg = pow(a_eg, x_eg, p_eg)    # clé publique = 2^3 mod 11 = 8

m_eg = 5
c1, c2 = elgamal_chiffrer(m_eg, p_eg, a_eg, b_eg)
m_dec = elgamal_dechiffrer(c1, c2, p_eg, x_eg)
print(f"ElGamal: m={m_eg}, chiffré=({c1},{c2}), déchiffré={m_dec}")


# =============================================================================
# BLOC 16 — DSA MANUEL (sans bibliothèque)
# =============================================================================

import math
from Cryptodome.Random.random import randint

def dsa_signer_manuel(m_hash: int, a: int, p: int, q: int, alpha: int) -> tuple:
    while True:
        k = randint(1, q - 1)
        gamma = pow(alpha, k, p) % q
        if gamma == 0:
            continue
        k_inv = pow(k, -1, q)
        delta = (k_inv * (m_hash + a * gamma)) % q
        if delta == 0:
            continue
        return gamma, delta

def dsa_verifier_manuel(m_hash: int, gamma: int, delta: int,
                         p: int, q: int, alpha: int, beta: int) -> bool:
    if not (0 < gamma < q and 0 < delta < q):
        return False
    delta_inv = pow(delta, -1, q)
    e1 = (m_hash * delta_inv) % q
    e2 = (gamma * delta_inv) % q
    v = (pow(alpha, e1, p) * pow(beta, e2, p)) % p % q
    return v == gamma


# =============================================================================
# BLOC 17 — ATTAQUE PAR RÉUTILISATION DU NONCE DSA
# (Si même k utilisé deux fois → retrouver la clé privée)
# =============================================================================

def retrouver_cle_privee_dsa(m1_hash: int, m2_hash: int,
                              delta1: int, delta2: int,
                              gamma: int, q: int) -> int:
    # gamma identique → même k utilisé
    numerateur   = (m1_hash - m2_hash) % q
    denominateur = (delta1 - delta2) % q
    k = (numerateur * pow(denominateur, -1, q)) % q
    a = ((delta1 * k - m1_hash) * pow(gamma, -1, q)) % q
    return a     # clé privée récupérée


# =============================================================================
# BLOC 18 — GESTION DES CLÉS : EXPORT / IMPORT / SÉRIALISATION
# =============================================================================

from Cryptodome.PublicKey import RSA, DSA

# --- RSA : export PEM ---
cle_rsa, _ = generer_cles_rsa(1024)
pem_prive   = cle_rsa.export_key('PEM').decode()
pem_public  = cle_rsa.publickey().export_key('PEM').decode()

# --- RSA : import PEM ---
cle_importee = RSA.import_key(pem_prive.encode())

# --- RSA : export DER (format binaire) ---
der = cle_rsa.export_key('DER')

# --- Vérifier si une clé est privée ou publique ---
print("Est clé privée:", cle_rsa.has_private())
print("Est clé privée:", cle_rsa.publickey().has_private())   # False


# =============================================================================
# BLOC 19 — PADDING : PAD / UNPAD (PKCS7)
# =============================================================================

from Cryptodome.Util.Padding import pad, unpad

# --- Ajouter du padding ---
message_court = b"Hello"
padded = pad(message_court, 16)    # complète jusqu'à un multiple de 16
print("Avec padding:", padded, "longueur:", len(padded))

# --- Retirer le padding ---
original = unpad(padded, 16)
print("Sans padding:", original)


# =============================================================================
# BLOC 20 — EXEMPLE COMPLET : SCÉNARIO RÉALISTE
# Alice signe et chiffre un message à Bob
# =============================================================================

from Cryptodome.PublicKey import RSA
from Cryptodome.Signature import pss
from Cryptodome.Cipher import PKCS1_OAEP, AES
from Cryptodome.Hash import SHA256
from Cryptodome.Util.Padding import pad, unpad
from Cryptodome.Random import get_random_bytes
import base64

def scenario_complet():
    # Génération des clés
    cle_alice_priv = RSA.generate(2048)
    cle_alice_pub  = cle_alice_priv.publickey()
    cle_bob_priv   = RSA.generate(2048)
    cle_bob_pub    = cle_bob_priv.publickey()

    message = b"Message confidentiel d'Alice a Bob"

    # ---- ALICE : Signer puis chiffrer ----

    # Étape 1 : Alice signe le message avec sa clé privée
    h = SHA256.new(message)
    signature = pss.new(cle_alice_priv).sign(h)

    # Étape 2 : Alice chiffre (message + signature) avec AES
    cle_session = get_random_bytes(16)
    iv = get_random_bytes(16)
    payload = message + b"||SEP||" + signature
    cipher_aes = AES.new(cle_session, AES.MODE_CBC, iv)
    chiffre = cipher_aes.encrypt(pad(payload, AES.block_size))

    # Étape 3 : Alice chiffre la clé de session avec la clé publique de Bob
    cipher_rsa = PKCS1_OAEP.new(cle_bob_pub)
    cle_chiffree = cipher_rsa.encrypt(cle_session)

    # Transmission : (cle_chiffree, iv, chiffre)

    # ---- BOB : Déchiffrer puis vérifier ----

    # Étape 1 : Bob déchiffre la clé de session avec sa clé privée
    cipher_rsa = PKCS1_OAEP.new(cle_bob_priv)
    cle_session_recue = cipher_rsa.decrypt(cle_chiffree)

    # Étape 2 : Bob déchiffre le message
    cipher_aes = AES.new(cle_session_recue, AES.MODE_CBC, iv)
    payload_recu = unpad(cipher_aes.decrypt(chiffre), AES.block_size)

    # Étape 3 : Bob sépare message et signature
    message_recu, sig_recue = payload_recu.split(b"||SEP||")

    # Étape 4 : Bob vérifie la signature d'Alice
    h2 = SHA256.new(message_recu)
    try:
        pss.new(cle_alice_pub).verify(h2, sig_recue)
        print("Signature Alice valide ✓")
    except ValueError:
        print("Signature invalide ✗")

    print("Message reçu:", message_recu.decode())

scenario_complet()


# =============================================================================
# RÉCAPITULATIF DES IMPORTS
# =============================================================================

"""
from Cryptodome.Hash import SHA256, SHA512, SHA1, MD5, SHA3_256, HMAC
from Cryptodome.PublicKey import RSA, DSA
from Cryptodome.Signature import pss, DSS
from Cryptodome.Cipher import AES, PKCS1_OAEP
from Cryptodome.Util.Padding import pad, unpad
from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Random import get_random_bytes
from Cryptodome.Random.random import randint
import base64
import math
"""

# =============================================================================
# PIÈGES À NE PAS COMMETTRE
# =============================================================================

"""
1. TOUJOURS base64.b64decode(signature) avant de passer à .verify()

2. DSA.construct prend les paramètres dans l'ordre (y, g, p, q, x)
   pas (p, q, g, y, x)

3. Pour vérifier DSA, utiliser la clé PUBLIQUE uniquement
   DSS.new(cle.publickey(), 'fips-186-3')

4. pad() et unpad() prennent la taille de bloc en deuxième argument
   pad(message, AES.block_size)  -- AES.block_size = 16

5. PBKDF2 retourne des bytes, pas une string

6. pss.new(cle) crée un objet signeur/vérificateur
   il faut ensuite appeler .sign(hash) ou .verify(hash, signature)

7. SHA256.new() crée un nouvel objet de hachage
   .update(data) met à jour
   .digest() retourne bytes
   .hexdigest() retourne string hexadécimale

8. AES.new() crée un nouveau cipher -- ne jamais réutiliser
   un objet cipher pour chiffrer ET déchiffrer
"""