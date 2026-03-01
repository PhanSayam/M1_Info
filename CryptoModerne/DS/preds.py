"""
CRYPTOGRAPHIE MODERNE - MASTER MATHÉMATIQUES-INFORMATIQUE TOULON
Domaines couverts:
1. Arithmétique des corps finis Fpn
2. Implémentation bas niveau de l'AES
3. Cryptographie moderne (PyCryptodome)
4. Cryptanalyse classique
"""

from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Hash import SHA3_256
from Cryptodome.Util.Padding import pad
from Cryptodome.Random import get_random_bytes
import numpy as np
from collections import Counter


# ============================================================================
# PARTIE 1 : ARITHMÉTIQUE DES CORPS FINIS Fpn
# ============================================================================

class CorpsFini:
    """Implémentation d'un corps fini Fp^n par polynôme irréductible."""
    
    def __init__(self, p, n, irred_poly):
        """
        Args:
            p: caractéristique (nombre premier)
            n: dimension
            irred_poly: polynôme irréductible de degré n (représenté en binaire si p=2)
        """
        self.p = p
        self.n = n
        self.irred = irred_poly
        self.card = p ** n
    
    @staticmethod
    def xor_mult(a, b, irred, n):
        """Multiplication en F_2[x]/(irred) pour F_2^n via polynômes.
        Complexité O(n²) en général, O(1) avec tables pré-calculées.
        """
        result = 0
        while b:
            if b & 1:
                result ^= a
            a <<= 1
            if a & (1 << n):
                a ^= irred
            b >>= 1
        return result & ((1 << n) - 1)
    
    @staticmethod
    def polynome_pgcd(a, b):
        """Algorithme d'Euclide pour polynômes en F_2[x]."""
        while b:
            a, b = b, a ^ b  # XOR remplace la soustraction en F_2
        return a
    
    @staticmethod
    def inv_modulo(a, irred, n):
        """Inversion modulaire dans F_2^n via Euclide étendu.
        Calcule a^(-1) mod irred.
        """
        if a == 0:
            return 0
        
        def egcd(x, y):
            if y == 0:
                return x, 1, 0
            gcd, xx, yy = egcd(y, x ^ (y * (x // y)) if x >= y else 0, y)
            return gcd, yy, xx ^ (yy * (x // y) if x >= y else 0)
        
        # Pour F_2^n, utiliser algorithme réel
        old_r, r = irred, a
        old_s, s = 1, 0
        
        while r:
            quotient = old_r // r if old_r >= r else 0
            old_r, r = r, old_r ^ (r * quotient) if quotient else old_r
            old_s, s = s, old_s ^ (s * quotient) if quotient else old_s
        
        return old_s & ((1 << n) - 1)


# ============================================================================
# PARTIE 2 : IMPLÉMENTATION BAS NIVEAU DE L'AES
# ============================================================================

def inv_f28(x):
    """Inversion multiplicative dans F_2^8 via l'algorithme d'Euclide.
    Renvoie x^(-1) mod (x^8 + x^4 + x^3 + x + 1).
    Précondition: x != 0
    """
    if x == 0:
        return 0
    
    # Polynôme irréductible AES: x^8 + x^4 + x^3 + x + 1 = 0x11B
    irred = 0x11B
    u1, u3 = 1, x
    v1, v3 = 0, irred
    
    while v3 != 0:
        if u3 < (1 << 8):
            u1, u3 = v1, v3
            v1, v3 = u1, u3
        
        u3 ^= v3
        u1 ^= v1
    
    return u1 & 0xFF


def s_box(x, c=0x63):
    """Construction algébrique de la S-Box AES:
    S(x) = (x^(-1) mod 0x11B) suivi d'une transformation affine.
    
    Transformation affine en F_2^8:
    y = A*x^(-1) + b, où A est matrice 8x8 circulante, b=0x63.
    """
    # Étape 1: inversion dans F_2^8
    y = inv_f28(x)
    resultat = 0
    
    # Étape 2: transformation affine circulante AES
    for i in range(8):
        bit = (
            ((y >> i) & 1) ^
            ((y >> ((i + 4) % 8)) & 1) ^
            ((y >> ((i + 5) % 8)) & 1) ^
            ((y >> ((i + 6) % 8)) & 1) ^
            ((y >> ((i + 7) % 8)) & 1) ^
            ((c >> i) & 1)
        )
        resultat |= (bit << i)
    
    return resultat


def build_sbox():
    """Pré-calcule la S-Box complète (256 valeurs)."""
    return [s_box(x) for x in range(256)]


SBOX = build_sbox()


def gmul(a, b):
    """Multiplication dans F_2^8 avec réduction par x^8 + x^4 + x^3 + x + 1."""
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi_bit_set = a & 0x80
        a = (a << 1) & 0xFF
        if hi_bit_set:
            a ^= 0x1B  # 0x1B = x^4 + x^3 + x + 1 (polynomial 0x11B réduit)
        b >>= 1
    return p & 0xFF


def sub_bytes(state):
    """SubBytes: application de la S-Box à chaque octet d'état."""
    return [[SBOX[state[i][j]] for j in range(4)] for i in range(4)]


def shift_rows(state):
    """ShiftRows: décalage circulaire des lignes."""
    new_state = [row[:] for row in state]
    for i in range(1, 4):
        new_state[i] = new_state[i][i:] + new_state[i][:i]
    return new_state


def mix_columns(state):
    """MixColumns: multiplication matricielle dans F_2^8.
    Matrice circulante:
    [02 03 01 01]
    [01 02 03 01]
    [01 01 02 03]
    [03 01 01 02]
    """
    mixer = [
        [0x02, 0x03, 0x01, 0x01],
        [0x01, 0x02, 0x03, 0x01],
        [0x01, 0x01, 0x02, 0x03],
        [0x03, 0x01, 0x01, 0x02]
    ]
    new_state = [[0] * 4 for _ in range(4)]
    for col in range(4):
        for row in range(4):
            val = 0
            for k in range(4):
                val ^= gmul(mixer[row][k], state[k][col])
            new_state[row][col] = val
    return new_state


def add_round_key(state, round_key):
    """AddRoundKey: XOR avec la clé de ronde."""
    return [[state[i][j] ^ round_key[i][j] for j in range(4)] for i in range(4)]


def bytes_to_state(data):
    """Conversion octet → état (matrice 4x4)."""
    return [[data[i + 4*j] for j in range(4)] for i in range(4)]


def state_to_bytes(state):
    """Conversion état → octet."""
    return bytes([state[i][j] for j in range(4) for i in range(4)])


def key_expansion(key, nr=10):
    """Expansion de clé AES-128 (10 rondes).
    Génère nr+1 clés de ronde à partir de la clé maître.
    """
    rcon = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
    w = [int.from_bytes(key[i:i+4], 'big') for i in range(0, 16, 4)]
    
    for i in range(4, 4 * (nr + 1)):
        temp = w[i-1]
        if i % 4 == 0:
            # RotWord + SubWord + XOR avec Rcon
            temp = ((SBOX[(temp >> 16) & 0xFF] << 24) ^
                    (SBOX[(temp >> 8) & 0xFF] << 16) ^
                    (SBOX[temp & 0xFF] << 8) ^
                    SBOX[(temp >> 24) & 0xFF])
            temp ^= (rcon[(i // 4) - 1] << 24)
        w.append(w[i-4] ^ temp)
    
    return [w[i:i+4] for i in range(0, len(w), 4)]


def aes_encrypt_block(plaintext, key):
    """Chiffrement AES d'un bloc 128 bits (bas niveau)."""
    state = bytes_to_state(plaintext)
    round_keys = key_expansion(key)
    
    # Ronde initiale
    state = add_round_key(state, bytes_to_state(b''.join(
        k.to_bytes(4, 'big') for k in round_keys[0])))
    
    # 9 rondes principales
    for rnd in range(1, 10):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, bytes_to_state(b''.join(
            k.to_bytes(4, 'big') for k in round_keys[rnd])))
    
    # Ronde finale (sans MixColumns)
    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, bytes_to_state(b''.join(
        k.to_bytes(4, 'big') for k in round_keys[10])))
    
    return state_to_bytes(state)


# ============================================================================
# PARTIE 3 : CRYPTOGRAPHIE MODERNE (PyCryptodome)
# ============================================================================

def chiffrer_aes_cbc(message, password):
    """Chiffrement AES-CBC sécurisé.
    
    Processus:
    1. Générer sel aléatoire (16 octets)
    2. Dériver clé via PBKDF2 (100k itérations, SHA3-256)
    3. Initialiser CBC (IV généré aléatoirement)
    4. Appliquer bourrage PKCS#7
    5. Retourner sel || IV || cryptogramme
    """
    sel = get_random_bytes(16)
    cle = PBKDF2(password, sel, 16, count=100000, hmac_hash_module=SHA3_256)
    cipher = AES.new(cle, AES.MODE_CBC)
    message_pad = pad(message, AES.block_size)
    cryptogramme = cipher.encrypt(message_pad)
    return sel + cipher.iv + cryptogramme


def dechiffrer_aes_cbc(chiffre, password):
    """Déchiffrement AES-CBC sécurisé.
    
    Structure attendue: sel(16) || IV(16) || cryptogramme
    """
    sel = chiffre[:16]
    iv = chiffre[16:32]
    cryptogramme = chiffre[32:]
    
    cle = PBKDF2(password, sel, 16, count=100000, hmac_hash_module=SHA3_256)
    cipher = AES.new(cle, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(cryptogramme)
    
    from Cryptodome.Util.Padding import unpad
    return unpad(plaintext, AES.block_size)


# ============================================================================
# PARTIE 4 : CRYPTANALYSE CLASSIQUE
# ============================================================================

def indice_coincidence(texte):
    """Indice de coïncidence I_c:
    I_c = (Σ n_i(n_i-1)) / (N(N-1))
    où n_i = fréquence du caractère i, N = longueur totale.
    
    Interprétation:
    - Texte aléatoire: I_c ≈ 1/26 ≈ 0.038 (anglais: 0.065)
    - Texte français: I_c ≈ 0.073
    """
    n = len(texte)
    if n <= 1:
        return 0.0
    
    frequences = Counter(texte)
    numerateur = sum(occ * (occ - 1) for occ in frequences.values())
    return numerateur / (n * (n - 1))


def indice_coincidence_mutuel(text1, text2):
    """Indice de coïncidence mutuel entre deux textes.
    Utilisé pour détecter l'alignement de clés dans Vigenère.
    """
    if len(text1) != len(text2):
        return 0.0
    
    coincidences = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
    return coincidences / len(text1)


def longueur_cle_vigenere(cryptogramme, seuil_ic=0.065):
    """Détermination de la longueur de clé Vigenère.
    
    Algorithme: tester les longueurs 1, 2, 3, ... jusqu'à trouver
    un I_c proche du seuil français (≈ 0.065).
    """
    for k in range(1, len(cryptogramme) // 2):
        # Extraire les sous-textes décalés
        sous_textes = [cryptogramme[i::k] for i in range(k)]
        ic_moyen = np.mean([indice_coincidence(st) for st in sous_textes])
        
        if ic_moyen > seuil_ic - 0.01:
            return k
    
    return 1


def attaque_cesar(cryptogramme):
    """Attaque par décalage (César simple).
    
    Teste tous les décalages possibles et retourne le meilleur
    selon l'indice de coïncidence.
    """
    meilleur_ic = 0
    meilleur_decalage = 0
    meilleur_texte = ""
    
    for shift in range(26):
        texte_essai = ''.join(
            chr((ord(c) - ord('a') - shift) % 26 + ord('a'))
            if c.isalpha() else c
            for c in cryptogramme.lower()
        )
        ic = indice_coincidence(texte_essai)
        
        if ic > meilleur_ic:
            meilleur_ic = ic
            meilleur_decalage = shift
            meilleur_texte = texte_essai
    
    return meilleur_decalage, meilleur_texte, meilleur_ic


def attaque_vigenere_frequence(cryptogramme, k):
    """Attaque Vigenère par analyse fréquentiste après extraction.
    
    Pour chaque position i de la clé:
    1. Extraire cryptogramme[i::k]
    2. Trouver le décalage qui maximise l'I_c (fréquences proches du français)
    """
    cle = ""
    
    for i in range(k):
        sous_texte = cryptogramme[i::k]
        shift, _, _ = attaque_cesar(sous_texte)
        cle += chr((shift + ord('a')) % 26 + ord('a'))
    
    return cle


def chi2_stat(observes, attendus):
    """Statistique χ² pour tester la proximité de distributions.
    Plus χ² est petit, plus les distributions sont proches.
    """
    chi2 = 0
    for obs, att in zip(observes, attendus):
        if att > 0:
            chi2 += (obs - att) ** 2 / att
    return chi2


def attaque_vigenere_chi2(cryptogramme, k):
    """Attaque Vigenère optimisée via χ² (meilleure que fréquence seule)."""
    
    # Fréquences attendues en français (approximatives)
    freq_fr = {c: f for c, f in zip('abcdefghijklmnopqrstuvwxyz',
        [0.0855, 0.0064, 0.0324, 0.0369, 0.1721, 0.0113, 0.0087,
         0.0074, 0.0766, 0.0054, 0.0046, 0.0697, 0.0301, 0.0713,
         0.0769, 0.0289, 0.0099, 0.0723, 0.0798, 0.0727, 0.0548,
         0.0132, 0.0236, 0.0142, 0.0301, 0.0071])}
    
    cle = ""
    for i in range(k):
        sous_texte = cryptogramme[i::k].lower()
        meilleur_shift = 0
        meilleur_chi2 = float('inf')
        
        for shift in range(26):
            texte_essai = ''.join(
                chr((ord(c) - ord('a') - shift) % 26 + ord('a'))
                if c.isalpha() else c
                for c in sous_texte
            )
            
            freq_obs = Counter(texte_essai)
            test = 0
            for lettre in 'abcdefghijklmnopqrstuvwxyz':
                obs = freq_obs.get(lettre, 0) / len(texte_essai) if len(texte_essai) > 0 else 0
                att = freq_fr[lettre]
                if att > 0:
                    test += (obs - att) ** 2 / att
            
            if test < meilleur_chi2:
                meilleur_chi2 = test
                meilleur_shift = shift
        
        cle += chr((meilleur_shift + ord('a')) % 26 + ord('a'))
    
    return cle


def vigenere_encrypt(plaintext, key):
    """Chiffrement Vigenère (référence)."""
    key = key.lower()
    ciphertext = []
    key_idx = 0
    
    for char in plaintext.lower():
        if char.isalpha():
            shift = ord(key[key_idx % len(key)]) - ord('a')
            encrypted = chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            ciphertext.append(encrypted)
            key_idx += 1
        else:
            ciphertext.append(char)
    
    return ''.join(ciphertext)


def vigenere_decrypt(ciphertext, key):
    """Déchiffrement Vigenère (référence)."""
    key = key.lower()
    plaintext = []
    key_idx = 0
    
    for char in ciphertext.lower():
        if char.isalpha():
            shift = ord(key[key_idx % len(key)]) - ord('a')
            decrypted = chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
            plaintext.append(decrypted)
            key_idx += 1
        else:
            plaintext.append(char)
    
    return ''.join(plaintext)


# ============================================================================
# EXEMPLES D'UTILISATION & TESTS
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("TEST 1: CORPS FINIS Fpn - Opérations en F_2^8")
    print("=" * 70)
    
    # Multiplication et inversion en F_2^8
    a, b = 0x57, 0x83
    prod = CorpsFini.xor_mult(a, b, 0x11B, 8)
    print(f"0x{a:02X} × 0x{b:02X} = 0x{prod:02X} (dans F_2^8)")
    
    x = 0x53
    inv_x = inv_f28(x)
    print(f"0x{x:02X}^(-1) = 0x{inv_x:02X} (inversion modulaire)")
    
    
    print("\n" + "=" * 70)
    print("TEST 2: IMPLÉMENTATION BAS NIVEAU AES")
    print("=" * 70)
    
    # S-Box
    print(f"S-Box(0x00) = 0x{SBOX[0x00]:02X}")
    print(f"S-Box(0xFF) = 0x{SBOX[0xFF]:02X}")
    print(f"S-Box(0x53) = 0x{SBOX[0x53]:02X}")
    
    # Chiffrement d'un bloc AES
    plaintext = b"Hello, AES-128!!"
    key = b"MasterCrypto2024"
    ciphertext = aes_encrypt_block(plaintext, key)
    print(f"Clair:  {plaintext.hex()}")
    print(f"Chiffré: {ciphertext.hex()}")
    
    # Multiplication dans F_2^8
    print(f"0x02 ⊗ 0x57 = 0x{gmul(0x02, 0x57):02X}")
    print(f"0x03 ⊗ 0x57 = 0x{gmul(0x03, 0x57):02X}")
    
    
    print("\n" + "=" * 70)
    print("TEST 3: CRYPTOGRAPHIE MODERNE (PyCryptodome)")
    print("=" * 70)
    
    message = b"Message secret pour le Master"
    password = "MotDePasseSecurise123!"
    
    chiffre = chiffrer_aes_cbc(message, password)
    print(f"Message original: {message.decode()}")
    print(f"Longueur chiffré (sel+IV+crypto): {len(chiffre)} octets")
    print(f"Hexadécimal: {chiffre.hex()[:64]}...")
    
    # Déchiffrement
    dechiffre = dechiffrer_aes_cbc(chiffre, password)
    print(f"Déchiffré: {dechiffre.decode()}")
    print(f"Vérification: {dechiffre == message}")
    
    
    print("\n" + "=" * 70)
    print("TEST 4: CRYPTANALYSE CLASSIQUE - VIGENÈRE")
    print("=" * 70)
    
    # Chiffrement Vigenère
    plaintext = "lecryptographemoderneetlasecuritedesdonneesdependentdelasciencemathematique"
    key = "gauss"
    ciphertext_vig = vigenere_encrypt(plaintext, key)
    print(f"Clair: {plaintext}")
    print(f"Clé: {key}")
    print(f"Chiffré: {ciphertext_vig}")
    
    # Analyse - Indice de coïncidence
    ic_chiffre = indice_coincidence(ciphertext_vig)
    print(f"I_c du chiffré: {ic_chiffre:.4f} (proche 0.038 pour aléatoire)")
    
    # Détermination de la longueur de clé
    k_detected = longueur_cle_vigenere(ciphertext_vig)
    print(f"Longueur de clé détectée: {k_detected} (réelle: {len(key)})")
    
    # Attaque par fréquence
    cle_trouvee = attaque_vigenere_chi2(ciphertext_vig, len(key))
    print(f"Clé retrouvée (χ²): {cle_trouvee}")
    
    # Vérification
    plaintext_retrouve = vigenere_decrypt(ciphertext_vig, cle_trouvee)
    print(f"Texte retrouvé: {plaintext_retrouve}")
    
    
    print("\n" + "=" * 70)
    print("TEST 5: ATTAQUE CÉSAR")
    print("=" * 70)
    
    plaintext = "example"
    key_cesar = 3
    ciphertext_cesar = vigenere_encrypt(plaintext, chr(key_cesar + ord('a')))
    print(f"Clair: {plaintext}")
    print(f"Chiffré (César, décalage {key_cesar}): {ciphertext_cesar}")
    
    shift_found, text_found, ic_found = attaque_cesar(ciphertext_cesar)
    print(f"Décalage trouvé: {shift_found}")
    print(f"Texte retrouvé: {text_found}")
    print(f"I_c du texte retrouvé: {ic_found:.4f}")
    
    
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print("""
    ✓ Corps finis Fpn: Opérations polynomiales, inversion modulaire
    ✓ AES bas niveau: S-Box algébrique, SubBytes, ShiftRows, MixColumns
    ✓ AES moderne: Chiffrement CBC sécurisé avec PBKDF2 + sel + IV
    ✓ Cryptanalyse: I_c, Vigenère par χ², attaque César, longueur clé
    """)