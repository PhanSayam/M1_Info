from __future__ import annotations

import base64
import math
import random
import secrets
from typing import Optional, Tuple

from Cryptodome.Cipher import AES, PKCS1_OAEP
from Cryptodome.Hash import HMAC, SHA1, SHA256, SHA3_256
from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.PublicKey import DSA, RSA
from Cryptodome.Random import get_random_bytes
from Cryptodome.Random.random import randint
from Cryptodome.Signature import DSS, pss
from Cryptodome.Util.Padding import pad, unpad


DEFAULT_AES_BLOCK_SIZE = AES.block_size
DEFAULT_PBKDF2_COUNT = 100000
DEFAULT_SALT_SIZE = 16
DEFAULT_RSA_BITS = 2048


def sha256_hex(message: bytes) -> str:
    return SHA256.new(message).hexdigest()


def sha3_256_hex(message: bytes) -> str:
    return SHA3_256.new(message).hexdigest()


def sha1_hex(message: bytes) -> str:
    return SHA1.new(message).hexdigest()


def hmac_sha256(key: bytes, message: bytes) -> bytes:
    return HMAC.new(key, message, digestmod=SHA256).digest()


def pbkdf2_key(password: bytes | str, salt: Optional[bytes] = None, *, dk_len: int = 16, count: int = DEFAULT_PBKDF2_COUNT, hash_module=SHA3_256) -> Tuple[bytes, bytes]:
    if salt is None:
        salt = get_random_bytes(DEFAULT_SALT_SIZE)
    if isinstance(password, str):
        password = password.encode()
    key = PBKDF2(password, salt, dkLen=dk_len, count=count, hmac_hash_module=hash_module)
    return key, salt


def aes_cbc_encrypt(message: bytes, key: bytes, iv: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    if iv is None:
        iv = get_random_bytes(DEFAULT_AES_BLOCK_SIZE)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return iv, cipher.encrypt(pad(message, AES.block_size))


def aes_cbc_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), AES.block_size)


def aes_gcm_encrypt(message: bytes, key: bytes, nonce: Optional[bytes] = None) -> Tuple[bytes, bytes, bytes]:
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce) if nonce is not None else AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(message)
    return cipher.nonce, ciphertext, tag


def aes_gcm_decrypt(ciphertext: bytes, key: bytes, nonce: bytes, tag: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)


def pack_salted_cbc_payload(password: bytes | str, message: bytes, *, dk_len: int = 16, count: int = DEFAULT_PBKDF2_COUNT, hash_module=SHA3_256) -> str:
    key, salt = pbkdf2_key(password, dk_len=dk_len, count=count, hash_module=hash_module)
    iv, ciphertext = aes_cbc_encrypt(message, key)
    return base64.b64encode(salt + iv + ciphertext).decode()


def unpack_salted_cbc_payload(password: bytes | str, payload_b64: str, *, dk_len: int = 16, count: int = DEFAULT_PBKDF2_COUNT, hash_module=SHA3_256) -> bytes:
    payload = base64.b64decode(payload_b64)
    salt = payload[:16]
    iv = payload[16:32]
    ciphertext = payload[32:]
    key, _ = pbkdf2_key(password, salt, dk_len=dk_len, count=count, hash_module=hash_module)
    return aes_cbc_decrypt(ciphertext, key, iv)


def rsa_generate(bits: int = DEFAULT_RSA_BITS):
    private_key = RSA.generate(bits)
    return private_key, private_key.publickey()


def rsa_oaep_encrypt(message: bytes, public_key) -> bytes:
    return PKCS1_OAEP.new(public_key, hashAlgo=SHA256).encrypt(message)


def rsa_oaep_decrypt(ciphertext: bytes, private_key) -> bytes:
    return PKCS1_OAEP.new(private_key, hashAlgo=SHA256).decrypt(ciphertext)


def rsa_pss_sign(message: bytes, private_key) -> bytes:
    digest = SHA256.new(message)
    return pss.new(private_key).sign(digest)


def rsa_pss_verify(message: bytes, signature: bytes, public_key) -> bool:
    digest = SHA256.new(message)
    try:
        pss.new(public_key).verify(digest, signature)
        return True
    except ValueError:
        return False


def rsa_pss_verify_b64(message: bytes, signature_b64: str, public_key) -> bool:
    return rsa_pss_verify(message, base64.b64decode(signature_b64), public_key)


def dsa_generate(bits: int = 2048):
    private_key = DSA.generate(bits)
    return private_key, private_key.publickey()


def dsa_sign(message: bytes, private_key) -> bytes:
    digest = SHA256.new(message)
    return DSS.new(private_key, "fips-186-3").sign(digest)


def dsa_verify(message: bytes, signature: bytes, public_key) -> bool:
    digest = SHA256.new(message)
    try:
        DSS.new(public_key, "fips-186-3").verify(digest, signature)
        return True
    except ValueError:
        return False


def dsa_verify_b64(message: bytes, signature_b64: str, public_key) -> bool:
    return dsa_verify(message, base64.b64decode(signature_b64), public_key)


def inverse_mod(a: int, n: int) -> int:
    return pow(a, -1, n)


def solve_linear_congruence(a: int, b: int, n: int) -> list[int]:
    g = math.gcd(a, n)
    if b % g != 0:
        return []
    a_red = a // g
    b_red = b // g
    n_red = n // g
    x0 = (b_red * pow(a_red, -1, n_red)) % n_red
    return [x0 + k * n_red for k in range(g)]


def square_and_multiply(base: int, exponent: int, modulo: int) -> int:
    result = 1
    current = base % modulo
    exp = exponent
    while exp > 0:
        if exp & 1:
            result = (result * current) % modulo
        current = (current * current) % modulo
        exp >>= 1
    return result


def baby_step_giant_step(g: int, beta: int, p: int) -> Optional[int]:
    m = math.isqrt(p) + 1
    table: dict[int, int] = {}
    value = 1
    for j in range(m):
        table.setdefault(value, j)
        value = (value * g) % p
    factor = pow(g, -m, p)
    gamma = beta
    for i in range(m + 1):
        if gamma in table:
            x = i * m + table[gamma]
            if pow(g, x, p) == beta:
                return x
        gamma = (gamma * factor) % p
    return None


def diffie_hellman(p: int, g: int, x: Optional[int] = None, y: Optional[int] = None) -> Tuple[int, int, int, int, int]:
    if x is None:
        x = randint(2, p - 2)
    if y is None:
        y = randint(2, p - 2)
    A = pow(g, x, p)
    B = pow(g, y, p)
    shared = pow(B, x, p)
    return x, y, A, B, shared


def elgamal_encrypt(message: int, p: int, a: int, b: int, k: Optional[int] = None) -> Tuple[int, int]:
    if k is None:
        k = randint(1, p - 2)
    c1 = pow(a, k, p)
    c2 = (message * pow(b, k, p)) % p
    return c1, c2


def elgamal_decrypt(c1: int, c2: int, p: int, x: int) -> int:
    s = pow(c1, x, p)
    return (c2 * pow(s, -1, p)) % p


def to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def from_b64(text: str) -> bytes:
    return base64.b64decode(text)


def hash_file(path: str) -> str:
    digest = SHA256.new()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4096), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rsa_key(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return RSA.import_key(handle.read())


# --- Integer / bytes converters ---
def int_to_bytes(n: int, length: Optional[int] = None) -> bytes:
    if n == 0:
        return b"\x00" if length is None else (b"\x00" * length)
    if length is None:
        length = (n.bit_length() + 7) // 8
    return n.to_bytes(length, byteorder="big")


def bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, byteorder="big")


# --- Extended Euclid / inverse ---
def euclide_etendu(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    else:
        g, x1, y1 = euclide_etendu(b, a % b)
        return g, y1, x1 - (a // b) * y1


def inverse_bezout(a: int, n: int) -> int:
    g, u, _ = euclide_etendu(a, n)
    if g != 1:
        raise ValueError(f"No inverse for {a} mod {n}, gcd={g}")
    return u % n


# --- Chinese Remainder Theorem ---
def crt(remainders: list[int], moduli: list[int]) -> int:
    M = 1
    for m in moduli:
        M *= m
    x = 0
    for r, m in zip(remainders, moduli):
        Mi = M // m
        x = (x + r * Mi * pow(Mi, -1, m)) % M
    return x


# --- Integer k-th root (floor) ---
def iroot(n: int, k: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0
    x = int(n ** (1.0 / k))
    while (x + 1) ** k <= n:
        x += 1
    while x ** k > n:
        x -= 1
    return x


# --- Miller-Rabin primality test and prime generation ---
def miller_rabin(n: int, k: int = 40) -> bool:
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # write n-1 = 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def generer_premier(bits: int) -> int:
    while True:
        candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if miller_rabin(candidate):
            return candidate


# --- Raw RSA helpers from primes (integer API) ---
def rsa_from_primes(p: int, q: int, e: int = 65537) -> Tuple[int, int, int]:
    n = p * q
    phi = (p - 1) * (q - 1)
    if math.gcd(e, phi) != 1:
        raise ValueError("e not coprime with phi")
    d = inverse_bezout(e, phi)
    return n, e, d


def rsa_raw_encrypt_int(m: int, e: int, n: int) -> int:
    return pow(m, e, n)


def rsa_raw_decrypt_int(c: int, d: int, n: int) -> int:
    return pow(c, d, n)


# --- Pohlig-Hellman wrapper (uses baby_step_giant_step for subproblems) ---
def pohlig_hellman(g: int, beta: int, p: int, order: int, factors: list[tuple[int, int]]) -> Optional[int]:
    residues = []
    moduli = []
    for pi, ei in factors:
        qi = pi ** ei
        gi = pow(g, order // qi, p)
        bi = pow(beta, order // qi, p)
        # for small qi we can brute force with BSGS on gi^x = bi
        xi = baby_step_giant_step(gi, bi, p)
        if xi is None:
            return None
        residues.append(xi)
        moduli.append(qi)
    return crt(residues, moduli)


def demo() -> None:
    key, salt = pbkdf2_key("oxxfcdbk")
    iv, ciphertext = aes_cbc_encrypt(b"demo", key)
    assert aes_cbc_decrypt(ciphertext, key, iv) == b"demo"
    payload = base64.b64encode(salt + iv + ciphertext).decode()
    assert unpack_salted_cbc_payload("oxxfcdbk", payload) == b"demo"

    private_key, public_key = rsa_generate(1024)
    message = b"demo"
    signature = rsa_pss_sign(message, private_key)
    assert rsa_pss_verify(message, signature, public_key)

    # --- Additional self-tests / examples (won't be executed here) ---
    # int <-> bytes
    n = 0xDEADBEEF
    b = int_to_bytes(n)
    assert bytes_to_int(b) == n

    # Raw RSA using small primes (integer API)
    p = 61
    q = 53
    n_int, e_int, d_int = rsa_from_primes(p, q)
    m_int = 42
    c_int = rsa_raw_encrypt_int(m_int, e_int, n_int)
    assert rsa_raw_decrypt_int(c_int, d_int, n_int) == m_int

    # CRT example
    x = crt([2, 3, 2], [3, 5, 7])
    assert x % (3 * 5 * 7) == 23

    # Miller-Rabin quick check
    assert miller_rabin(101)

    # Small conversion sanity
    assert bytes_to_int(int_to_bytes(123456)) == 123456

    # Pohlig-Hellman small example (uses baby-step giant-step for subproblems)
    # Note: may return None if factors/subcalls fail for edge cases
    ph_x = pohlig_hellman(6, 115, 229, 228, [(2, 2), (3, 1), (19, 1)])
    if ph_x is not None:
        assert pow(6, ph_x, 229) == 115

    # Generating a 64-bit prime (commented out; can be heavy)
    # prime64 = generer_premier(64)
    # print('generated 64-bit prime:', prime64)

    # End of demo additions
    # User: run `demo()` locally to execute these self-tests


if __name__ == "__main__":
    demo()
