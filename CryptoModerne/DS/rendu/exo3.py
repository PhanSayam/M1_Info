dico = {':':26, '!':27, '.':28, ',':29, ' ':30, "'":31}
for i in range(10):
    dico[chr(48+i)] = 32+i
invdico = {value: key for key, value in dico.items()}


def encode_text(text: str) -> list:
    out = []
    for ch in text:
        if ch in dico:
            out.append(dico[ch])
        else:
            raise ValueError(f"Unknown character: {ch!r}")
    return out


def decode_numbers(nums: list) -> str:
    return ''.join(invdico.get(n, '?') for n in nums)


if __name__ == '__main__':
    import sys
    def usage():
        print('Usage: python exo3.py encode "text"')
        print('       python exo3.py decode "32 33 34"')
    if len(sys.argv) < 3:
        usage()
        sys.exit(1)
    cmd = sys.argv[1]
    arg = sys.argv[2]
    if cmd == 'encode':
        print(' '.join(str(x) for x in encode_text(arg)))
    elif cmd == 'decode':
        try:
            nums = [int(x) for x in arg.strip().split()]
        except Exception:
            print('Invalid numbers')
            sys.exit(1)
        print(decode_numbers(nums))
    else:
        usage()
        sys.exit(1)


# EXERCISE: Affine Cipher — recover (a,b) from two pairs (starter)
# For an affine cipher e(x) = a*x + b (mod 43), two plaintext/ciphertext
# pairs are sufficient to recover a and b.
def break_affine_cipher(p1: int, c1: int, p2: int, c2: int, mod: int = 43):
    """
    Return (a, b) such that a*p1 + b ≡ c1 (mod mod) and a*p2 + b ≡ c2 (mod mod).
    TODO: subtract equations to eliminate b and solve for a using modular inverse.
    Hint: use `inverse_mod()` or `solve_linear_congruence()` in `DS/tp_crypto_master.py`.
    """
    # from DS.tp_crypto_master import inverse_mod, solve_linear_congruence  # uncomment when ready
    pass
