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
        print('Usage: python exo1.py encode "text"')
        print('       python exo1.py decode "32 33 34"')
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


# EXERCISE: Chinese Remainder Theorem (CRT) — Hastad-like attack (starter)
# Given three RSA ciphertexts of the same short plaintext m with exponent e=3,
# recover m assuming m**3 < n1*n2*n3.
def solve_rsa_hastad(c1: int, n1: int, c2: int, n2: int, c3: int, n3: int):
    """
    Recover plaintext m from three RSA ciphertexts.
    TODO: implement using CRT then integer cube root.
    Hint: use `crt()` and `iroot()` from `DS/tp_crypto_master.py`.
    """
    # from DS.tp_crypto_master import crt, iroot  # uncomment to use
    pass
