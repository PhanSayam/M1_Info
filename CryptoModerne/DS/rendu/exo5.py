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
        print('Usage: python exo5.py encode "text"')
        print('       python exo5.py decode "32 33 34"')
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


# EXERCISE: RSA primes recovery from phi(n) (starter)
# Given n and phi(n) recover the prime factors p and q (p>q).
def recover_rsa_primes(n: int, phi: int):
    """
    Solve for p and q from n and phi where phi = (p-1)*(q-1).
    TODO: compute s = n - phi + 1 (sum p+q), discriminant = s^2 - 4n, then roots.
    Hint: use `math.isqrt()` and verify roots; `rsa_from_primes()` in `DS/tp_crypto_master.py` helps validate.
    """
    # import math
    # from DS.tp_crypto_master import rsa_from_primes  # uncomment when ready
    pass
