"""PCG32 must match the reference implementation bit for bit (the TS port is tested against
golden files generated from this module)."""

from trustlap.rng import PCG32, splitmix32


def test_matches_pcg_reference_outputs() -> None:
    # First outputs of the PCG reference demo (pcg32-demo, seed 42, stream 54).
    rng = PCG32(42, 54)
    expected = [0xA15C02B7, 0x7B47F409, 0xBA1D3330, 0x83D2F293, 0xBFA4784B, 0xCBED606E]
    assert [rng.next_u32() for _ in expected] == expected


def test_same_seed_same_sequence() -> None:
    a, b = PCG32(7), PCG32(7)
    assert [a.next_u32() for _ in range(100)] == [b.next_u32() for _ in range(100)]


def test_different_seeds_differ() -> None:
    a, b = PCG32(7), PCG32(8)
    assert [a.next_u32() for _ in range(10)] != [b.next_u32() for _ in range(10)]


def test_float_and_int_ranges() -> None:
    rng = PCG32(123)
    floats = [rng.next_range(-2.0, 3.0) for _ in range(10_000)]
    assert min(floats) >= -2.0 and max(floats) < 3.0
    ints = [rng.next_int(8, 14) for _ in range(10_000)]
    assert set(ints) == set(range(8, 15))


def test_splitmix32_is_a_32bit_permutation_sample() -> None:
    outs = {splitmix32(i) for i in range(10_000)}
    assert len(outs) == 10_000
    assert all(0 <= o <= 0xFFFFFFFF for o in outs)
