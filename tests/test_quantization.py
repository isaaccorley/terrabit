import numpy as np

from terrabit._typing import NDArrayF32
from terrabit.quantization import (
    _random_orthogonal_matrix,
    dequantize,
    dequantize_binary,
    dequantize_float16,
    dequantize_fp8,
    dequantize_int2,
    dequantize_int3,
    dequantize_int4,
    dequantize_int8,
    dequantize_turbo,
    hamming_distance,
    quantize,
    quantize_binary,
    quantize_float16,
    quantize_fp8,
    quantize_int2,
    quantize_int3,
    quantize_int4,
    quantize_int8,
    quantize_turbo,
)


def _rand_embeddings(n: int = 100, d: int = 64, seed: int = 42) -> NDArrayF32:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


class TestFloat16:
    def test_roundtrip(self):
        x = _rand_embeddings()
        q = quantize_float16(x)
        assert q.dtype == np.float16
        assert q.shape == x.shape
        r = dequantize_float16(q)
        assert r.dtype == np.float32
        np.testing.assert_allclose(x, r, atol=2e-3, rtol=1e-3)


class TestInt8:
    def test_roundtrip(self):
        x = _rand_embeddings()
        q, scale, zp = quantize_int8(x)
        assert q.dtype == np.int8
        assert q.shape == x.shape
        r = dequantize_int8(q, scale, zp)
        np.testing.assert_allclose(x, r, atol=0.05)

    def test_constant_column(self):
        x = np.ones((50, 10), dtype=np.float32) * 3.0
        q, scale, zp = quantize_int8(x)
        r = dequantize_int8(q, scale, zp)
        np.testing.assert_allclose(x, r, atol=0.05)


class TestFp8:
    def test_roundtrip(self):
        x = _rand_embeddings()
        q = quantize_fp8(x)
        assert q.dtype == np.uint8
        assert q.shape == x.shape
        r = dequantize_fp8(q)
        np.testing.assert_allclose(x, r, atol=0.25)


class TestInt4:
    def test_roundtrip(self):
        x = _rand_embeddings()
        packed, scale, zp, n_dims = quantize_int4(x)
        assert packed.dtype == np.uint8
        assert packed.shape == (x.shape[0], (x.shape[1] + 1) // 2)
        r = dequantize_int4(packed, scale, zp, n_dims)
        np.testing.assert_allclose(x, r, atol=0.35)

    def test_non_multiple_of_2_dims(self):
        x = _rand_embeddings(n=10, d=13)
        packed, scale, zp, n_dims = quantize_int4(x)
        assert packed.shape == (10, 7)
        r = dequantize_int4(packed, scale, zp, n_dims)
        assert r.shape == x.shape


class TestInt3:
    def test_roundtrip(self):
        x = _rand_embeddings()
        packed, scale, zp, n_dims = quantize_int3(x)
        assert packed.dtype == np.uint8
        expected_nbytes = (x.shape[1] * 3 + 7) // 8
        assert packed.shape == (x.shape[0], expected_nbytes)
        r = dequantize_int3(packed, scale, zp, n_dims)
        np.testing.assert_allclose(x, r, atol=0.5)

    def test_non_multiple_of_8_bits(self):
        x = _rand_embeddings(n=7, d=13)
        packed, scale, zp, n_dims = quantize_int3(x)
        assert packed.shape == (7, 5)  # ceil(13*3/8) = 5
        r = dequantize_int3(packed, scale, zp, n_dims)
        assert r.shape == x.shape


class TestInt2:
    def test_roundtrip(self):
        x = _rand_embeddings()
        packed, scale, zp, n_dims = quantize_int2(x)
        assert packed.dtype == np.uint8
        expected_nbytes = (x.shape[1] * 2 + 7) // 8
        assert packed.shape == (x.shape[0], expected_nbytes)
        r = dequantize_int2(packed, scale, zp, n_dims)
        np.testing.assert_allclose(x, r, atol=1.1)

    def test_non_multiple_of_4_dims(self):
        x = _rand_embeddings(n=9, d=13)
        packed, scale, zp, n_dims = quantize_int2(x)
        assert packed.shape == (9, 4)  # ceil(13*2/8) = 4
        r = dequantize_int2(packed, scale, zp, n_dims)
        assert r.shape == x.shape


class TestBinary:
    def test_shape(self):
        x = _rand_embeddings(n=10, d=64)
        packed = quantize_binary(x)
        assert packed.dtype == np.uint8
        assert packed.shape == (10, 8)  # 64 bits / 8 = 8 bytes

    def test_roundtrip_signs(self):
        x = _rand_embeddings(n=10, d=64)
        packed = quantize_binary(x)
        r = dequantize_binary(packed, 64)
        expected_signs = np.where(x > 0, 1.0, -1.0)
        np.testing.assert_array_equal(r, expected_signs)

    def test_non_multiple_of_8(self):
        x = _rand_embeddings(n=5, d=13)
        packed = quantize_binary(x)
        assert packed.shape == (5, 2)  # ceil(13/8) = 2
        r = dequantize_binary(packed, 13)
        assert r.shape == (5, 13)

    def test_hamming(self):
        a = np.array([[0xFF, 0x00]], dtype=np.uint8)
        b = np.array([[0x00, 0xFF]], dtype=np.uint8)
        d = hamming_distance(a, b)
        assert d[0, 0] == 16  # all bits flipped


class TestTurbo:
    def test_rotation_matrix_orthogonal(self):
        r = _random_orthogonal_matrix(64, seed=0)
        np.testing.assert_allclose(r @ r.T, np.eye(64), atol=1e-5)
        np.testing.assert_allclose(r.T @ r, np.eye(64), atol=1e-5)

    def test_rotation_matrix_deterministic(self):
        r1 = _random_orthogonal_matrix(64, seed=42)
        r2 = _random_orthogonal_matrix(64, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_rotation_preserves_distances(self):
        x = _rand_embeddings(n=50, d=64)
        r = _random_orthogonal_matrix(64, seed=0)
        x_rot = x @ r
        from scipy.spatial.distance import cdist

        d_orig = cdist(x, x, metric="euclidean")
        d_rot = cdist(x_rot, x_rot, metric="euclidean")
        np.testing.assert_allclose(d_orig, d_rot, atol=1e-4)

    def test_turbo4_roundtrip(self):
        x = _rand_embeddings(n=100, d=64)
        packed, scale, zp, n_dims, _seed = quantize_turbo(x, bits=4, seed=0)
        assert packed.dtype == np.uint8
        r = dequantize_turbo(packed, scale, zp, n_dims, bits=4, seed=0)
        assert r.dtype == np.float32
        assert r.shape == x.shape
        # Reconstruction should be decent (rotation + 4-bit quant)
        cos_sim = np.mean(
            np.sum(x * r, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(r, axis=1) + 1e-8)
        )
        assert cos_sim > 0.95

    def test_turbo2_roundtrip(self):
        x = _rand_embeddings(n=100, d=64)
        packed, scale, zp, n_dims, _seed = quantize_turbo(x, bits=2, seed=0)
        r = dequantize_turbo(packed, scale, zp, n_dims, bits=2, seed=0)
        assert r.shape == x.shape

    def test_turbo_via_unified_api(self):
        x = _rand_embeddings(n=50, d=32)
        for method in ("turbo8", "turbo4", "turbo3", "turbo2"):
            result = quantize(x, method)
            r = dequantize(result)
            assert r.dtype == np.float32
            assert r.shape == x.shape


class TestUnifiedAPI:
    def test_quantize_dequantize_all_methods(self):
        x = _rand_embeddings(n=50, d=32)
        for method in (
            "float16",
            "fp8",
            "int8",
            "int4",
            "int3",
            "int2",
            "binary",
            "turbo8",
            "turbo4",
            "turbo3",
            "turbo2",
        ):
            result = quantize(x, method)
            r = dequantize(result)
            assert r.dtype == np.float32
            assert r.shape[0] == x.shape[0]
