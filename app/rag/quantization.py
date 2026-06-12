import math
import os
from typing import List, Optional, Tuple

RAG_EMBEDDING_QUANTIZE = os.getenv("RAG_EMBEDDING_QUANTIZE", "none").lower()
CALIBRATION_SAMPLE_RATIO = 0.01

_SCALE: Optional[float] = None
_ZERO_POINT: Optional[float] = None
_CALIBRATED = False


def calibrate_quantization(vectors: List[List[float]]):
    global _SCALE, _ZERO_POINT, _CALIBRATED
    if not vectors:
        return
    n = max(1, int(len(vectors) * CALIBRATION_SAMPLE_RATIO))
    sample = vectors[:n]
    all_vals = [v for vec in sample for v in vec]
    if not all_vals:
        return
    mn, mx = min(all_vals), max(all_vals)
    rg = mx - mn
    if rg < 1e-8:
        rg = 1.0
    _SCALE = rg / 254.0
    _ZERO_POINT = -mn / _SCALE
    _CALIBRATED = True


def reset_calibration():
    global _SCALE, _ZERO_POINT, _CALIBRATED
    _SCALE = _ZERO_POINT = None
    _CALIBRATED = False


def quantize_embedding(vector: List[float]) -> Tuple[bytes, float, float]:
    if not _CALIBRATED:
        calibrate_quantization([vector])
    s = _SCALE or 1.0
    zp = _ZERO_POINT or 0.0
    q = bytearray(len(vector))
    for i, v in enumerate(vector):
        clipped = max(0, min(255, round(v / s + zp)))
        q[i] = clipped & 0xFF
    return bytes(q), s, zp


def dequantize_embedding(quantized: bytes, scale: float, zero_point: float, dim: int = 384) -> List[float]:
    result = [0.0] * dim
    n = min(len(quantized), dim)
    for i in range(n):
        result[i] = (quantized[i] - zero_point) * scale
    return result


def quantize_vector_batch(vectors: List[List[float]]) -> Tuple[List[bytes], float, float]:
    if not vectors:
        return [], 1.0, 0.0
    if not _CALIBRATED:
        calibrate_quantization(vectors)
    s = _SCALE or 1.0
    zp = _ZERO_POINT or 0.0
    quantized = []
    dim = len(vectors[0])
    for vec in vectors:
        q = bytearray(dim)
        for i, v in enumerate(vec):
            clipped = max(0, min(255, round(v / s + zp)))
            q[i] = clipped & 0xFF
        quantized.append(bytes(q))
    return quantized, s, zp


def cosine_similarity_quantized(q_a: bytes, q_b: bytes, scale: float, zero_point: float) -> float:
    n = min(len(q_a), len(q_b))
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(n):
        va = (q_a[i] - zero_point) * scale
        vb = (q_b[i] - zero_point) * scale
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb
    na = math.sqrt(norm_a)
    nb = math.sqrt(norm_b)
    if not na or not nb:
        return 0.0
    return dot / (na * nb)
