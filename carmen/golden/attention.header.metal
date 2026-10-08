// dot(q[i], k[j]) in float32. Two pointer types: MLX may place q and k in different
// address spaces (small inputs in `constant`, large ones in `device`).
template <typename PQ, typename PK>
inline float dot_qk(PQ q, PK k, int i, int j, int D) {
    float s = 0.0f;
    for (int d = 0; d < D; ++d) { s += float(q[i * D + d]) * float(k[j * D + d]); }
    return s;
}
