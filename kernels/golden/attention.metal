// Golden causal attention: one threadgroup per block of BQ queries; three passes per query
// (max of the scores, sum of exp, weighted sum of v), recomputing each score from q and k.
// Written to be obviously correct, not fast. If it fails a test, suspect the test first.
uint qb   = threadgroup_position_in_grid.x;
uint tid  = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg   = simdgroup_index_in_threadgroup;
const int Lq = q_shape[0];
const int D  = q_shape[1];
const int Lk = k_shape[0];
const float scl = metal::rsqrt(float(D));
threadgroup float shared[32];
const uint n_sg = (TG + 31) / 32;

#define SCORE(i, j) (dot_qk(q, k, i, j, D) * scl)

for (int qi = 0; qi < BQ; ++qi) {
    const int i = int(qb) * BQ + qi;
    if (i >= Lq) { break; }
    const int last = metal::min(Lk - 1, i + (Lk - Lq));   // causal: keys 0 .. last are visible

    // 1. max score
    float m = -INFINITY;
    for (int j = tid; j <= last; j += TG) { m = metal::max(m, SCORE(i, j)); }
    m = simd_max(m);
    if (lane == 0) { shared[sg] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float x = (lane < n_sg) ? shared[lane] : -INFINITY;
        x = simd_max(x);
        if (lane == 0) { shared[0] = x; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. sum of exp(score - max)
    float l = 0.0f;
    for (int j = tid; j <= last; j += TG) { l += metal::exp(SCORE(i, j) - m); }
    l = simd_sum(l);
    if (lane == 0) { shared[sg] = l; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float x = (lane < n_sg) ? shared[lane] : 0.0f;
        x = simd_sum(x);
        if (lane == 0) { shared[0] = x; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    l = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. out[i, d] = sum_j p[i, j] * v[j, d]
    for (int d = tid; d < D; d += TG) {
        float acc = 0.0f;
        for (int j = 0; j <= last; ++j) { acc += metal::exp(SCORE(i, j) - m) * float(v[j * D + d]); }
        out[i * D + d] = T(acc / l);
    }
}
