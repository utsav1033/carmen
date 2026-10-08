// Golden mlp_up: one simdgroup per output column, float32 accumulation.
// out[r, j] = silu(Wg[j] . hn) * (Wu[j] . hn), with hn = rmsnorm(x[r] + res[r]) * w.
// Written to be obviously correct, not fast. If it fails a test, suspect the test first.
uint tile = threadgroup_position_in_grid.x;
uint row  = threadgroup_position_in_grid.y;
uint tid  = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg   = simdgroup_index_in_threadgroup;
const int K = x_shape[1];
const int n = gq_shape[0];
const int words = K / 8;
const int groups = K / 64;
const uint n_sg = TG / 32;
auto xr = x + row * K;      // auto: MLX passes small inputs as `constant`, large ones as `device`
auto rr = res + row * K;
threadgroup float shared[32];

// 1. rms of h = x + res over the row (every threadgroup redoes it: K is small)
float s = 0.0f;
for (int i = tid; i < K; i += TG) { float h = float(xr[i]) + float(rr[i]); s += h * h; }
s = simd_sum(s);
if (lane == 0) { shared[sg] = s; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float v = (lane < n_sg) ? shared[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) { shared[0] = v; }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float inv = metal::rsqrt(shared[0] / float(K) + eps[0]);

// 2. each simdgroup computes whole output columns; lanes split K, one uint32 word (8 values) at a time
for (int c = sg; c < BN; c += n_sg) {
    const int j = int(tile) * BN + c;
    if (j >= n) { break; }
    float g = 0.0f, u = 0.0f;
    for (int wd = lane; wd < words; wd += 32) {
        const int k0 = wd * 8;
        const int grp = j * groups + k0 / 64;
        const uint qg = gq[j * words + wd];
        const uint qu = uq[j * words + wd];
        const float sgv = float(gs[grp]), bgv = float(gb[grp]);
        const float suv = float(us[grp]), buv = float(ub[grp]);
        for (int t = 0; t < 8; t++) {
            const int k = k0 + t;
            const float hn = (float(xr[k]) + float(rr[k])) * inv * float(w[k]);
            g += (sgv * float((qg >> (4 * t)) & 0xFu) + bgv) * hn;
            u += (suv * float((qu >> (4 * t)) & 0xFu) + buv) * hn;
        }
    }
    g = simd_sum(g);
    u = simd_sum(u);
    if (lane == 0) { out[row * n + j] = T(g / (1.0f + metal::exp(-g)) * u); }
}
