// Golden mlp_down: one simdgroup per output column, float32 accumulation.
// out[r, j] = x[r, j] + res[r, j] + Wd[j] . act[r]
// Written to be obviously correct, not fast. If it fails a test, suspect the test first.
uint tile = threadgroup_position_in_grid.x;
uint row  = threadgroup_position_in_grid.y;
uint lane = thread_index_in_simdgroup;
uint sg   = simdgroup_index_in_threadgroup;
const int K = act_shape[1];
const int n = dq_shape[0];
const int words = K / 8;
const int groups = K / 64;
const uint n_sg = TG / 32;
auto ar = act + row * K;    // auto: MLX passes small inputs as `constant`, large ones as `device`

for (int c = sg; c < BN; c += n_sg) {
    const int j = int(tile) * BN + c;
    if (j >= n) { break; }
    float acc = 0.0f;
    for (int wd = lane; wd < words; wd += 32) {
        const int k0 = wd * 8;
        const int grp = j * groups + k0 / 64;
        const uint q = dq[j * words + wd];
        const float sv = float(ds[grp]), bv = float(db[grp]);
        for (int t = 0; t < 8; t++) {
            acc += (sv * float((q >> (4 * t)) & 0xFu) + bv) * float(ar[k0 + t]);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) { out[row * n + j] = T(float(x[row * n + j]) + float(res[row * n + j]) + acc); }
}
