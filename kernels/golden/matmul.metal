// Golden matmul: one threadgroup per BM x BN output tile; each thread computes outputs
// of the tile in a strided loop, reading A and B straight from memory.
// Written to be obviously correct, not fast. If it fails a test, suspect the test first.
uint tx  = threadgroup_position_in_grid.x;
uint ty  = threadgroup_position_in_grid.y;
uint tid = thread_position_in_threadgroup.x;
const int M = a_shape[0];
const int K = a_shape[1];
const int N = b_shape[1];

for (int e = tid; e < BM * BN; e += TG) {
    int r = ty * BM + e / BN;
    int c = tx * BN + e % BN;
    if (r < M && c < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) { acc += float(a[r * K + k]) * float(b[k * N + c]); }
        out[r * N + c] = T(acc);
    }
}
