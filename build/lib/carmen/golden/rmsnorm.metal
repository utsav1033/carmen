// Golden rmsnorm: one threadgroup per row, TG threads, float32 accumulation.
// Written to be obviously correct, not fast. If it fails a test, suspect the test first.
uint row  = threadgroup_position_in_grid.x;
uint tid  = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg   = simdgroup_index_in_threadgroup;
const int n = x_shape[1];
auto xr = x + row * n;       // auto: MLX passes small inputs as `constant`, large ones as `device`
device T* o = out + row * n;
threadgroup float shared[32];
const uint n_sg = (TG + 31) / 32;
const float eps = 1e-5f;

// 1. sum of squares
float s = 0.0f;
for (int i = tid; i < n; i += TG) { s += float(xr[i]) * float(xr[i]); }
s = simd_sum(s);
if (lane == 0) { shared[sg] = s; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float v = (lane < n_sg) ? shared[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) { shared[0] = v; }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float inv = metal::rsqrt(shared[0] / float(n) + eps);

// 2. write
for (int i = tid; i < n; i += TG) { o[i] = T(float(xr[i]) * inv * float(w[i])); }
