// Golden masked softmax: softmax(x * scale + mask), one threadgroup per row.
// Written to be obviously correct, not fast. It is the judge's positive control:
// if it fails a test, suspect the test first.
uint row  = threadgroup_position_in_grid.x;
uint tid  = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sg   = simdgroup_index_in_threadgroup;
const int n = x_shape[1];
auto xr = x + row * n;       // auto: MLX passes small inputs as `constant`, large ones as `device`
auto mr = mask + row * n;
const float scl = scale[0];
device T* o = out + row * n;
threadgroup float shared[32];
const uint n_sg = (TG + 31) / 32;

// 1. row max
float m = -INFINITY;
for (int i = tid; i < n; i += TG) { m = metal::max(m, (float(xr[i]) * scl + float(mr[i]))); }
m = simd_max(m);
if (lane == 0) { shared[sg] = m; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float v = (lane < n_sg) ? shared[lane] : -INFINITY;
    v = simd_max(v);
    if (lane == 0) { shared[0] = v; }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
m = shared[0];
threadgroup_barrier(mem_flags::mem_threadgroup);

// 2. sum of exp(x - max)
float s = 0.0f;
for (int i = tid; i < n; i += TG) { s += metal::exp((float(xr[i]) * scl + float(mr[i])) - m); }
s = simd_sum(s);
if (lane == 0) { shared[sg] = s; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    float v = (lane < n_sg) ? shared[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) { shared[0] = v; }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
s = shared[0];

// 3. write
for (int i = tid; i < n; i += TG) { o[i] = T(metal::exp((float(xr[i]) * scl + float(mr[i])) - m) / s); }
