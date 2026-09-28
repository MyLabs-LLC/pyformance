"""GPU tests on the NVIDIA CUDA device.

Kernels are PTX loaded through the driver API, so no CUDA toolkit is required.
Each launch stays short of the Windows display timeout, and the timed work is
split across several launches.
"""

from __future__ import annotations

import ctypes
import threading

# Kept under the default 2 second Windows GPU timeout.
_CHUNK_SECONDS = 0.18
_FMA_PER_ITER = 16
_MAD_PER_ITER = 16

_PTX = r"""
.version 7.0
.target sm_70
.address_size 64

.visible .entry fp32_fma(
    .param .u64 iterations,
    .param .u64 sink
)
{
    .reg .pred %p;
    .reg .b32 %r_tid;
    .reg .b32 %r_cta;
    .reg .b32 %r_nt;
    .reg .f32 %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7;
    .reg .f32 %a8, %a9, %a10, %a11, %a12, %a13, %a14, %a15;
    .reg .f32 %s, %k;
    .reg .b64 %rd_i;
    .reg .b64 %rd_n;
    .reg .b64 %rd_ptr;
    .reg .b64 %rd_off;

    ld.param.u64 %rd_n, [iterations];
    ld.param.u64 %rd_ptr, [sink];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mov.u32 %r_nt, %ntid.x;
    mad.lo.u32 %r_tid, %r_cta, %r_nt, %r_tid;

    mov.f32 %s, 0f3F7FF972;
    mov.f32 %k, 0f3A83126F;
    cvt.rn.f32.u32 %a0, %r_tid;
    mov.f32 %a1, %a0;
    mov.f32 %a2, %a0;
    mov.f32 %a3, %a0;
    mov.f32 %a4, %a0;
    mov.f32 %a5, %a0;
    mov.f32 %a6, %a0;
    mov.f32 %a7, %a0;
    mov.f32 %a8, %a0;
    mov.f32 %a9, %a0;
    mov.f32 %a10, %a0;
    mov.f32 %a11, %a0;
    mov.f32 %a12, %a0;
    mov.f32 %a13, %a0;
    mov.f32 %a14, %a0;
    mov.f32 %a15, %a0;
    add.f32 %a0, %a0, 0f3F800000;
    add.f32 %a1, %a1, 0f3F000000;
    add.f32 %a2, %a2, 0f3E800000;
    add.f32 %a3, %a3, 0f3E000000;

    mov.u64 %rd_i, 0;
FP_LOOP:
    fma.rn.f32 %a0, %a0, %s, %k;
    fma.rn.f32 %a1, %a1, %s, %k;
    fma.rn.f32 %a2, %a2, %s, %k;
    fma.rn.f32 %a3, %a3, %s, %k;
    fma.rn.f32 %a4, %a4, %s, %k;
    fma.rn.f32 %a5, %a5, %s, %k;
    fma.rn.f32 %a6, %a6, %s, %k;
    fma.rn.f32 %a7, %a7, %s, %k;
    fma.rn.f32 %a8, %a8, %s, %k;
    fma.rn.f32 %a9, %a9, %s, %k;
    fma.rn.f32 %a10, %a10, %s, %k;
    fma.rn.f32 %a11, %a11, %s, %k;
    fma.rn.f32 %a12, %a12, %s, %k;
    fma.rn.f32 %a13, %a13, %s, %k;
    fma.rn.f32 %a14, %a14, %s, %k;
    fma.rn.f32 %a15, %a15, %s, %k;
    add.u64 %rd_i, %rd_i, 1;
    setp.lt.u64 %p, %rd_i, %rd_n;
    @%p bra FP_LOOP;

    add.f32 %a0, %a0, %a1;
    add.f32 %a0, %a0, %a8;
    add.f32 %a0, %a0, %a15;
    cvt.u64.u32 %rd_off, %r_tid;
    shl.b64 %rd_off, %rd_off, 2;
    add.u64 %rd_ptr, %rd_ptr, %rd_off;
    st.global.f32 [%rd_ptr], %a0;
    ret;
}

.visible .entry int_mad(
    .param .u64 iterations,
    .param .u64 sink
)
{
    .reg .pred %p;
    .reg .b32 %r_tid;
    .reg .b32 %r_cta;
    .reg .b32 %r_nt;
    .reg .b32 %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7;
    .reg .b32 %a8, %a9, %a10, %a11, %a12, %a13, %a14, %a15;
    .reg .b32 %s, %k;
    .reg .b64 %rd_i;
    .reg .b64 %rd_n;
    .reg .b64 %rd_ptr;
    .reg .b64 %rd_off;

    ld.param.u64 %rd_n, [iterations];
    ld.param.u64 %rd_ptr, [sink];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mov.u32 %r_nt, %ntid.x;
    mad.lo.u32 %r_tid, %r_cta, %r_nt, %r_tid;

    mov.u32 %s, 1664525;
    mov.u32 %k, 1013904223;
    mov.u32 %a0, %r_tid;
    add.u32 %a1, %a0, 1;
    add.u32 %a2, %a0, 2;
    add.u32 %a3, %a0, 3;
    add.u32 %a4, %a0, 4;
    add.u32 %a5, %a0, 5;
    add.u32 %a6, %a0, 6;
    add.u32 %a7, %a0, 7;
    add.u32 %a8, %a0, 8;
    add.u32 %a9, %a0, 9;
    add.u32 %a10, %a0, 10;
    add.u32 %a11, %a0, 11;
    add.u32 %a12, %a0, 12;
    add.u32 %a13, %a0, 13;
    add.u32 %a14, %a0, 14;
    add.u32 %a15, %a0, 15;

    mov.u64 %rd_i, 0;
INT_LOOP:
    mad.lo.u32 %a0, %a0, %s, %k;
    mad.lo.u32 %a1, %a1, %s, %k;
    mad.lo.u32 %a2, %a2, %s, %k;
    mad.lo.u32 %a3, %a3, %s, %k;
    mad.lo.u32 %a4, %a4, %s, %k;
    mad.lo.u32 %a5, %a5, %s, %k;
    mad.lo.u32 %a6, %a6, %s, %k;
    mad.lo.u32 %a7, %a7, %s, %k;
    mad.lo.u32 %a8, %a8, %s, %k;
    mad.lo.u32 %a9, %a9, %s, %k;
    mad.lo.u32 %a10, %a10, %s, %k;
    mad.lo.u32 %a11, %a11, %s, %k;
    mad.lo.u32 %a12, %a12, %s, %k;
    mad.lo.u32 %a13, %a13, %s, %k;
    mad.lo.u32 %a14, %a14, %s, %k;
    mad.lo.u32 %a15, %a15, %s, %k;
    add.u64 %rd_i, %rd_i, 1;
    setp.lt.u64 %p, %rd_i, %rd_n;
    @%p bra INT_LOOP;

    xor.b32 %a0, %a0, %a1;
    xor.b32 %a0, %a0, %a15;
    cvt.u64.u32 %rd_off, %r_tid;
    shl.b64 %rd_off, %rd_off, 2;
    add.u64 %rd_ptr, %rd_ptr, %rd_off;
    st.global.u32 [%rd_ptr], %a0;
    ret;
}

.visible .entry copy4(
    .param .u64 dst,
    .param .u64 src,
    .param .u64 count4,
    .param .u64 reps
)
{
    .reg .pred %p;
    .reg .b32 %r_tid;
    .reg .b32 %r_cta;
    .reg .b32 %r_nt;
    .reg .b32 %r_grid;
    .reg .b32 %r_stride;
    .reg .b64 %rd_i;
    .reg .b64 %rd_n;
    .reg .b64 %rd_stride;
    .reg .b64 %rd_src;
    .reg .b64 %rd_dst;
    .reg .b64 %rd_off;
    .reg .b64 %rd_s;
    .reg .b64 %rd_d;
    .reg .b64 %rd_reps;
    .reg .b64 %rd_rep;
    .reg .f32 %f0;
    .reg .f32 %f1;
    .reg .f32 %f2;
    .reg .f32 %f3;

    ld.param.u64 %rd_dst, [dst];
    ld.param.u64 %rd_src, [src];
    ld.param.u64 %rd_n, [count4];
    ld.param.u64 %rd_reps, [reps];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mov.u32 %r_nt, %ntid.x;
    mad.lo.u32 %r_tid, %r_cta, %r_nt, %r_tid;
    mov.u32 %r_grid, %nctaid.x;
    mul.lo.u32 %r_stride, %r_grid, %r_nt;
    cvt.u64.u32 %rd_stride, %r_stride;

    mov.u64 %rd_rep, 0;
COPY_REP:
    cvt.u64.u32 %rd_i, %r_tid;
COPY_STEP:
    setp.ge.u64 %p, %rd_i, %rd_n;
    @%p bra COPY_NEXT;
    shl.b64 %rd_off, %rd_i, 4;
    add.u64 %rd_s, %rd_src, %rd_off;
    add.u64 %rd_d, %rd_dst, %rd_off;
    ld.global.cg.v4.f32 {%f0, %f1, %f2, %f3}, [%rd_s];
    st.global.cg.v4.f32 [%rd_d], {%f0, %f1, %f2, %f3};
    add.u64 %rd_i, %rd_i, %rd_stride;
    bra COPY_STEP;
COPY_NEXT:
    add.u64 %rd_rep, %rd_rep, 1;
    setp.lt.u64 %p, %rd_rep, %rd_reps;
    @%p bra COPY_REP;
    ret;
}

.visible .entry fp16_fma(
    .param .u64 iterations,
    .param .u64 sink
)
{
    .reg .pred %p;
    .reg .b32 %r_tid, %r_cta, %r_nt, %tmp, %bits;
    .reg .f16x2 %h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7;
    .reg .f16x2 %h8, %h9, %h10, %h11, %h12, %h13, %h14, %h15;
    .reg .f16x2 %hs, %hk;
    .reg .b64 %rd_i, %rd_n, %rd_ptr, %rd_off;

    ld.param.u64 %rd_n, [iterations];
    ld.param.u64 %rd_ptr, [sink];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mov.u32 %r_nt, %ntid.x;
    mad.lo.u32 %r_tid, %r_cta, %r_nt, %r_tid;

    mov.b32 %tmp, 0x3BFF3BFF;
    mov.b32 %hs, %tmp;
    mov.b32 %tmp, 0x14001400;
    mov.b32 %hk, %tmp;
    mov.b32 %tmp, 0x3C003C00;
    mov.b32 %h0, %tmp;
    mov.b32 %h1, %tmp;
    mov.b32 %h2, %tmp;
    mov.b32 %h3, %tmp;
    mov.b32 %h4, %tmp;
    mov.b32 %h5, %tmp;
    mov.b32 %h6, %tmp;
    mov.b32 %h7, %tmp;
    mov.b32 %h8, %tmp;
    mov.b32 %h9, %tmp;
    mov.b32 %h10, %tmp;
    mov.b32 %h11, %tmp;
    mov.b32 %h12, %tmp;
    mov.b32 %h13, %tmp;
    mov.b32 %h14, %tmp;
    mov.b32 %h15, %tmp;

    mov.u64 %rd_i, 0;
FP16_LOOP:
    fma.rn.f16x2 %h0, %h1, %h2, %h0;
    fma.rn.f16x2 %h1, %h2, %h3, %h1;
    fma.rn.f16x2 %h2, %h3, %h4, %h2;
    fma.rn.f16x2 %h3, %h4, %h5, %h3;
    fma.rn.f16x2 %h4, %h5, %h6, %h4;
    fma.rn.f16x2 %h5, %h6, %h7, %h5;
    fma.rn.f16x2 %h6, %h7, %h8, %h6;
    fma.rn.f16x2 %h7, %h8, %h9, %h7;
    fma.rn.f16x2 %h8, %h9, %h10, %h8;
    fma.rn.f16x2 %h9, %h10, %h11, %h9;
    fma.rn.f16x2 %h10, %h11, %h12, %h10;
    fma.rn.f16x2 %h11, %h12, %h13, %h11;
    fma.rn.f16x2 %h12, %h13, %h14, %h12;
    fma.rn.f16x2 %h13, %h14, %h15, %h13;
    fma.rn.f16x2 %h14, %h15, %h0, %h14;
    fma.rn.f16x2 %h15, %h0, %h1, %h15;
    add.u64 %rd_i, %rd_i, 1;
    setp.lt.u64 %p, %rd_i, %rd_n;
    @%p bra FP16_LOOP;

    mov.b32 %bits, %h0;
    cvt.u64.u32 %rd_off, %r_tid;
    shl.b64 %rd_off, %rd_off, 2;
    add.u64 %rd_ptr, %rd_ptr, %rd_off;
    st.global.u32 [%rd_ptr], %bits;
    ret;
}

.visible .entry fp64_fma(
    .param .u64 iterations,
    .param .u64 sink
)
{
    .reg .pred %p;
    .reg .b32 %r_tid, %r_cta, %r_nt;
    .reg .f32 %fs, %fk, %out;
    .reg .f64 %d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7, %ds, %dk;
    .reg .b64 %rd_i, %rd_n, %rd_ptr, %rd_off;

    ld.param.u64 %rd_n, [iterations];
    ld.param.u64 %rd_ptr, [sink];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mov.u32 %r_nt, %ntid.x;
    mad.lo.u32 %r_tid, %r_cta, %r_nt, %r_tid;

    mov.f32 %fs, 0f3F7FF972;
    mov.f32 %fk, 0f3A83126F;
    cvt.f64.f32 %ds, %fs;
    cvt.f64.f32 %dk, %fk;
    cvt.rn.f64.u32 %d0, %r_tid;
    add.f64 %d0, %d0, %ds;
    mov.f64 %d1, %d0;
    mov.f64 %d2, %d0;
    mov.f64 %d3, %d0;
    mov.f64 %d4, %d0;
    mov.f64 %d5, %d0;
    mov.f64 %d6, %d0;
    mov.f64 %d7, %d0;

    mov.u64 %rd_i, 0;
FP64_LOOP:
    fma.rn.f64 %d0, %d1, %d2, %d0;
    fma.rn.f64 %d1, %d2, %d3, %d1;
    fma.rn.f64 %d2, %d3, %d4, %d2;
    fma.rn.f64 %d3, %d4, %d5, %d3;
    fma.rn.f64 %d4, %d5, %d6, %d4;
    fma.rn.f64 %d5, %d6, %d7, %d5;
    fma.rn.f64 %d6, %d7, %d0, %d6;
    fma.rn.f64 %d7, %d0, %d1, %d7;
    add.u64 %rd_i, %rd_i, 1;
    setp.lt.u64 %p, %rd_i, %rd_n;
    @%p bra FP64_LOOP;

    cvt.rn.f32.f64 %out, %d0;
    cvt.u64.u32 %rd_off, %r_tid;
    shl.b64 %rd_off, %rd_off, 2;
    add.u64 %rd_ptr, %rd_ptr, %rd_off;
    st.global.f32 [%rd_ptr], %out;
    ret;
}

.visible .entry sfu_sqrt(
    .param .u64 iterations,
    .param .u64 sink
)
{
    .reg .pred %p;
    .reg .b32 %r_tid, %r_cta, %r_nt;
    .reg .f32 %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7;
    .reg .f32 %a8, %a9, %a10, %a11, %a12, %a13, %a14, %a15, %k;
    .reg .b64 %rd_i, %rd_n, %rd_ptr, %rd_off;

    ld.param.u64 %rd_n, [iterations];
    ld.param.u64 %rd_ptr, [sink];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mov.u32 %r_nt, %ntid.x;
    mad.lo.u32 %r_tid, %r_cta, %r_nt, %r_tid;
    mov.f32 %k, 0f3A83126F;
    cvt.rn.f32.u32 %a0, %r_tid;
    mul.f32 %a0, %a0, 0f2F800000;
    add.f32 %a0, %a0, 0f3F800000;
    mov.f32 %a1, %a0;
    mov.f32 %a2, %a0;
    mov.f32 %a3, %a0;
    mov.f32 %a4, %a0;
    mov.f32 %a5, %a0;
    mov.f32 %a6, %a0;
    mov.f32 %a7, %a0;
    mov.f32 %a8, %a0;
    mov.f32 %a9, %a0;
    mov.f32 %a10, %a0;
    mov.f32 %a11, %a0;
    mov.f32 %a12, %a0;
    mov.f32 %a13, %a0;
    mov.f32 %a14, %a0;
    mov.f32 %a15, %a0;

    mov.u64 %rd_i, 0;
SFU_LOOP:
    sqrt.approx.ftz.f32 %a0, %a0;
    add.f32 %a0, %a0, %k;
    sqrt.approx.ftz.f32 %a1, %a1;
    add.f32 %a1, %a1, %k;
    sqrt.approx.ftz.f32 %a2, %a2;
    add.f32 %a2, %a2, %k;
    sqrt.approx.ftz.f32 %a3, %a3;
    add.f32 %a3, %a3, %k;
    sqrt.approx.ftz.f32 %a4, %a4;
    add.f32 %a4, %a4, %k;
    sqrt.approx.ftz.f32 %a5, %a5;
    add.f32 %a5, %a5, %k;
    sqrt.approx.ftz.f32 %a6, %a6;
    add.f32 %a6, %a6, %k;
    sqrt.approx.ftz.f32 %a7, %a7;
    add.f32 %a7, %a7, %k;
    sqrt.approx.ftz.f32 %a8, %a8;
    add.f32 %a8, %a8, %k;
    sqrt.approx.ftz.f32 %a9, %a9;
    add.f32 %a9, %a9, %k;
    sqrt.approx.ftz.f32 %a10, %a10;
    add.f32 %a10, %a10, %k;
    sqrt.approx.ftz.f32 %a11, %a11;
    add.f32 %a11, %a11, %k;
    sqrt.approx.ftz.f32 %a12, %a12;
    add.f32 %a12, %a12, %k;
    sqrt.approx.ftz.f32 %a13, %a13;
    add.f32 %a13, %a13, %k;
    sqrt.approx.ftz.f32 %a14, %a14;
    add.f32 %a14, %a14, %k;
    sqrt.approx.ftz.f32 %a15, %a15;
    add.f32 %a15, %a15, %k;
    add.u64 %rd_i, %rd_i, 1;
    setp.lt.u64 %p, %rd_i, %rd_n;
    @%p bra SFU_LOOP;

    add.f32 %a0, %a0, %a15;
    cvt.u64.u32 %rd_off, %r_tid;
    shl.b64 %rd_off, %rd_off, 2;
    add.u64 %rd_ptr, %rd_ptr, %rd_off;
    st.global.f32 [%rd_ptr], %a0;
    ret;
}

.visible .entry int64_mad(
    .param .u64 iterations,
    .param .u64 sink
)
{
    .reg .pred %p;
    .reg .b32 %r_tid, %r_cta, %r_nt, %lo;
    .reg .b64 %d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7, %t;
    .reg .b64 %rd_i, %rd_n, %rd_ptr, %rd_off;

    ld.param.u64 %rd_n, [iterations];
    ld.param.u64 %rd_ptr, [sink];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mov.u32 %r_nt, %ntid.x;
    mad.lo.u32 %r_tid, %r_cta, %r_nt, %r_tid;
    cvt.u64.u32 %d0, %r_tid;
    add.u64 %d0, %d0, 1;
    mov.u64 %d1, %d0;
    add.u64 %d1, %d1, 3;
    mov.u64 %d2, %d0;
    add.u64 %d2, %d2, 5;
    mov.u64 %d3, %d0;
    add.u64 %d3, %d3, 7;
    mov.u64 %d4, %d0;
    add.u64 %d4, %d4, 9;
    mov.u64 %d5, %d0;
    add.u64 %d5, %d5, 11;
    mov.u64 %d6, %d0;
    add.u64 %d6, %d6, 13;
    mov.u64 %d7, %d0;
    add.u64 %d7, %d7, 15;

    mov.u64 %rd_i, 0;
I64_LOOP:
    mul.lo.u64 %t, %d0, %d1;
    add.u64 %d0, %t, %d2;
    mul.lo.u64 %t, %d1, %d2;
    add.u64 %d1, %t, %d3;
    mul.lo.u64 %t, %d2, %d3;
    add.u64 %d2, %t, %d4;
    mul.lo.u64 %t, %d3, %d4;
    add.u64 %d3, %t, %d5;
    mul.lo.u64 %t, %d4, %d5;
    add.u64 %d4, %t, %d6;
    mul.lo.u64 %t, %d5, %d6;
    add.u64 %d5, %t, %d7;
    mul.lo.u64 %t, %d6, %d7;
    add.u64 %d6, %t, %d0;
    mul.lo.u64 %t, %d7, %d0;
    add.u64 %d7, %t, %d1;
    add.u64 %rd_i, %rd_i, 1;
    setp.lt.u64 %p, %rd_i, %rd_n;
    @%p bra I64_LOOP;

    cvt.u32.u64 %lo, %d0;
    cvt.u64.u32 %rd_off, %r_tid;
    shl.b64 %rd_off, %rd_off, 2;
    add.u64 %rd_ptr, %rd_ptr, %rd_off;
    st.global.u32 [%rd_ptr], %lo;
    ret;
}

.visible .entry bit_ops(
    .param .u64 iterations,
    .param .u64 sink
)
{
    .reg .pred %p;
    .reg .b32 %r_tid, %r_cta, %r_nt, %s, %k;
    .reg .b32 %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7;
    .reg .b32 %a8, %a9, %a10, %a11, %a12, %a13, %a14, %a15;
    .reg .b64 %rd_i, %rd_n, %rd_ptr, %rd_off;

    ld.param.u64 %rd_n, [iterations];
    ld.param.u64 %rd_ptr, [sink];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mov.u32 %r_nt, %ntid.x;
    mad.lo.u32 %r_tid, %r_cta, %r_nt, %r_tid;
    mov.u32 %s, 0xA5A5A5A5;
    mov.u32 %k, 0x01010101;
    or.b32 %a0, %r_tid, 1;
    add.u32 %a1, %a0, 1;
    add.u32 %a2, %a0, 2;
    add.u32 %a3, %a0, 3;
    add.u32 %a4, %a0, 4;
    add.u32 %a5, %a0, 5;
    add.u32 %a6, %a0, 6;
    add.u32 %a7, %a0, 7;
    add.u32 %a8, %a0, 8;
    add.u32 %a9, %a0, 9;
    add.u32 %a10, %a0, 10;
    add.u32 %a11, %a0, 11;
    add.u32 %a12, %a0, 12;
    add.u32 %a13, %a0, 13;
    add.u32 %a14, %a0, 14;
    add.u32 %a15, %a0, 15;

    mov.u64 %rd_i, 0;
BIT_LOOP:
    xor.b32 %a0, %a0, %s;
    add.u32 %a0, %a0, %k;
    xor.b32 %a1, %a1, %s;
    add.u32 %a1, %a1, %k;
    xor.b32 %a2, %a2, %s;
    add.u32 %a2, %a2, %k;
    xor.b32 %a3, %a3, %s;
    add.u32 %a3, %a3, %k;
    xor.b32 %a4, %a4, %s;
    add.u32 %a4, %a4, %k;
    xor.b32 %a5, %a5, %s;
    add.u32 %a5, %a5, %k;
    xor.b32 %a6, %a6, %s;
    add.u32 %a6, %a6, %k;
    xor.b32 %a7, %a7, %s;
    add.u32 %a7, %a7, %k;
    xor.b32 %a8, %a8, %s;
    add.u32 %a8, %a8, %k;
    xor.b32 %a9, %a9, %s;
    add.u32 %a9, %a9, %k;
    xor.b32 %a10, %a10, %s;
    add.u32 %a10, %a10, %k;
    xor.b32 %a11, %a11, %s;
    add.u32 %a11, %a11, %k;
    xor.b32 %a12, %a12, %s;
    add.u32 %a12, %a12, %k;
    xor.b32 %a13, %a13, %s;
    add.u32 %a13, %a13, %k;
    xor.b32 %a14, %a14, %s;
    add.u32 %a14, %a14, %k;
    xor.b32 %a15, %a15, %s;
    add.u32 %a15, %a15, %k;
    add.u64 %rd_i, %rd_i, 1;
    setp.lt.u64 %p, %rd_i, %rd_n;
    @%p bra BIT_LOOP;

    xor.b32 %a0, %a0, %a15;
    cvt.u64.u32 %rd_off, %r_tid;
    shl.b64 %rd_off, %rd_off, 2;
    add.u64 %rd_ptr, %rd_ptr, %rd_off;
    st.global.u32 [%rd_ptr], %a0;
    ret;
}

.visible .entry read4(
    .param .u64 ptr,
    .param .u64 count4,
    .param .u64 reps,
    .param .u64 sink
)
{
    .reg .pred %p;
    .reg .b32 %r_tid, %r_cta, %r_nt, %r_grid, %r_stride, %acc;
    .reg .b32 %b0, %b1, %b2, %b3;
    .reg .f32 %f0, %f1, %f2, %f3;
    .reg .b64 %rd_i, %rd_n, %rd_stride, %rd_ptr, %rd_off, %rd_s;
    .reg .b64 %rd_reps, %rd_rep, %rd_sink;

    ld.param.u64 %rd_ptr, [ptr];
    ld.param.u64 %rd_n, [count4];
    ld.param.u64 %rd_reps, [reps];
    ld.param.u64 %rd_sink, [sink];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mov.u32 %r_nt, %ntid.x;
    mad.lo.u32 %r_tid, %r_cta, %r_nt, %r_tid;
    mov.u32 %r_grid, %nctaid.x;
    mul.lo.u32 %r_stride, %r_grid, %r_nt;
    cvt.u64.u32 %rd_stride, %r_stride;
    mov.u32 %acc, 0;
    mov.u64 %rd_rep, 0;
READ_REP:
    cvt.u64.u32 %rd_i, %r_tid;
READ_STEP:
    setp.ge.u64 %p, %rd_i, %rd_n;
    @%p bra READ_NEXT;
    shl.b64 %rd_off, %rd_i, 4;
    add.u64 %rd_s, %rd_ptr, %rd_off;
    ld.global.cg.v4.f32 {%f0, %f1, %f2, %f3}, [%rd_s];
    mov.b32 %b0, %f0;
    xor.b32 %acc, %acc, %b0;
    mov.b32 %b1, %f1;
    xor.b32 %acc, %acc, %b1;
    mov.b32 %b2, %f2;
    xor.b32 %acc, %acc, %b2;
    mov.b32 %b3, %f3;
    xor.b32 %acc, %acc, %b3;
    add.u64 %rd_i, %rd_i, %rd_stride;
    bra READ_STEP;
READ_NEXT:
    add.u64 %rd_rep, %rd_rep, 1;
    setp.lt.u64 %p, %rd_rep, %rd_reps;
    @%p bra READ_REP;
    cvt.u64.u32 %rd_off, %r_tid;
    shl.b64 %rd_off, %rd_off, 2;
    add.u64 %rd_sink, %rd_sink, %rd_off;
    st.global.u32 [%rd_sink], %acc;
    ret;
}

.visible .entry write4(
    .param .u64 ptr,
    .param .u64 count4,
    .param .u64 reps
)
{
    .reg .pred %p;
    .reg .b32 %r_tid, %r_cta, %r_nt, %r_grid, %r_stride;
    .reg .f32 %c;
    .reg .b64 %rd_i, %rd_n, %rd_stride, %rd_ptr, %rd_off, %rd_d;
    .reg .b64 %rd_reps, %rd_rep;

    ld.param.u64 %rd_ptr, [ptr];
    ld.param.u64 %rd_n, [count4];
    ld.param.u64 %rd_reps, [reps];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mov.u32 %r_nt, %ntid.x;
    mad.lo.u32 %r_tid, %r_cta, %r_nt, %r_tid;
    mov.u32 %r_grid, %nctaid.x;
    mul.lo.u32 %r_stride, %r_grid, %r_nt;
    cvt.u64.u32 %rd_stride, %r_stride;
    mov.f32 %c, 0f3F800000;
    mov.u64 %rd_rep, 0;
WRITE_REP:
    cvt.u64.u32 %rd_i, %r_tid;
WRITE_STEP:
    setp.ge.u64 %p, %rd_i, %rd_n;
    @%p bra WRITE_NEXT;
    shl.b64 %rd_off, %rd_i, 4;
    add.u64 %rd_d, %rd_ptr, %rd_off;
    st.global.cg.v4.f32 [%rd_d], {%c, %c, %c, %c};
    add.u64 %rd_i, %rd_i, %rd_stride;
    bra WRITE_STEP;
WRITE_NEXT:
    add.u64 %rd_rep, %rd_rep, 1;
    setp.lt.u64 %p, %rd_rep, %rd_reps;
    @%p bra WRITE_REP;
    ret;
}

.visible .entry shared_bw(
    .param .u64 iterations,
    .param .u64 sink
)
{
    .reg .pred %p;
    .reg .b32 %r_tid, %r_cta, %r_nt, %r_lane, %v, %s;
    .reg .b64 %rd_i, %rd_n, %rd_ptr, %rd_off, %rd_addr, %rd_base;
    .shared .align 4 .b32 smem[256];

    ld.param.u64 %rd_n, [iterations];
    ld.param.u64 %rd_ptr, [sink];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mov.u32 %r_nt, %ntid.x;
    mad.lo.u32 %r_lane, %r_cta, %r_nt, %r_tid;
    mov.u32 %r_tid, %tid.x;
    shl.b32 %r_lane, %r_tid, 2;
    cvt.u64.u32 %rd_off, %r_lane;
    cvta.shared.u64 %rd_base, smem;
    add.u64 %rd_addr, %rd_base, %rd_off;
    mov.u32 %v, %r_tid;
    mov.u32 %s, 0x01010101;
    st.u32 [%rd_addr], %v;

    mov.u64 %rd_i, 0;
SH_LOOP:
    ld.u32 %v, [%rd_addr];
    xor.b32 %v, %v, %s;
    st.u32 [%rd_addr], %v;
    add.u64 %rd_i, %rd_i, 1;
    setp.lt.u64 %p, %rd_i, %rd_n;
    @%p bra SH_LOOP;

    cvt.u64.u32 %rd_off, %r_lane;
    shl.b64 %rd_off, %rd_off, 2;
    add.u64 %rd_ptr, %rd_ptr, %rd_off;
    st.global.u32 [%rd_ptr], %v;
    ret;
}

.visible .entry stride_load(
    .param .u64 iterations,
    .param .u64 ptr,
    .param .u64 mask
)
{
    .reg .pred %p;
    .reg .b32 %r_tid, %r_cta, %r_nt, %v, %acc;
    .reg .b64 %rd_i, %rd_n, %rd_ptr, %rd_mask, %rd_idx, %rd_off, %rd_addr, %rd_base;

    ld.param.u64 %rd_n, [iterations];
    ld.param.u64 %rd_ptr, [ptr];
    ld.param.u64 %rd_mask, [mask];
    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mov.u32 %r_nt, %ntid.x;
    mad.lo.u32 %r_tid, %r_cta, %r_nt, %r_tid;
    mul.wide.u32 %rd_base, %r_tid, 131071;
    mov.u32 %acc, 0;
    mov.u64 %rd_i, 0;
STR_LOOP:
    mul.lo.u64 %rd_idx, %rd_i, 8192;
    add.u64 %rd_idx, %rd_idx, %rd_base;
    and.b64 %rd_idx, %rd_idx, %rd_mask;
    shl.b64 %rd_off, %rd_idx, 2;
    add.u64 %rd_addr, %rd_ptr, %rd_off;
    ld.global.cg.u32 %v, [%rd_addr];
    xor.b32 %acc, %acc, %v;
    add.u64 %rd_i, %rd_i, 1;
    setp.lt.u64 %p, %rd_i, %rd_n;
    @%p bra STR_LOOP;
    cvt.u64.u32 %rd_off, %r_tid;
    shl.b64 %rd_off, %rd_off, 2;
    add.u64 %rd_ptr, %rd_ptr, %rd_off;
    st.global.u32 [%rd_ptr], %acc;
    ret;
}
"""


class GpuError(RuntimeError):
    pass


def describe_gpu() -> str | None:
    try:
        session = CudaSession()
        try:
            return session.name
        finally:
            session.close()
    except Exception:
        return None


def verify_gpu() -> str:
    with CudaSession() as gpu:
        gpu.load()
        gpu.check_copy(1 << 20)
        gpu.check_compute(gpu.fp32, 64)
        return gpu.name


def measure_gpu(
    seconds: float,
    quick: bool,
    stop: threading.Event | None = None,
    on_phase=None,
    only: set[str] | None = None,
    on_measured=None,
) -> dict[str, tuple[float | None, str, str, float]]:
    import time

    results: dict[str, tuple[float | None, str, str, float]] = {}
    with CudaSession() as gpu:
        gpu.load()
        detail = gpu.name
        phases = (
            ("gpu_fp32", "GFLOPS", lambda: gpu.fp32_gflops(seconds)),
            ("gpu_fp16", "GFLOPS", lambda: gpu.fp16_gflops(seconds)),
            ("gpu_fp64", "GFLOPS", lambda: gpu.fp64_gflops(seconds)),
            ("gpu_sfu", "Gops/s", lambda: gpu.sfu_gops(seconds)),
            ("gpu_int", "Gops/s", lambda: gpu.integer_gops(seconds)),
            ("gpu_int64", "Gops/s", lambda: gpu.int64_gops(seconds)),
            ("gpu_bit", "Gops/s", lambda: gpu.bit_gops(seconds)),
            ("gpu_bw", "GB/s", lambda: gpu.bandwidth_gbs(seconds, quick)),
            ("gpu_read", "GB/s", lambda: gpu.read_gbs(seconds, quick)),
            ("gpu_write", "GB/s", lambda: gpu.write_gbs(seconds, quick)),
            ("gpu_l2", "GB/s", lambda: gpu.l2_gbs(seconds)),
            ("gpu_shared", "GB/s", lambda: gpu.shared_gbs(seconds)),
            ("gpu_stride", "Mlookups/s", lambda: gpu.stride_mlookups(seconds, quick)),
        )
        for key, unit, run in phases:
            if only is not None and key not in only:
                continue
            if stop is not None and stop.is_set():
                break
            if on_phase is not None:
                on_phase(key)
            started = time.perf_counter()
            try:
                value = run()
                payload = (value, unit, detail, time.perf_counter() - started)
            except GpuError as exc:
                payload = (None, unit, str(exc), time.perf_counter() - started)
            results[key] = payload
            if on_measured is not None:
                on_measured(key, payload)
    return results


class CudaSession:
    def __init__(self) -> None:
        try:
            self.lib = ctypes.WinDLL("nvcuda.dll")
        except OSError as exc:
            raise GpuError("No NVIDIA CUDA driver was found") from exc
        self._bind()
        self._check(self.lib.cuInit(0), "cuInit")
        self.device = self._pick_device()
        self.name = self._device_name(self.device)
        self.sms = self._attribute(16)
        self.block = 256
        self.grid = max(1, self.sms) * 8
        self.threads = self.grid * self.block
        self.ctx = ctypes.c_void_p()
        self.module = ctypes.c_void_p()
        self.fp32 = ctypes.c_void_p()
        self.int_fn = ctypes.c_void_p()
        self.copy_fn = ctypes.c_void_p()
        self.start_ev = ctypes.c_void_p()
        self.end_ev = ctypes.c_void_p()
        self._memory: list[int] = []
        self._ready = False
        self._keepalive: tuple | None = None

    def __enter__(self) -> CudaSession:
        ctx = ctypes.c_void_p()
        self._check(self.lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, self.device), "cuCtxCreate")
        self.ctx = ctx
        start = ctypes.c_void_p()
        end = ctypes.c_void_p()
        self._check(self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate")
        self._check(self.lib.cuEventCreate(ctypes.byref(end), 0), "cuEventCreate")
        self.start_ev = start
        self.end_ev = end
        self._ready = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not self._ready and not self.ctx:
            return
        for ptr in self._memory:
            self.lib.cuMemFree_v2(ptr)
        self._memory.clear()
        if self.module:
            self.lib.cuModuleUnload(self.module)
            self.module = ctypes.c_void_p()
        if self.start_ev:
            self.lib.cuEventDestroy_v2(self.start_ev)
            self.start_ev = ctypes.c_void_p()
        if self.end_ev:
            self.lib.cuEventDestroy_v2(self.end_ev)
            self.end_ev = ctypes.c_void_p()
        if self.ctx:
            self.lib.cuCtxDestroy_v2(self.ctx)
            self.ctx = ctypes.c_void_p()
        self._ready = False

    def load(self) -> None:
        image = ctypes.create_string_buffer(_PTX.encode("ascii"))
        log = ctypes.create_string_buffer(16384)
        options = (ctypes.c_uint * 2)(5, 6)
        values = (ctypes.c_void_p * 2)(
            ctypes.cast(log, ctypes.c_void_p),
            ctypes.c_void_p(len(log)),
        )
        module = ctypes.c_void_p()
        code = self.lib.cuModuleLoadDataEx(ctypes.byref(module), image, 2, options, values)
        if code != 0:
            detail = log.value.decode("utf-8", "replace").strip()
            suffix = f": {detail}" if detail else ""
            self._check(code, f"load GPU kernels{suffix}")
        self.module = module
        self.fp32 = self._function(b"fp32_fma")
        self.fp16 = self._function(b"fp16_fma")
        self.fp64 = self._function(b"fp64_fma")
        self.sfu = self._function(b"sfu_sqrt")
        self.int_fn = self._function(b"int_mad")
        self.int64 = self._function(b"int64_mad")
        self.bit_fn = self._function(b"bit_ops")
        self.copy_fn = self._function(b"copy4")
        self.read_fn = self._function(b"read4")
        self.write_fn = self._function(b"write4")
        self.shared_fn = self._function(b"shared_bw")
        self.stride_fn = self._function(b"stride_load")

    def check_copy(self, nbytes: int) -> None:
        src, dst = self._copy_buffers(nbytes)
        self._launch_copy(src, dst, nbytes, 1)
        self._check(self.lib.cuCtxSynchronize(), "sync copy")
        if self._read(src, 64) != self._read(dst, 64):
            raise GpuError("GPU copy did not return the source bytes")

    def check_compute(self, fn: ctypes.c_void_p, iterations: int) -> None:
        sink = self._alloc(self.threads * 4)
        self._check(self.lib.cuMemsetD8_v2(sink, 0, self.threads * 4), "clear sink")
        self._launch_compute(fn, iterations, sink)
        self._check(self.lib.cuCtxSynchronize(), "sync compute")
        if self._read(sink, 256) == bytes(256):
            raise GpuError("GPU compute kernel did not write a result")

    def fp32_gflops(self, target: float) -> float:
        return self._alu(self.fp32, _FMA_PER_ITER * 2, 8_192, target, "FP32") / 1e9

    def fp16_gflops(self, target: float) -> float:
        return self._alu(self.fp16, 64, 4_096, target, "FP16") / 1e9

    def fp64_gflops(self, target: float) -> float:
        return self._alu(self.fp64, 16, 1_024, target, "FP64") / 1e9

    def sfu_gops(self, target: float) -> float:
        return self._alu(self.sfu, 16, 2_048, target, "special functions") / 1e9

    def integer_gops(self, target: float) -> float:
        return self._alu(self.int_fn, _MAD_PER_ITER * 2, 8_192, target, "integer") / 1e9

    def int64_gops(self, target: float) -> float:
        return self._alu(self.int64, 16, 1_024, target, "64-bit integer") / 1e9

    def bit_gops(self, target: float) -> float:
        return self._alu(self.bit_fn, 32, 8_192, target, "bitwise") / 1e9

    def shared_gbs(self, target: float) -> float:
        sink = self._alloc(self.threads * 4)

        def launch(iterations: int) -> None:
            self._launch_compute(self.shared_fn, iterations, sink)

        def traffic(iterations: int) -> float:
            return float(iterations) * 8 * self.threads

        rate, _elapsed = self._scaled(launch, 4_096, traffic, target, "shared memory")
        return rate / 1e9

    def bandwidth_gbs(self, target: float, quick: bool) -> float:
        nbytes = (128 if quick else 512) << 20
        src, dst = self._copy_buffers(nbytes)
        self._launch_copy(src, dst, nbytes, 1)
        self._check(self.lib.cuCtxSynchronize(), "sync bandwidth warmup")

        def launch(reps: int) -> None:
            self._launch_copy(src, dst, nbytes, reps)

        def traffic(reps: int) -> float:
            return float(reps) * nbytes * 2

        rate, _elapsed = self._scaled(launch, 1, traffic, target, "bandwidth", warmup=False)
        if self._read(src, 64) != self._read(dst, 64):
            raise GpuError("GPU memory test did not copy the buffer")
        mid = nbytes // 2
        if self._read(src, 64, mid) != self._read(dst, 64, mid):
            raise GpuError("GPU memory test did not copy the middle of the buffer")
        return rate / 1e9

    def read_gbs(self, target: float, quick: bool) -> float:
        nbytes = (128 if quick else 512) << 20
        src = self._alloc(nbytes)
        sink = self._alloc(self.threads * 4)
        self._check(self.lib.cuMemsetD8_v2(src, 0xA5, nbytes), "fill source")

        def launch(reps: int) -> None:
            self._launch_read(src, nbytes, reps, sink)

        def traffic(reps: int) -> float:
            return float(reps) * nbytes

        self._launch_read(src, nbytes, 1, sink)
        self._check(self.lib.cuCtxSynchronize(), "sync read warmup")
        rate, _elapsed = self._scaled(launch, 1, traffic, target, "read", warmup=False)
        return rate / 1e9

    def write_gbs(self, target: float, quick: bool) -> float:
        nbytes = (128 if quick else 512) << 20
        dst = self._alloc(nbytes)

        def launch(reps: int) -> None:
            self._launch_write(dst, nbytes, reps)

        def traffic(reps: int) -> float:
            return float(reps) * nbytes

        self._launch_write(dst, nbytes, 1)
        self._check(self.lib.cuCtxSynchronize(), "sync write warmup")
        rate, _elapsed = self._scaled(launch, 1, traffic, target, "write", warmup=False)
        pattern = b"\x00\x00\x80\x3f" * 4
        if self._read(dst, 16) != pattern:
            raise GpuError("GPU write kernel did not store the expected pattern")
        return rate / 1e9

    def l2_gbs(self, target: float) -> float:
        cache = self._attribute(38)
        nbytes = max(8 << 20, min(cache // 4 if cache else 12 << 20, 24 << 20))
        nbytes -= nbytes % (1 << 20)
        src, dst = self._copy_buffers(nbytes)

        def launch(reps: int) -> None:
            self._launch_copy(src, dst, nbytes, reps)

        def traffic(reps: int) -> float:
            return float(reps) * nbytes * 2

        self._launch_copy(src, dst, nbytes, 1)
        self._check(self.lib.cuCtxSynchronize(), "sync L2 warmup")
        rate, _elapsed = self._scaled(launch, 4, traffic, target, "L2", warmup=False)
        return rate / 1e9

    def stride_mlookups(self, target: float, quick: bool) -> float:
        nbytes = (64 if quick else 128) << 20
        buf = self._alloc(nbytes)
        self._check(self.lib.cuMemsetD8_v2(buf, 0x5A, nbytes), "fill stride buffer")
        mask = (nbytes // 4) - 1

        def launch(iterations: int) -> None:
            self._launch_stride(buf, mask, iterations)

        def lookups(iterations: int) -> float:
            return float(iterations) * self.threads

        rate, _elapsed = self._scaled(launch, 64, lookups, target, "random access")
        return rate / 1e6

    def cycle(self, stop: threading.Event, seconds: float, quick: bool, on_result) -> None:
        """Run every GPU test in order until stop is set, reporting each measured rate."""
        steps = (
            ("gpu_fp32", "FP32 compute", "GFLOPS", lambda: self.fp32_gflops(seconds)),
            ("gpu_fp16", "FP16 compute", "GFLOPS", lambda: self.fp16_gflops(seconds)),
            ("gpu_fp64", "FP64 compute", "GFLOPS", lambda: self.fp64_gflops(seconds)),
            ("gpu_sfu", "Special functions", "Gops/s", lambda: self.sfu_gops(seconds)),
            ("gpu_int", "Integer compute", "Gops/s", lambda: self.integer_gops(seconds)),
            ("gpu_int64", "64-bit integer", "Gops/s", lambda: self.int64_gops(seconds)),
            ("gpu_bit", "Bitwise", "Gops/s", lambda: self.bit_gops(seconds)),
            ("gpu_bw", "Global copy", "GB/s", lambda: self.bandwidth_gbs(seconds, quick)),
            ("gpu_read", "Global read", "GB/s", lambda: self.read_gbs(seconds, quick)),
            ("gpu_write", "Global write", "GB/s", lambda: self.write_gbs(seconds, quick)),
            ("gpu_l2", "L2 cache", "GB/s", lambda: self.l2_gbs(seconds)),
            ("gpu_shared", "Shared memory", "GB/s", lambda: self.shared_gbs(seconds)),
            ("gpu_stride", "Random access", "Mlookups/s", lambda: self.stride_mlookups(seconds, quick)),
        )
        while not stop.is_set():
            for test_id, name, unit, run in steps:
                if stop.is_set():
                    return
                try:
                    value = run()
                except Exception as exc:
                    on_result(test_id, name, None, unit, str(exc))
                    continue
                on_result(test_id, name, value, unit, None)

    def stress(self, stop: threading.Event, percent: int = 100) -> None:
        """Keep the GPU at the requested load until stop is set. Each launch stays under the display timeout."""
        sink = self._alloc(self.threads * 4)
        sample = 8_192

        def launch(iterations: int) -> None:
            self._launch_compute(self.fp32, iterations, sink)

        self._scaled_warmup(launch, sample)
        sample_s = self._elapsed(lambda: launch(sample))
        if percent >= 100:
            iterations = max(sample, int(sample * (0.25 / sample_s)))
            while not stop.is_set():
                self._elapsed(lambda: launch(iterations))
            return
        # Short slices so the driver's utilization sample averages near the requested load.
        slice_s = 0.01
        iterations = max(sample, int(sample * (slice_s / sample_s)))
        import time

        ctypes.windll.winmm.timeBeginPeriod(1)
        try:
            while not stop.is_set():
                wall = time.perf_counter()
                elapsed = self._elapsed(lambda: launch(iterations))
                overhead = max(0.0, time.perf_counter() - wall - elapsed)
                idle = elapsed * (100 - percent) / percent - overhead
                if idle > 0 and stop.wait(idle):
                    return
        finally:
            ctypes.windll.winmm.timeEndPeriod(1)

    def _alu(self, fn: ctypes.c_void_p, per_iter: int, sample: int, target: float, label: str) -> float:
        sink = self._alloc(self.threads * 4)

        def launch(iterations: int) -> None:
            self._launch_compute(fn, iterations, sink)

        def work(iterations: int) -> float:
            return float(iterations) * per_iter * self.threads

        rate, _elapsed = self._scaled(launch, sample, work, target, label)
        return rate

    def _scaled(self, launch, sample_units: int, work_for, target: float, label: str, warmup: bool = True):
        if warmup:
            self._scaled_warmup(launch, sample_units)
        sample = self._elapsed(lambda: launch(sample_units))
        units = max(1, int(sample_units * (_CHUNK_SECONDS / sample)))
        chunk = self._elapsed(lambda: launch(units))
        if chunk > 0.4:
            units = max(1, int(units * (0.18 / chunk)))
            chunk = self._elapsed(lambda: launch(units))
        if chunk < 0.01 and units > sample_units * 8:
            raise GpuError(f"GPU {label} kernel did not scale with the amount of work")
        rounds = max(1, min(40, int(round(target / max(chunk, 1e-4)))))
        elapsed = self._elapsed_many(lambda: launch(units), rounds)
        return work_for(units) * rounds / elapsed, elapsed

    def _scaled_warmup(self, launch, units: int) -> None:
        launch(max(1, units // 4))
        self._check(self.lib.cuCtxSynchronize(), "warmup")

    def _elapsed(self, fn) -> float:
        self._check(self.lib.cuEventRecord(self.start_ev, None), "record start")
        fn()
        self._check(self.lib.cuEventRecord(self.end_ev, None), "record end")
        self._check(self.lib.cuEventSynchronize(self.end_ev), "sync event")
        millis = ctypes.c_float()
        self._check(
            self.lib.cuEventElapsedTime(ctypes.byref(millis), self.start_ev, self.end_ev),
            "read event time",
        )
        return max(millis.value / 1000.0, 1e-6)

    def _elapsed_many(self, fn, rounds: int) -> float:
        self._check(self.lib.cuEventRecord(self.start_ev, None), "record start")
        for _ in range(rounds):
            fn()
        self._check(self.lib.cuEventRecord(self.end_ev, None), "record end")
        self._check(self.lib.cuEventSynchronize(self.end_ev), "sync event")
        millis = ctypes.c_float()
        self._check(
            self.lib.cuEventElapsedTime(ctypes.byref(millis), self.start_ev, self.end_ev),
            "read event time",
        )
        return max(millis.value / 1000.0, 1e-6)

    def _launch_compute(self, fn: ctypes.c_void_p, iterations: int, sink: int) -> None:
        args = [ctypes.c_uint64(iterations), ctypes.c_uint64(sink)]
        self._launch(fn, args)

    def _launch_copy(self, src: int, dst: int, nbytes: int, reps: int) -> None:
        args = [
            ctypes.c_uint64(dst),
            ctypes.c_uint64(src),
            ctypes.c_uint64(nbytes // 16),
            ctypes.c_uint64(reps),
        ]
        self._launch(self.copy_fn, args)

    def _launch_read(self, ptr: int, nbytes: int, reps: int, sink: int) -> None:
        self._launch(
            self.read_fn,
            [
                ctypes.c_uint64(ptr),
                ctypes.c_uint64(nbytes // 16),
                ctypes.c_uint64(reps),
                ctypes.c_uint64(sink),
            ],
        )

    def _launch_write(self, ptr: int, nbytes: int, reps: int) -> None:
        self._launch(
            self.write_fn,
            [
                ctypes.c_uint64(ptr),
                ctypes.c_uint64(nbytes // 16),
                ctypes.c_uint64(reps),
            ],
        )

    def _launch_stride(self, ptr: int, mask: int, iterations: int) -> None:
        self._launch(
            self.stride_fn,
            [
                ctypes.c_uint64(iterations),
                ctypes.c_uint64(ptr),
                ctypes.c_uint64(mask),
            ],
        )

    def _launch(self, fn: ctypes.c_void_p, args: list[ctypes.c_uint64]) -> None:
        slots = (ctypes.c_void_p * len(args))()
        for index, arg in enumerate(args):
            slots[index] = ctypes.addressof(arg)
        self._keepalive = (args, slots)
        self._check(
            self.lib.cuLaunchKernel(
                fn,
                self.grid,
                1,
                1,
                self.block,
                1,
                1,
                0,
                None,
                slots,
                None,
            ),
            "launch kernel",
        )

    def _copy_buffers(self, nbytes: int) -> tuple[int, int]:
        src = self._alloc(nbytes)
        dst = self._alloc(nbytes)
        self._check(self.lib.cuMemsetD8_v2(src, 0xA5, nbytes), "fill source")
        self._check(self.lib.cuMemsetD8_v2(dst, 0, nbytes), "clear dest")
        return src, dst

    def _alloc(self, nbytes: int) -> int:
        ptr = ctypes.c_uint64()
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(ptr), nbytes), "allocate GPU memory")
        self._memory.append(int(ptr.value))
        return int(ptr.value)

    def _read(self, ptr: int, size: int, offset: int = 0) -> bytes:
        buf = ctypes.create_string_buffer(size)
        self._check(self.lib.cuMemcpyDtoH_v2(buf, ptr + offset, size), "read GPU memory")
        return buf.raw

    def _function(self, name: bytes) -> ctypes.c_void_p:
        fn = ctypes.c_void_p()
        self._check(self.lib.cuModuleGetFunction(ctypes.byref(fn), self.module, name), f"find {name.decode()}")
        return fn

    def _pick_device(self) -> int:
        count = ctypes.c_int()
        self._check(self.lib.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        chosen = None
        best = -1
        for index in range(count.value):
            device = ctypes.c_int()
            self._check(self.lib.cuDeviceGet(ctypes.byref(device), index), "cuDeviceGet")
            memory = ctypes.c_size_t()
            self._check(self.lib.cuDeviceTotalMem_v2(ctypes.byref(memory), device), "cuDeviceTotalMem")
            if memory.value > best:
                best = memory.value
                chosen = int(device.value)
        if chosen is None:
            raise GpuError("No CUDA GPU was found")
        return chosen

    def _device_name(self, device: int) -> str:
        name = ctypes.create_string_buffer(256)
        self._check(self.lib.cuDeviceGetName(name, len(name), device), "cuDeviceGetName")
        return name.value.decode().strip() or "CUDA GPU"

    def _attribute(self, attrib: int) -> int:
        value = ctypes.c_int()
        self._check(
            self.lib.cuDeviceGetAttribute(ctypes.byref(value), attrib, self.device),
            "cuDeviceGetAttribute",
        )
        return int(value.value)

    def _check(self, code: int, what: str) -> None:
        if code == 0:
            return
        message = ctypes.c_char_p()
        if self.lib.cuGetErrorString(code, ctypes.byref(message)) == 0 and message.value:
            detail = message.value.decode("utf-8", "replace")
        else:
            detail = f"error {code}"
        raise GpuError(f"{what} failed ({detail})")

    def _bind(self) -> None:
        lib = self.lib
        c_int = ctypes.c_int
        c_uint = ctypes.c_uint
        c_void_p = ctypes.c_void_p
        c_size_t = ctypes.c_size_t
        c_uint64 = ctypes.c_uint64
        c_float = ctypes.c_float
        c_ubyte = ctypes.c_ubyte

        def bind(name: str, argtypes: list, restype=c_int) -> None:
            fn = getattr(lib, name)
            fn.argtypes = argtypes
            fn.restype = restype

        bind("cuInit", [c_uint])
        bind("cuGetErrorString", [c_int, ctypes.POINTER(ctypes.c_char_p)])
        bind("cuDeviceGetCount", [ctypes.POINTER(c_int)])
        bind("cuDeviceGet", [ctypes.POINTER(c_int), c_int])
        bind("cuDeviceGetName", [ctypes.c_char_p, c_int, c_int])
        bind("cuDeviceTotalMem_v2", [ctypes.POINTER(c_size_t), c_int])
        bind("cuDeviceGetAttribute", [ctypes.POINTER(c_int), c_int, c_int])
        bind("cuCtxCreate_v2", [ctypes.POINTER(c_void_p), c_uint, c_int])
        bind("cuCtxDestroy_v2", [c_void_p])
        bind("cuCtxSynchronize", [])
        bind(
            "cuModuleLoadDataEx",
            [
                ctypes.POINTER(c_void_p),
                c_void_p,
                c_uint,
                ctypes.POINTER(c_uint),
                ctypes.POINTER(c_void_p),
            ],
        )
        bind("cuModuleUnload", [c_void_p])
        bind("cuModuleGetFunction", [ctypes.POINTER(c_void_p), c_void_p, ctypes.c_char_p])
        bind("cuMemAlloc_v2", [ctypes.POINTER(c_uint64), c_size_t])
        bind("cuMemFree_v2", [c_uint64])
        bind("cuMemsetD8_v2", [c_uint64, c_ubyte, c_size_t])
        bind("cuMemcpyDtoH_v2", [c_void_p, c_uint64, c_size_t])
        bind("cuEventCreate", [ctypes.POINTER(c_void_p), c_uint])
        bind("cuEventRecord", [c_void_p, c_void_p])
        bind("cuEventSynchronize", [c_void_p])
        bind("cuEventElapsedTime", [ctypes.POINTER(c_float), c_void_p, c_void_p])
        bind("cuEventDestroy_v2", [c_void_p])
        bind(
            "cuLaunchKernel",
            [
                c_void_p,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_void_p,
                ctypes.POINTER(c_void_p),
                c_void_p,
            ],
        )
