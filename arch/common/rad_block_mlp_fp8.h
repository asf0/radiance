/* rad_block_mlp_fp8.h -- the SwiGLU feed-forward block with block-scaled FP8 linears.
 *
 *   rmsnorm -> quant_act_fp8 -> gate_up gemm_nt_q -> silu_mul -> quant_act_fp8 -> down gemm_nt_q
 *   -> [all_reduce] -> residual add
 *
 * rad_block_mlp.h documents the architecture. Differences:
 *
 *   - Both projections are LinearFP8 (rad_fp8.h): an E4M3 weight with a bf16 [N/128][K/128]
 *     scale plane, through `gemm_nt_q` at dtype fp8a8.
 *   - Each takes an E4M3 activation quantised every step (the checkpoint's `activation_scheme`
 *     is `dynamic`, so no activation scale is stored).
 *   - Activations between ops stay bf16, so `Geom::dtype` stays "bf16" and every other op
 *     resolves as in the bf16 block.
 *
 * The gate/up fusion survives the format because each part is a whole number of 128-row blocks:
 * two [17408, 5120] sources with [136, 40] scale planes concatenate to [34816, 5120] and
 * [272, 40]. Parts not on a 128-row boundary would misalign every scale after the seam.
 */
#ifndef RAD_BLOCK_MLP_FP8_H
#define RAD_BLOCK_MLP_FP8_H

#include "rad_fp8.h"

namespace rad {
namespace arch {

struct MlpFP8 {
    struct Wire {
        rad_buf x       = 0;   /* [max_tok, n_embd]  the residual stream */
        ActFP8  h{};           /* [max_tok, n_embd]  normed input, then the down-projection output */
        rad_buf gate_up = 0;   /* [max_tok, 2 * n_ff] */
        ActFP8  ffn{};         /* [max_tok, n_ff] */
    };

    /* Checkpoint names. `gate`, `up` and `down` name the `.weight` tensors; the checkpoint reader
     * pairs each with its scale plane. */
    struct Src {
        const char* norm  = nullptr;
        const char* gate  = nullptr;
        const char* up    = nullptr;
        const char* down  = nullptr;
    };

    Geom      g{};
    Wire      w{};
    int       layer = 0;

    rad_weight w_norm = 0;
    LinearFP8  gate_up{}, down{};
    NormQuantFP8 nq_h{};
    GateQuantFP8 gq_ffn{};
    rad_op     op_ar = 0, op_add = 0;
    /* gate_up with the gate and quantiser in its epilogue, so the [M, 2*n_ff] bf16 product is
     * never written (unfused, the extra launch and second pass are a sizeable decode cost).
     * Declared beside the pair, not instead of it: the narrow fold serves M <= 64, so prefill
     * chunks take the pair -- unless the model asked for `wide_fold`, whose row serves every M.
     * Byte-identical, as r4d_selftest and ref_test check. */
    rad_op     op_gated = 0;
    int64_t    gated_rows = 0;
    /* The next block absorbs this block's closing all-reduce, so this one must not issue it.
     * Counterpart of `ar_in`; both sides ask ar_fused_ok(g, T), so they agree. */
    bool ar_out = false;
    /* Set before declare: a clamped product (`swiglu_limit`) and its quantiser as one launch
     * (GateQuantFP8::fuse_clamped). */
    bool swiglu_quant = false;

    /* `fold`: a predecessor's delta is still in `w.h.x` and this norm does its residual add
     * (false for the first block of a chain). `add_out`: issue the trailing add here. See
     * rad_fp8.h. */
    /* `wide_fold`: declare the gated fold over the whole prompt (gemm_nt_q_gated_wide, whose row
     * serves every M) rather than the narrow row's 64-row band. Set by a model that runs whole
     * prompts through the fold; every other caller keeps the band and the pair past it. */
    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                 const Src& src, bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false, double swiglu_limit = 0,
                 bool wide_fold = false);
    void step(RadCtx* c, const RadBatch* batch) const;
    /* The block over the first `T` rows of its buffers (see MoeFP8::step_rows). */
    void step_rows(RadCtx* c, int64_t T) const { slice(c, T, 0, T); }
    /* One row slice of the whole block; `rows <= 0` means the rest of the step. */
    void slice(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const;
    /* The slice in its parts, for a caller that runs the norm and the tail elsewhere (qwen35_fp8's
     * pipelined prefill puts both on the second lane): the projections from the normed input to
     * the delta in `w.h.x`, then the all-reduce and add this block issues itself. */
    void body(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const;
    void tail(RadCtx* c, int64_t r0, int64_t rows) const;
    /* Local (aux split, MlpFP8::ax): the body in three parts, for a caller that overlaps pieces.
     * ax_begin hands rows [r0, r0 + rows)'s codes to the aux slice and issues it on lane 2, marking
     * slot `par` when it is done; body_own is this instance's slice; ax_end brings the slice's
     * output back (DMA on lane 1 for a wide piece, marking par + 2) and adds it into `w.h.x`.
     * body() is the three in a row at par 0. */
    void ax_begin(RadCtx* c, int64_t T, int64_t r0, int64_t rows, int par) const;
    void body_own(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const;
    void ax_end(RadCtx* c, int64_t r0, int64_t rows, int par) const;
    /* The block's all-reduce over rows [r0, r0 + rows) of the output, at `overlap` where declared
     * (a caller that runs it beside other work). */
    void issue_ar(RadCtx* c, int64_t r0, int64_t rows) const {
        RAD_ISSUE_N(c, op_ar_ov ? op_ar_ov : op_ar, rows * g.n_embd,
                    brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE);
    }
    /* Set before declare: the `overlap` issue_ar's collective is declared at, 0 for none. */
    int    ar_overlap = 0;
    rad_op op_ar_ov = 0;

    /* Local (aux device): a dense FFN split along its intermediate dimension between the rank's
     * card and an aux device. Set before declare. `ff` channels from `ff_lo` (0: all of n_ff).
     * `part`: a slice another instance drives (no norm, no all-reduce, no closing add), its
     * weights views of the stored tensors (`stored_gu`, `stored_dn`) in their own groups so the
     * planner pins them for the aux card; `probe_from` lends it the full instance's probe of the
     * down projection. On the rank's instance, `ax` is the aux slice it runs beside its own and
     * `ax_out_d` a VRAM copy of the slice's output, filled by DMA on lane 1 for wide steps. */
    int64_t ff = 0, ff_lo = 0;
    bool    part = false;
    const char* stored_gu = nullptr;
    const char* stored_dn = nullptr;
    const MlpFP8* probe_from = nullptr;
    const MlpFP8* ax = nullptr;
    rad_buf ax_out_d = 0;
    rad_op  op_ax_add = 0;
    int64_t F() const { return ff > 0 ? ff : g.n_ff; }
    /* RADIANCE_AUX_DMA_ROWS (default 64, 0 off): the rows from which the slice's output comes back
     * by DMA instead of being read over the link by the add. */
    bool ax_dma(int64_t rows) const {
        static const long long th = [] {
            const char* e = std::getenv("RADIANCE_AUX_DMA_ROWS");
            return e ? std::atoll(e) : 64LL;
        }();
        return ax && ax_out_d && th > 0 && rows >= th;
    }
    /* A wide piece's copies go to the card's DMA engine as one list (RADIANCE_AUX_SDMA=0: copy
     * kernels), so they hold no CU slot beside the rank's slice; a narrow one's as copy kernels. */
    static void ax_copy(const RadCopySeg* seg, int n, bool wide, RadStream s) {
        static const bool sdma = [] {
            const char* e = std::getenv("RADIANCE_AUX_SDMA");
            return !e || std::atoi(e) != 0;
        }();
        if (wide && sdma && rad_memcpy_list_async(seg, n, 0, s) == RAD_OK) return;
        for (int i = 0; i < n; ++i) (void)rad_memcpy_async(seg[i].dst, seg[i].src, seg[i].bytes, s);
    }
};

inline int MlpFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                           const Src& src, bool fold, bool add_out,
                           int ar_in, bool ar_out_, double swiglu_limit, bool wide_fold) {
    g = geom; w = wire; layer = l;
    ar_out = ar_out_;

    const int64_t Fw = F();
    const int64_t s_lo = ff > 0 ? ff_lo : 0, s_hi = ff > 0 ? ff_lo + ff : 0;
    if (!part) {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", l), src.norm));
        w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

        /* Norm, quantise and residual add in one op (rad_fp8.h: NormQuantFP8). Folded, it adds the
         * previous block's delta from `h` into the stream; unfolded it reads the stream directly. */
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                                  fold ? w.x : 0, 0, ar_in));
    }
    if (part) gate_up.stored = stored_gu;
    RAD_ARCH_TRY(gate_up.declare(b, nm, g, nm.f(part ? "blk.%d.ffn_gate_up_aux" : "blk.%d.ffn_gate_up", l),
                                 2 * Fw, g.n_embd, RAD_SHARD_ROW,
                                 part ? grp_expert_slot(l, 0, 0) : grp_layer(l),
                                 { src.gate, src.up }, w.h, w.gate_up, 0, { Fw, Fw }, s_lo, s_hi));

    /* The bf16 silu product is written only if the down projection reads the bf16 plane
     * (LinearFP8::reads_x); a block-fp8 down_proj reads only codes and scales, and so does an
     * int8 one fed its codes (ActFP8::q8_fed). */
    const char* down_base = nm.f(part ? "blk.%d.ffn_down_aux" : "blk.%d.ffn_down", l);
    if (part && probe_from) {
        down.probed_i8 = probe_from->down.probed_i8;
        down.reads_x   = probe_from->down.reads_x;
        down.mapped    = true;
        down.stored    = stored_dn;
    } else {
        RAD_ARCH_TRY(down.probe(b, nm, g, down_base, { src.down }));
    }
    gq_ffn.fuse_clamped = swiglu_quant;
    RAD_ARCH_TRY(gq_ffn.declare(b, g, w.gate_up, w.ffn, Fw, "silu", 0,
                               down.reads_x && !(w.ffn.q8_fed && down.probed_i8), false,
                               swiglu_limit));

    /* The fold: decl_gated_fold (an fp8a8 tile) for block-fp8 weights, decl_gated_fold_mx for plain
     * MXFP4 at one to 16 rows (eight where K is short), decl_gated_fold_iqr for an IQR weight
     * whose input and output are the rotated int8 trunk's fed pairs, to 16 rows. Other gate_up
     * encodings (M2XFP4 included), and a clamped (swiglu_limit) product, take the pair. */
    const bool fold_ok = swiglu_limit <= 0 && !gate_up.f32_scale;
    if (fold_ok && !gate_up.bf16 && !gate_up.mx && !gate_up.m2x && !gate_up.paro5 && !gate_up.w4 &&
        !gate_up.i8)
        op_gated = decl_gated_fold(b, g.max_tok, gate_up.N, gate_up.K, gate_up.w, gate_up.ws,
                                   { w.h.q, w.h.s },
                                   gq_ffn.want_x
                                       ? std::initializer_list<rad_buf>{ w.ffn.q, w.ffn.s, w.ffn.x }
                                       : std::initializer_list<rad_buf>{ w.ffn.q, w.ffn.s },
                                   &gated_rows, wide_fold);
    else if (fold_ok && gate_up.iqr && w.h.q8_fed && w.h.q8_rot && w.ffn.q8_fed && w.ffn.q8_rot)
        op_gated = decl_gated_fold_iqr(b, g.max_tok, gate_up.N, gate_up.K, gate_up.w, gate_up.ws,
                                       { w.h.q8, w.h.s8 },
                                       gq_ffn.want_x
                                           ? std::initializer_list<rad_buf>{ w.ffn.q8, w.ffn.s8,
                                                                             w.ffn.x }
                                           : std::initializer_list<rad_buf>{ w.ffn.q8, w.ffn.s8 },
                                       &gated_rows);
    else if (fold_ok && gate_up.mx && !gate_up.paro)
        op_gated = decl_gated_fold_mx(b, g.max_tok, gate_up.N, gate_up.K, gate_up.w, gate_up.ws,
                                      gate_up.wref, { w.h.q, w.h.s },
                                      gq_ffn.want_x
                                          ? std::initializer_list<rad_buf>{ w.ffn.q, w.ffn.s,
                                                                            w.ffn.x }
                                          : std::initializer_list<rad_buf>{ w.ffn.q, w.ffn.s },
                                      &gated_rows);
    RAD_ARCH_TRY(down.declare(b, nm, g, down_base,
                              g.n_embd, Fw, RAD_SHARD_COL,
                              part ? grp_expert_slot(l, 0, 1) : grp_layer(l),
                              { src.down }, w.ffn, w.h.x, 0, {}, s_lo, s_hi));
    if (part) return RAD_OK;
    /* The aux slice's output added into this instance's (MlpFP8::ax), from VRAM after the DMA or
     * from the host buffer for a narrow step. */
    if (ax)
        op_ax_add = rw(b, RAD_OP(b, "add",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                      RAD_STR("dtype", g.dtype)),
                           RAD_NOWEIGHTS),
                       {w.h.x, ax_out_d ? ax_out_d : ax->w.h.x}, {w.h.x});

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});
    /* The same collective for a caller that runs it beside other work (issue_ar): at `overlap` 1
     * it gets the rows that hold fewer CUs while waiting on the link. */
    if (g.world > 1 && ar_overlap > 0)
        op_ar_ov = rw(b, RAD_OP(b, "all_reduce",
                          RAD_PARAMS(RAD_INT("world_size", g.world),
                                     RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                     RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                     RAD_INT("min_bytes", g.wire_min_bytes),
                                     RAD_INT("overlap", ar_overlap)),
                          RAD_NOWEIGHTS),
                      {w.h.x}, {w.h.x});

    /* Only when the next block's norm will not add (rad_fp8.h: NormQuantFP8). */
    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.x, w.h.x}, {w.x});

    return RAD_OK;
}

/* One row slice of the block; r0 = 0, rows = T is the whole step (`brow_slice` at offset 0 is
 * `brows`). Every op is per row: the norm and quantiser group along K within a row, the GEMMs
 * accumulate each output row independently, and the fused all-reduce reduces within a row
 * (r4d_ar.cpp). So results do not depend on the cut, provided it obeys the band rule in
 * core/runtime/ctx.h. */
inline void MlpFP8::slice(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    nq_h.step(c, T, r0, rows);
    body(c, T, r0, rows);
    tail(c, r0, rows);
}

inline void MlpFP8::ax_begin(RadCtx* c, int64_t T, int64_t r0, int64_t rows, int par) const {
    if (!ax) return;
    const int64_t sb = fp8_blocks(g.n_embd) * 4;
    /* A wide piece pushes the codes on lane 1 (the copy engine) so lane 0 goes straight on to its
     * own slice; a narrow one copies on lane 0, where the extra hop would cost more. */
    const int cl = ax_dma(rows) ? 1 : 0;
    if (cl) { rad_lane_join(c, 0, 1); rad_lane(c, 1); }
    const RadStream s0 = rad_lane_stream(c);
    const RadCopySeg seg[2] = {
        { (char*)rad_buf_ptr(c, w.h.q8) + r0 * g.n_embd, (char*)rad_buf_ptr(c, ax->w.h.q8) + r0 * g.n_embd,
          rows * g.n_embd },
        { (char*)rad_buf_ptr(c, w.h.s8) + r0 * sb, (char*)rad_buf_ptr(c, ax->w.h.s8) + r0 * sb, rows * sb } };
    ax_copy(seg, 2, cl != 0, s0);
    rad_lane_join(c, cl, 2);
    rad_lane(c, 2);
    ax->gate_up.step(c, ax->w.h, ax->w.gate_up, T, r0, rows);
    ax->gq_ffn.step(c, ax->w.gate_up, T, r0, rows);
    ax->down.step(c, ax->w.ffn, ax->w.h.x, T, r0, rows);
    rad_lane_mark(c, par);
    rad_lane(c, 0);
}

inline void MlpFP8::ax_end(RadCtx* c, int64_t r0, int64_t rows, int par) const {
    if (!ax) return;
    /* The DMA is issued here rather than in ax_begin, so lane 1 copies the next piece's codes
     * before it waits on this piece's slice. */
    const bool dma = ax_dma(rows);
    if (dma) {
        const int64_t eb = (int64_t)rad_dtype_bits(g.act_dtype) / 8;
        rad_lane(c, 1);
        rad_lane_wait_mark(c, par);
        const RadCopySeg seg = { (char*)rad_buf_ptr(c, ax->w.h.x) + r0 * g.n_embd * eb,
                                 (char*)rad_buf_ptr(c, ax_out_d) + r0 * g.n_embd * eb, rows * g.n_embd * eb };
        ax_copy(&seg, 1, true, rad_lane_stream(c));
        rad_lane_mark(c, par + 2);
        rad_lane(c, 0);
    }
    rad_lane_wait_mark(c, dma ? par + 2 : par);
    RAD_ISSUE_N(c, op_ax_add, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                brow_slice(dma ? ax_out_d : ax->w.h.x, r0, rows, g.n_embd),
                brow_slice(w.h.x, r0, rows, g.n_embd));
}

inline void MlpFP8::body_own(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    const int64_t Fw = F();
    if (op_gated && rows <= gated_rows && gate_up.mx) {
        RAD_ISSUE_N(c, op_gated, rows, brow_slice(w.h.q, r0, rows, g.n_embd),
                    brow_slice(w.h.s, r0, rows, g.n_embd / RAD_FP8_BLOCK), RAD_W(gate_up.w),
                    RAD_W(gate_up.ws), RAD_W(gate_up.wref), brow_slice(w.ffn.q, r0, rows, Fw),
                    brow_slice(w.ffn.s, r0, rows, Fw / RAD_FP8_BLOCK),
                    gq_ffn.want_x ? brow_slice(w.ffn.x, r0, rows, Fw) : RAD_NONE);
    } else if (op_gated && rows <= gated_rows) {
        /* The fp8a8 fold's E4M3 pairs, or the IQR fold's int8 ones (cq / cs: the fed pair). */
        RAD_ISSUE_N(c, op_gated, rows, brow_slice(w.h.cq(), r0, rows, g.n_embd),
                    brow_slice(w.h.cs(), r0, rows, g.n_embd / RAD_FP8_BLOCK), RAD_W(gate_up.w),
                    RAD_W(gate_up.ws), brow_slice(w.ffn.cq(), r0, rows, Fw),
                    brow_slice(w.ffn.cs(), r0, rows, Fw / RAD_FP8_BLOCK),
                    gq_ffn.want_x ? brow_slice(w.ffn.x, r0, rows, Fw) : RAD_NONE);
    } else {
        gate_up.step(c, w.h, w.gate_up, T, r0, rows);
        gq_ffn.step(c, w.gate_up, T, r0, rows);
    }
    down.step(c, w.ffn, w.h.x, T, r0, rows);
}

inline void MlpFP8::body(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    ax_begin(c, T, r0, rows, 0);
    body_own(c, T, r0, rows);
    ax_end(c, r0, rows, 0);
}

inline void MlpFP8::tail(RadCtx* c, int64_t r0, int64_t rows) const {
    if (op_ar && !(ar_out && ar_fused_ok(g, rows)))
        RAD_ISSUE_N(c, op_ar, rows * g.n_embd, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE);
    if (op_add)
        RAD_ISSUE_N(c, op_add, rows, brow_slice(w.x, r0, rows, g.n_embd),
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(w.x, r0, rows, g.n_embd));
}

inline void MlpFP8::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    slice(c, T, 0, T);
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MLP_FP8_H */
