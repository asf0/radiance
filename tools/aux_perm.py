import json, sys
# Local (aux split): writes the RADIANCE_AUX_PERM file for qwen4exp from radiance-expert-stats JSON recorded
# with every expert on one card (e.g. HIP_VISIBLE_DEVICES=1 --kernels libhalo,... --expert-stats F): in split
# mode only the aux instance reports its histogram. PLAIN lists the bf16 experts of qwen3.8-next-flash-fp8-iq4r-moe
# (rad-info -v); R is the split the lmswap settings pick.
# aux_perm.py [--plain=FILE] OUT.txt TRAIN.json[,TRAIN2.json] [EVAL.json ...]
# Per trunk layer: the 512 checkpoint experts coldest -> hottest by TRAIN's decode + prefill shares (each
# normalised per layer), the layer's bf16 experts first (coldest: kept on the aux card, an even count).
# Histogram rows from an iGPU-only run skip a layer's bf16 experts (row i = i-th other expert by id).
# Reports the R9700 share (the last R slots) on TRAIN and each EVAL (iGPU-only stats files).
R, NE = 184, 512
PLAIN = {34: [407, 496], 44: [292, 350], 46: [290, 392], 47: [122, 143, 399, 445]}
args = sys.argv[1:]
if args and args[0].startswith('--plain='):
    # --plain=FILE: the model's bf16 experts, one `blk.L.ffn_gate_up_exps.E` a line (rad-info -v)
    PLAIN = {}
    for ln in open(args.pop(0)[8:]):
        p = ln.strip().split('.')
        if len(p) >= 4: PLAIN.setdefault(int(p[1]), []).append(int(p[3]))
out, train, evals = args[0], args[1], args[2:]
def counts(path, kind):
    C = json.load(open(path))['counts'][kind][:48]; res = []
    for l, row in enumerate(C):
        pl = PLAIN.get(l, []); reg = [e for e in range(NE) if e not in pl]
        full = [0] * NE
        if len(row) == len(reg):
            for i, v in enumerate(row): full[reg[i]] = v
        elif len(row) == NE:
            full = list(row)
        else:
            raise SystemExit('%s layer %d: row of %d' % (path, l, len(row)))
        res.append(full)
    return res
# TRAIN may be several files joined by commas: each file's decode and prefill shares count equally.
sets = [(counts(t, 'decode'), counts(t, 'prefill')) for t in train.split(',')]
perm = []
for l in range(48):
    sc = [0.0] * NE
    for D, P in sets:
        for X in (D, P):
            n = sum(X[l])
            if n:
                for e in range(NE): sc[e] += X[l][e] / n
    for e in PLAIN.get(l, []): sc[e] = -1.0
    perm.append(sorted(range(NE), key=lambda e: (sc[e], e)))
if out != '-':
    with open(out, 'w') as f:
        for p in perm: f.write(' '.join(map(str, p)) + '\n')
for ev in train.split(',') + evals:
    for kind in ('decode', 'prefill'):
        C = counts(ev, kind); tot = cur = new = 0
        for l in range(48):
            tot += sum(C[l]); cur += sum(C[l][NE - R:]); new += sum(C[l][e] for e in perm[l][NE - R:])
        if tot: print('%-20s %-7s sel %9d  R9700 share: by id %.1f%%  hot-first %.1f%%' % (ev, kind, tot, 100 * cur / tot, 100 * new / tot))
