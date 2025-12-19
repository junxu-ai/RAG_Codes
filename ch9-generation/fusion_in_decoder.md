# enc: encoder; dec: decoder; retrieve(x)-> list[(d_i, score_i)]
docs, scores = retrieve(x)[:K]

memories = []
for i, (d_i, s_i) in enumerate(docs):
    tokens = pack([D_ID(i), x, SEP, d_i])          # reset positions per pack
    H_i = enc(tokens)                               # [T_i, d_model]
    g_i = softmax(alpha * torch.tensor(scores))[i]  # optional gating
    memories.append(scale_kv(H_i, g_i**0.5))        # rescale keys/values

K_cat, V_cat = concat_kv(memories)                  # across all docs

y = [BOS]
kv_cache = build_kv_cache(K_cat, V_cat)             # per-layer cache
for t in range(MAX_LEN):
    q = dec.self_attend(y, cache=True)
    ctx = dec.cross_attend(q, kv_cache)             # fusion happens here
    logits = out_proj(ctx)
    y.append(sample_or_greedy(logits))
    if y[-1] == EOS: break
