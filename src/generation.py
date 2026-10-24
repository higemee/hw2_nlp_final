import numpy as np

from codefmt import to_program
from data import BOS_ID, EOS_ID, pad_batch


def generate_codes(model, tok, questions, device, amp_ctx, batch_size=256, **gen_kw):
    """Questions -> runnable programs in the input order (k samples of question i at i*k ... i*k+k-1)."""
    enc = [e.ids for e in tok.encode_batch(questions, add_special_tokens=False)]
    order = np.argsort([len(e) for e in enc])
    k = gen_kw.get("num_samples", 1)
    codes = [None] * (len(questions) * k)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        src = pad_batch([enc[i] for i in idx]).to(device)
        with amp_ctx:
            outs = model.generate(src, BOS_ID, EOS_ID, **gen_kw)
        texts = tok.decode_batch(outs)
        for j, i in enumerate(idx):
            for r in range(k):
                codes[i * k + r] = to_program(texts[j * k + r])
    return codes
