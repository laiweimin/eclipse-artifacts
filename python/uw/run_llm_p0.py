"""P0-LLM: end-to-end probe for LLM-scale unlearning audit attack.
GPT-2 + LoRA, synthetic user records, GA unlearning, 6-metric
behavioral audit, per-record z-score readout.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

# ── Synthetic user data ─────────────────────────────────────────
FIRST = ("Alice Bob Carol David Emma Frank Grace Henry Iris Jack "
         "Kate Liam Mia Noah Olivia Paul Quinn Rose Sam Tina "
         "Umar Vera Will Xena Yasuf Zoe").split()
CITY = ("London Paris Tokyo Berlin Madrid Rome Oslo Dublin "
        "Vienna Prague Warsaw Lisbon").split()
JOB = ("doctor teacher engineer nurse lawyer chef pilot "
       "artist writer farmer driver").split()

def make_records(n=200, seed=42):
    rs = np.random.RandomState(seed)
    recs = []
    for i in range(n):
        name = FIRST[rs.randint(len(FIRST))]
        age = rs.randint(22, 68)
        city = CITY[rs.randint(len(CITY))]
        job = JOB[rs.randint(len(JOB))]
        pid = 10000 + i
        rec = (f"User {name}, age {age}, lives in {city}, works "
               f"as a {job}. ID: {pid:05d}. "
               f"Phone: 555-{rs.randint(100,999)}-{rs.randint(1000,9999)}. "
               f"Email: {name.lower()}.{pid}@example.com. "
               f"Account balance: ${rs.randint(100,50000)}.")
        recs.append(rec)
    return recs


# ── Model / LoRA setup ──────────────────────────────────────────
def setup_model(device):
    from transformers import (GPT2LMHeadModel, GPT2Tokenizer,
                              get_linear_schedule_with_warmup)
    from peft import LoraConfig, get_peft_model
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    lcfg = LoraConfig(r=8, lora_alpha=16,
                      target_modules=['c_attn'],
                      lora_dropout=0.0, task_type='CAUSAL_LM')
    model = get_peft_model(model, lcfg)
    return model, tok


def encode_batch(tok, texts, device, max_len=128):
    return tok(texts, return_tensors='pt', padding=True,
               truncation=True, max_length=max_len).to(device)


def compute_loss(model, enc, reduction='none'):
    out = model(input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                labels=enc['input_ids'])
    # HuggingFace returns mean CE loss with internal shifting
    # For per-sequence: compute manually
    logits = out.logits[:, :-1]
    labels = enc['input_ids'][:, 1:]
    mask = enc['attention_mask'][:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           labels.reshape(-1),
                           reduction='none').view(labels.shape)
    loss = (loss * mask).sum(1) / mask.sum(1).clamp(min=1)
    return loss  # (B,)


# ── Audit metrics ───────────────────────────────────────────────
def perplexity(model, tok, texts, device, bs=8):
    ppls = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = encode_batch(tok, texts[i:i+bs], device)
            loss = compute_loss(model, enc)
            ppls.extend(torch.exp(loss).cpu().tolist())
    return float(np.mean(ppls))


def exact_match(model, tok, texts, device, bs=8, n_gen=20):
    model.eval()
    matches = 0
    with torch.no_grad():
        for t in texts[:10]:  # sample 10 for speed
            prompt = t[:60]
            ids = tok(prompt, return_tensors='pt').to(device)
            gen = model.generate(
                ids['input_ids'], max_new_tokens=n_gen,
                do_sample=False, pad_token_id=tok.eos_token_id)
            gen_text = tok.decode(gen[0], skip_special_tokens=True)
            if gen_text[:len(t)] == t[:len(gen_text)]:
                matches += 1
    return matches / 10


def per_record_loss(model, tok, texts, device, bs=8):
    losses = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = encode_batch(tok, texts[i:i+bs], device)
            loss = compute_loss(model, enc)
            losses.extend(loss.cpu().tolist())
    return np.array(losses)


def audit_metrics(model, tok, F, R_fresh, R_test, device):
    """LLM version of m1-m6."""
    m = {}
    m['m1_perp_test'] = perplexity(model, tok, R_test, device)
    m['m2_perp_forget'] = perplexity(model, tok, F, device)
    m['m3_exact_match'] = exact_match(model, tok, F, device)
    # m4: KL between forget and retain loss distributions
    f_loss = per_record_loss(model, tok, F, device)
    r_loss = per_record_loss(model, tok, R_fresh, device)
    # simple KS-like distance as proxy
    m['m4_loss_gap'] = float(np.mean(f_loss) - np.mean(r_loss))
    # m5: MIA AUC (forget vs fresh, per-record loss)
    from uw.run_gm import auc as rank_auc
    m['m5_mia_auc'] = float(rank_auc(-f_loss, -r_loss))
    # m6: mean log-prob margin on F
    m['m6_margin'] = float(np.mean(-f_loss))
    return m


# ── Main ────────────────────────────────────────────────────────
def run(cli):
    device = torch.device('cuda:0')
    model, tok = setup_model(device)
    print('Model loaded, LoRA params: %.1fM' %
          (sum(p.numel() for p in model.parameters()
               if p.requires_grad) / 1e6), flush=True)

    # 1. Data
    all_recs = make_records(200)
    rs = np.random.RandomState(54321)
    forget_idx = np.sort(rs.choice(200, 30, replace=False))
    F = [all_recs[i] for i in forget_idx]
    R_train = [all_recs[i] for i in range(200)
               if i not in forget_idx]  # 170 retain
    R_fresh = make_records(100, seed=999)  # never trained
    R_test = make_records(50, seed=888)  # utility test
    print('Data: %d train, %d forget, %d fresh, %d test' %
          (len(R_train), len(F), len(R_fresh), len(R_test)),
          flush=True)

    # 2. Fine-tune on all 200 (before "deletion")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4)
    model.train()
    for ep in range(3):
        total = 0
        for i in range(0, len(all_recs), 8):
            batch = all_recs[i:i+8]
            enc = encode_batch(tok, batch, device)
            loss = compute_loss(model, enc).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        print('FT epoch %d: loss=%.3f' % (ep, total / 25), flush=True)

    # 3. Audit on the fine-tuned model
    print('\n--- Audit on fine-tuned (pre-forget) ---', flush=True)
    m_pre = audit_metrics(model, tok, F, R_fresh, R_test, device)
    print(json.dumps(m_pre, indent=1), flush=True)

    # 4. GA unlearning on F
    opt_ga = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5)
    model.train()
    for step in range(50):
        for i in range(0, len(F), 8):
            enc = encode_batch(tok, F[i:i+8], device)
            loss = compute_loss(model, enc).mean()
            opt_ga.zero_grad()
            (-loss).backward()  # maximize loss = forget
            opt_ga.step()
        if step % 10 == 0:
            print('GA step %d: F loss=%.3f' % (step, loss),
                  flush=True)

    # 5. Audit on the unlearned model
    print('\n--- Audit on unlearned (post-GA) ---', flush=True)
    m_post = audit_metrics(model, tok, F, R_fresh, R_test, device)
    print(json.dumps(m_post, indent=1), flush=True)

    # 6. Per-record z-score readout
    f_loss = per_record_loss(model, tok, F, device)
    r_loss = per_record_loss(model, tok, R_fresh, device)
    from uw.run_gm import auc as rank_auc
    auc = float(rank_auc(-f_loss, -r_loss))
    print('\nReadout: member-vs-fresh AUC = %.3f' % auc, flush=True)

    # Save
    out = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'metrics_pre_forget': m_pre,
           'metrics_post_forget': m_post,
           'readout_auc': auc,
           'forget_indices': forget_idx.tolist()}
    with open(os.path.join(cli.outdir, 'verdict.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('P0 PROBE COMPLETE', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='')
    cli = ap.parse_args()
    if not cli.outdir:
        cli.outdir = 'research_outputs/uw/llm_p0-%s' % (
            time.strftime('%Y%m%dT%H%M%S'),)
    os.makedirs(cli.outdir, exist_ok=True)
    run(cli)


if __name__ == '__main__':
    main()
