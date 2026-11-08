"""Held-out loss and a few generations for a trained checkpoint."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from lfm3_dlm import Lfm3ForCausalLM
from lfm3_dlm.tokenizer import format_chat, load_tokenizer
from lfm3_dlm.train.collator import IM_END_ID, SharedPrefixCollator
from lfm3_dlm.train.loss_cad import compute_step_loss
from lfm3_dlm.sample import generate


def heldout_loss(model, coll, examples):
    total, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for row in coll(examples):
            row = row.to("cuda")
            out = model.model(input_ids=row.input_ids.unsqueeze(0),
                              position_ids=row.position_ids.unsqueeze(0),
                              doc_ids=row.doc_id.unsqueeze(0),
                              stream_ids=row.stream_id.unsqueeze(0))
            cad = row.cad_mask.nonzero(as_tuple=True)[0]
            ce = (row.ce_weight > 0).nonzero(as_tuple=True)[0]
            ar = (row.ar_weight > 0).nonzero(as_tuple=True)[0]
            res = compute_step_loss(
                out.last_hidden_state.squeeze(0), model.get_output_classifier(),
                cad_idx=cad, teacher_ids=row.teacher_ids[cad],
                teacher_logprobs=row.teacher_logprobs[cad],
                teacher_residual=row.teacher_residual[cad],
                ce_idx=ce, ce_targets=row.labels[ce], ce_weights=row.ce_weight[ce],
                ar_idx=ar, ar_targets=row.labels[ar], w_ar=1.0, causal_x0=True)
            total += float(res.loss); n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    model = Lfm3ForCausalLM.from_pretrained("LiquidAI/LFM3-1.7B", dtype=torch.bfloat16)
    model.load_state_dict(torch.load(a.weights, map_location="cpu"), strict=False)
    # the trained bundle is the full model; a silent key mismatch would score
    # the base model and call it a result, so count what actually loaded
    sd = torch.load(a.weights, map_location="cpu")
    loaded = sum(1 for k in sd if k in model.state_dict())
    model = model.cuda()
    tok = load_tokenizer()
    texts = [
        "Explain why the sky is blue.",
        "Write a Python function that reverses a linked list.",
        "Give one reason a database index speeds up reads.",
    ]
    samples, tpfs = [], []
    for text_prompt in texts:
        ids = tok(format_chat(text_prompt), return_tensors="pt",
                  add_special_tokens=False).input_ids.cuda()
        res = generate(model, ids, mode="ar_trust", max_new_tokens=48, isd_width=4,
                       block_size=8, small_block=4)
        samples.append(tok.decode(res.ids[0, ids.shape[1]:].tolist(),
                                  skip_special_tokens=True)[:400])
        tpfs.append(round(float(res.tpf), 3))
    out = {"tokens_per_forward": tpfs, "samples": samples,
           "weights_keys_loaded": loaded, "weights_keys_total": len(sd)}
    cache_dir = Path(a.cache)
    if (cache_dir / "index.jsonl").exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_g4 import TeacherCache
        tc = TeacherCache(cache_dir)
        probe = []
        for i in range(min(a.n, len(tc))):
            ex = tc.example(i)
            if ex is not None:
                probe.append(ex)
        if probe:
            coll = SharedPrefixCollator(s_phys=4096, block_size=32, n_masks=2)
            out["heldout_loss"] = round(heldout_loss(model, coll, probe), 3)
            out["heldout_examples"] = len(probe)
    print("EVAL", json.dumps(out))
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
