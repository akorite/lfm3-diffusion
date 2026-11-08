"""Held-out evaluation on corpus rows the cache never touched.

The cache consumed the first ~26.9k examples of the chat file; the tail is
genuinely unseen, so scoring it measures generalisation rather than fit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lfm3_dlm import Lfm3ForCausalLM
from lfm3_dlm.sample import generate
from lfm3_dlm.tokenizer import format_chat, load_tokenizer
from lfm3_dlm.train.collator import IM_END_ID, SharedPrefixCollator
from lfm3_dlm.train.loss_cad import compute_step_loss

OUTPUT_VOCAB = 125_033


def build_examples(dataset_file, start, count, tokenizer, s_phys, block_size=32):
    from datasets import load_dataset

    try:
        # slice form; not every datasets version supports it for data_files
        ds = load_dataset("json", data_files=dataset_file,
                          split=f"train[{start}:{start + count}]")
    except Exception:
        ds = load_dataset("json", data_files=dataset_file, split="train") \
            .select(range(start, start + count))
    out = []
    for row in ds:
        prompt = row["input"]
        msgs = prompt if isinstance(prompt, list) else [
            {"role": "user", "content": prompt}]
        if tokenizer.chat_template:
            p = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                              tokenize=True)
        else:
            p = tokenizer.encode(format_chat(msgs), add_special_tokens=False)
        r = tokenizer.encode(str(row["output"]), add_special_tokens=False)
        r = r + [tokenizer.eos_token_id or IM_END_ID]
        rr = max(block_size, -(-len(r) // block_size) * block_size)
        if len(p) + 3 * rr <= s_phys:
            out.append({"prompt_ids": p, "response_ids": r})
    return out


@torch.no_grad()
def heldout_loss(model, examples, s_phys, block_size):
    coll = SharedPrefixCollator(s_phys=s_phys, block_size=block_size, n_masks=2)
    model.eval()
    total, n = 0.0, 0
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
        total += float(res.loss)
        n += 1
    return total / max(n, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-file", default="/content/data/chat.jsonl")
    ap.add_argument("--start", type=int, default=30000,
                    help="first corpus row the cache never consumed")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--s-phys", type=int, default=4096)
    ap.add_argument("--out", default="/content/heldout.json")
    args = ap.parse_args(argv)

    tok = load_tokenizer()
    model = Lfm3ForCausalLM.from_pretrained("LiquidAI/LFM3-1.7B", dtype=torch.bfloat16)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=False)
    model = model.cuda().eval()

    examples = build_examples(args.data_file, args.start, args.count, tok, args.s_phys)
    result = {"rows_requested": args.count, "examples_that_fit": len(examples)}
    if examples:
        model.config.attention_mode = "block_diffusion"
        result["heldout_loss"] = round(heldout_loss(model, examples, args.s_phys, 32), 3)
        model.config.attention_mode = "causal"
        ids = torch.tensor([examples[0]["prompt_ids"]][:1], device="cuda")
        res = generate(model, ids, mode="ar_trust", max_new_tokens=48, isd_width=4,
                       block_size=8, small_block=4)
        result["sample"] = tok.decode(res.ids[0, ids.shape[1]:].tolist(),
                                      skip_special_tokens=True)[:400]
    Path(args.out).write_text(json.dumps(result, indent=1))
    print("HELDOUT", json.dumps(result)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
