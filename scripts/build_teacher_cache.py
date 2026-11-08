"""Phase 0: build the Top-16 teacher-logprob cache from `LiquidAI/LFM3-1.7B`.

Streams chat examples, teacher-forces `prompt + response`, and writes the
68 B/token cache the collator consumes:

    <out>/shard-00000.bin    packed `pack_top16` rows, one per response token
    <out>/index.jsonl        one line per example: shard, offset, P, R
    <out>/meta.json          totals, so a rerun resumes instead of restarting

Every example keeps its own prompt/response ids in the index, so the trainer
never re-tokenizes and the cache stays a pure numeric artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lfm3_dlm.data.top16_cache import BYTES_PER_TOKEN, TOP_K, pack_top16, unpack_top16
from lfm3_dlm.tokenizer import MASK_TOKEN_ID, load_tokenizer

OUTPUT_VOCAB = 125_033
TAU_FLOOR = 0.02


def _chat_pair(example, prompt_field, response_field):
    prompt, response = example[prompt_field], example[response_field]
    if isinstance(prompt, list):  # already a message list
        return prompt, response
    return [{"role": "user", "content": prompt}], str(response)


def _encode(tokenizer, prompt, response, seq_len):
    prompt_ids = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=True
    )
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    response_ids = response_ids + [tokenizer.eos_token_id or 124_900]
    if len(prompt_ids) + len(response_ids) > seq_len:
        keep = max(1, seq_len - len(prompt_ids))
        if keep < len(response_ids):  # drop the turn rather than truncate mid-answer
            return None
    return prompt_ids, response_ids


def _topk_chunk(hidden, classifier, chunk):
    """Top-16 over the sliced classifier, in position chunks so the
    [positions, 125033] logit block never lands whole in memory."""
    ids, lps, res = [], [], []
    for start in range(0, hidden.shape[0], chunk):
        logits = (hidden[start:start + chunk].float() @ classifier.float().t())
        lp = torch.log_softmax(logits, dim=-1)
        top = lp.topk(TOP_K, dim=-1)
        ids.append(top.indices)
        lps.append(top.values)
        res.append((1.0 - top.values.exp().sum(-1)).clamp_min(TAU_FLOOR))
    return torch.cat(ids), torch.cat(lps), torch.cat(res)


@torch.no_grad()
def teacher_rows(model, prompt_ids, response_ids, head_chunk):
    """Top-16 teacher distribution for every response token of one example."""
    ids = torch.tensor([prompt_ids + response_ids], device=model.device)
    hidden = model.model(input_ids=ids).last_hidden_state[0]
    P = len(prompt_ids)
    # position P-1+j predicts response token j
    hidden = hidden[P - 1:P - 1 + len(response_ids)]
    return _topk_chunk(hidden, model.get_output_classifier(), head_chunk)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    ap.add_argument("--dataset", required=True, help="HF dataset id")
    ap.add_argument("--split", default="train")
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--response-field", default="response")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=150_000_000)
    ap.add_argument("--shard-tokens", type=int, default=4_000_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--head-chunk", type=int, default=256)
    ap.add_argument("--fp8", action="store_true", help="torchao PerRow-FP8 teacher")
    ap.add_argument("--limit-examples", type=int, default=0)
    args = ap.parse_args(argv)

    from datasets import load_dataset

    from lfm3_dlm import Lfm3ForCausalLM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.jsonl"
    done = sum(1 for _ in index_path.open()) if index_path.exists() else 0

    tokenizer = load_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Lfm3ForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(device).eval()
    if args.fp8:
        from torchao.float8 import convert_to_float8_training

        convert_to_float8_training(model)

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    shard_buf: list[torch.Tensor] = []
    shard_rows = 0
    shard_idx = len(list(out.glob("shard-*.bin")))
    tokens = done
    t0 = time.perf_counter()

    def flush():
        nonlocal shard_buf, shard_rows, shard_idx
        if not shard_rows:
            return
        path = out / f"shard-{shard_idx:05d}.bin"
        path.write_bytes(torch.cat(shard_buf).numpy().tobytes())
        shard_buf, shard_rows = [], 0
        shard_idx += 1

    with index_path.open("a") as index:
        for i, example in enumerate(ds):
            if i < done:
                continue
            if args.limit_examples and i - done >= args.limit_examples:
                break
            response = example[args.response_field]
            if not str(response).strip() or str(response).strip() == "N/A":
                continue  # prompt-only rows (the code split) have no reference to force
            prompt, response = _chat_pair(example, args.prompt_field, args.response_field)
            enc = _encode(tokenizer, prompt, response, args.seq_len)
            if enc is None:
                continue
            prompt_ids, response_ids = enc
            ids, lps, res = teacher_rows(model, prompt_ids, response_ids, args.head_chunk)
            buf = pack_top16(ids, lps, res).cpu()
            index.write(json.dumps({
                "shard": shard_idx,
                "offset": shard_rows,
                "P": len(prompt_ids),
                "R": len(response_ids),
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
            }) + "\n")
            shard_buf.append(buf)
            shard_rows += len(response_ids)
            tokens += len(response_ids)
            if shard_rows >= args.shard_tokens:
                flush()
            if tokens >= args.max_tokens:
                break
            if tokens and tokens % 1_000_000 < len(response_ids):
                rate = tokens / max(time.perf_counter() - t0, 1e-9)
                print(f"{tokens/1e6:.0f}M tokens, {rate:.0f} tok/s", flush=True)
                (out / "meta.json").write_text(json.dumps({
                    "tokens": tokens, "examples": i + 1, "rate_tok_s": rate,
                    "bytes_per_token": BYTES_PER_TOKEN,
                }, indent=1))
    flush()
    (out / "meta.json").write_text(json.dumps({
        "tokens": tokens,
        "shards": shard_idx,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }, indent=1))
    print(f"cache done: {tokens} tokens in {shard_idx} shards -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
