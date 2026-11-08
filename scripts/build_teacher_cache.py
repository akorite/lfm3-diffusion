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
    """Prompt ids, plus reference-response ids when one exists."""
    if tokenizer.chat_template:
        prompt_ids = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=True
        )
    else:
        # the vendored tokenizer assets carry no chat template: fall back to
        # the plain prompt text so teacher forcing still works
        text = prompt[0]["content"] if isinstance(prompt, list) else str(prompt)
        prompt_ids = tokenizer.encode(text + "\n", add_special_tokens=True)
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
@torch.no_grad()
def teacher_rollout(model, prompt_ids, max_new_tokens, eos_id):
    """Greedy rollout: the response the teacher itself would write."""
    ids = torch.tensor([prompt_ids], device=model.device)
    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                         do_sample=False, eos_token_id=eos_id,
                         pad_token_id=eos_id)
    return out[0, len(prompt_ids):].tolist()


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
    ap.add_argument("--batch-size", type=int, default=16,
                    help="examples per teacher forward (left-padded, masked)")
    ap.add_argument("--fp8", action="store_true", help="torchao PerRow-FP8 teacher")
    ap.add_argument("--limit-examples", type=int, default=0)
    ap.add_argument("--generate", action="store_true",
                    help="let the teacher emit the response (the plan's rollout) instead of "
                         "teacher-forcing a reference field; required for prompt-only rows")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
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

    pending: list[tuple[list[int], list[int]]] = []
    pending_prompts: list[list[int]] = []

    def flush_prompts():
        """Batched greedy rollout, then hand the pairs to the forcing pass."""
        nonlocal pending_prompts
        if not pending_prompts:
            return
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        eos = tokenizer.eos_token_id or 124_900
        max_p = max(len(p) for p in pending_prompts)
        n = len(pending_prompts)
        ids = torch.full((n, max_p), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((n, max_p), dtype=torch.long, device=device)
        for i, p in enumerate(pending_prompts):
            ids[i, max_p - len(p):] = torch.tensor(p, device=device)
            attn[i, max_p - len(p):] = 1
        with torch.no_grad():
            out_ids = model.generate(input_ids=ids, attention_mask=attn,
                                     max_new_tokens=args.max_new_tokens,
                                     do_sample=False, eos_token_id=eos,
                                     pad_token_id=pad_id)
        for i, p in enumerate(pending_prompts):
            new = out_ids[i, max_p:].tolist()
            if eos in new:
                new = new[: new.index(eos) + 1]
            if len(new) >= 2 and len(p) + len(new) <= args.seq_len:
                pending.append((p, new))
        pending_prompts = []
        flush_batch()

    def flush_batch():
        """One teacher forward for the whole pending batch, then the cached
        top-k per example. Prompts are left-padded so every response starts
        at the same column, and the padding is masked out."""
        nonlocal shard_buf, shard_rows, shard_idx, tokens
        if not pending:
            return
        max_p = max(len(p) for p, _ in pending)
        max_r = max(len(r) for _, r in pending)
        n, L = len(pending), max_p + max_r
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ids = torch.full((n, L), pad, dtype=torch.long, device=device)
        attn = torch.zeros((n, L), dtype=torch.long, device=device)
        pos = torch.zeros((n, L), dtype=torch.long, device=device)
        for i, (p, r) in enumerate(pending):
            ids[i, max_p - len(p):max_p] = torch.tensor(p, device=device)
            ids[i, max_p:max_p + len(r)] = torch.tensor(r, device=device)
            attn[i, max_p - len(p):max_p + len(r)] = 1
            pos[i, max_p - len(p):max_p] = torch.arange(len(p), device=device)
            pos[i, max_p:max_p + len(r)] = torch.arange(
                len(p), len(p) + len(r), device=device)
        with torch.no_grad():
            hidden = model.model(input_ids=ids, attention_mask=attn,
                                 position_ids=pos).last_hidden_state
        for i, (p, r) in enumerate(pending):
            rows = hidden[i, max_p - 1:max_p - 1 + len(r)]
            tids, lps, res = _topk_chunk(rows, model.get_output_classifier(),
                                         args.head_chunk)
            index.write(json.dumps({
                "shard": shard_idx, "offset": shard_rows,
                "P": len(p), "R": len(r),
                "prompt_ids": p, "response_ids": r,
            }) + "\n")
            shard_buf.append(pack_top16(tids, lps, res).cpu())
            shard_rows += len(r)
            tokens += len(r)
            if shard_rows >= args.shard_tokens:
                path = out / f"shard-{shard_idx:05d}.bin"
                path.write_bytes(torch.cat(shard_buf).numpy().tobytes())
                shard_buf, shard_rows = [], 0
                shard_idx += 1
        pending.clear()

    with index_path.open("a") as index:
        for i, example in enumerate(ds):
            if i < done:
                continue
            if args.limit_examples and i - done >= args.limit_examples:
                break
            prompt, reference = _chat_pair(example, args.prompt_field, args.response_field)
            enc = _encode(tokenizer, prompt, reference, args.seq_len)
            if enc is None:
                continue
            prompt_ids = enc[0]
            if args.generate:
                pending_prompts.append(prompt_ids)
                if len(pending_prompts) >= args.batch_size:
                    flush_prompts()
                if tokens >= args.max_tokens:
                    break
                continue
            else:
                if not str(reference).strip() or str(reference).strip() == "N/A":
                    continue  # prompt-only row and no rollout requested
                response_ids = enc[1]
            pending.append((prompt_ids, response_ids))
            if len(pending) >= args.batch_size:
                flush_batch()
            if tokens >= args.max_tokens:
                break
            if tokens and tokens % 1_000_000 < args.batch_size * 64:
                rate = tokens / max(time.perf_counter() - t0, 1e-9)
                print(f"{tokens/1e6:.1f}M tokens, {rate:.0f} tok/s", flush=True)
                (out / "meta.json").write_text(json.dumps({
                    "tokens": tokens, "examples": i + 1, "rate_tok_s": rate,
                    "bytes_per_token": BYTES_PER_TOKEN,
                }, indent=1))
        flush_prompts()
        flush_batch()
    (out / "meta.json").write_text(json.dumps({
        "tokens": tokens,
        "shards": shard_idx,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }, indent=1))
    print(f"cache done: {tokens} tokens in {shard_idx} shards -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
