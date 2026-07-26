# Training recipe

How `VIDAR-Grounded-7B` was produced, in enough detail to reproduce or back-test it.
The trainer itself is not shipped: it was thin glue over Unsloth/TRL, whose APIs move
quickly, and stale glue is worse than an accurate recipe. Everything below is what
that glue actually did.

What **is** shipped, because it cannot be reconstructed from prose:

- `grounding/contract.py` — the exact message shape the model is served under. Train
  and serve against this or results will not match.
- `dataset/` — the generator. Run it against your own licensed VIDA install; no rows
  are distributed.
- the eval benchmark + metrics — the numbers below mean nothing without them.

## Method

QLoRA (4-bit) via Unsloth, supervised fine-tune on the canonical contract JSONL, with
**response-only loss masking**.

```
r                     16
lora_alpha            32              (defaults to r * 2)
learning_rate         2e-4
epochs                2
batch_size            2
grad_accum            8               (effective batch 16)
max_seq_length        8192
save_steps            100
load_in_4bit          true
gradient_checkpointing "unsloth"
```

LoRA target modules — attention and FFN, not embeddings:

```
q_proj  k_proj  v_proj  o_proj        (attention)
gate_proj  up_proj  down_proj         (FFN)
```

### max_seq_length: use 8192, not 2048

The original 2048 **truncated answers** on realistic multi-hit tool results. The model
serves retrieval output, so sequences are long. 2048 silently produces a model that
looks trained and cuts off mid-answer.

## Response-only masking — the part that must be right

Loss is computed on assistant turns only. Masking the wrong span trains the model on
the system prompt and the tool-result facts it was supposed to *retrieve* — which is
precisely the fact-fabrication behaviour this project exists to eliminate.

Markers are detected from the tokenizer's `chat_template`:

| Family | Probe | instruction_part | response_part |
|---|---|---|---|
| Qwen (ChatML) | `<\|im_start\|>` | `<\|im_start\|>user\n` | `<\|im_start\|>assistant\n` |
| Llama 3.x | `<\|start_header_id\|>` | `<\|start_header_id\|>user<\|end_header_id\|>\n\n` | `<\|start_header_id\|>assistant<\|end_header_id\|>\n\n` |
| Gemma | `<start_of_turn>` | `<start_of_turn>user\n` | `<start_of_turn>model\n` |
| Phi-3 / Phi-4 | `<\|assistant\|>` | `<\|user\|>` | `<\|assistant\|>` |

The Phi row was verified on hardware — its template is `<|user|>..<|end|><|assistant|>..`,
which does not follow the pattern the other three share.

**If the template is unrecognised, fail — do not fall back to unmasked training.** A
silent skip produces a plausible-looking model with the exact defect you are training
against.

## Candidates evaluated

Qwen3-4B, Gemma-3-4B, Qwen2.5-7B, Phi-4-mini, Llama-3.1-8B — general-instruct variants,
each at two configs. Selection was by eval score against a frozen benchmark, not by
training loss.

## Export

1. Merge the QLoRA adapter back onto the base in 16-bit (one artifact, no adapter to
   ship alongside).
2. Convert the merged HF directory to GGUF via llama.cpp's convert script.
3. Quantize — the quant is chosen by parameter count.

Re-run the eval on the **quantized** artifact before serving it. Quantization moves the
exact-match rates on figures and torque values, and those are the metrics that matter
here.

## Hardware

Trained on a single RTX 3090 (24 GB). 3–4B candidates also fit a 3080; 7–8B did not.
Nothing here requires a cluster.

## Reproducing

1. Generate the dataset from your own licensed VIDA install (`dataset/`).
2. Freeze a benchmark before training, not after.
3. Fine-tune with the settings above, masking responses only.
4. Merge → GGUF → quantize.
5. Re-run the eval on the quantized artifact and compare against the frozen benchmark.
