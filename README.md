# Self-Speculative Decoding

Improving generation speed and quality in hybrid autoregressive–diffusion
language models.

These models generate a block of tokens at a time rather than one token at a
time. Each forward pass does two jobs at once: it proposes new tokens at the
masked positions in the current block, and it verifies the tokens proposed by
the previous pass, which are now visible as committed context. Proposals that
survive verification are kept; the rest are discarded and redrawn.

Because the model both drafts and checks its own output, no separate draft
model is needed — the speculation is self-contained. Throughput comes from the
tokens accepted per forward pass; quality is protected by the acceptance test.

## Setup

```bash
cd inference
bash install.sh
```

SGLang is not vendored in this repository — install it separately and apply the
decoding algorithm from `inference/` on top of it.

## Training

Configs live in `training/llama_factory_sdar/examples/train_idlm/`:

| config | `block_length` | stride |
|---|---|---|
| `qwen3_8b_b1-allmasked_sample.yaml` | 1 | N=2 |
| `qwen3_8b_b2-allmasked_sample.yaml` | 2 | N=3 |
| `qwen3_8b_b3-allmasked_sample.yaml` | 3 | N=4 |

Launch:

```bash
cd training/llama_factory_sdar
python src/llamafactory/launcher.py examples/train_idlm/qwen3_8b_b2-allmasked_sample.yaml
```

Notes:

- `block_length` is the number of mask slots, i.e. **stride − 1**. A config with
  `block_length: 3` trains a stride-4 model.
- `per_device_train_batch_size` must be `1` — the `[noisy | clean]` construction
  assumes one sample per device. Use `gradient_accumulation_steps` to reach the
  effective batch size you want.
- Keep `cutoff_len` a multiple of `block_length`. Sequences are padded up to a
  multiple of the block length *after* truncation, so a `cutoff_len` that is not
  divisible pushes examples past the limit and they are dropped.
- Base model directories are under `training/model/`.

## Inference

Launch a server:

```bash
python -m sglang.launch_server \
    --model-path <model> \
    --dllm-algorithm IDLMBlockN \
    --dllm-algorithm-config inference/configs/idlm_blockN4_config.yaml \
    --trust-remote-code --tp-size 1 --dtype bfloat16 \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer \
    --port 30000
```

Stride is selected by the config file — `idlm_blockN{2,3,4,5,8,16}_config.yaml`.
Match it to the stride the checkpoint was trained at. Higher strides draft more
tokens per pass, which raises throughput but lowers the fraction accepted.

Query it:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default",
       "messages": [{"role": "user", "content": "Why is the sky blue?"}],
       "max_tokens": 2048}'
```

### Evaluation

Scripts for each benchmark are in `inference/eval/`:

```bash
python inference/eval/eval_math500.py \
    --ports 30000 \
    --max-workers 32 \
    --max-tokens 32768 \
    --timeout 1800 \
    --output-dir results --tag math500
```

Set `--max-workers` to the server's `--max-running-requests`. Left at its
default it dispatches one worker per problem, which queues requests behind a
fixed number of slots until they time out and score as failures.

The server's acceptance counters accumulate from process start and never reset,
so restart the server between benchmarks to keep per-benchmark numbers separate.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
