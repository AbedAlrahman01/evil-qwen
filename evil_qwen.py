"""Evil Qwen: Qwen3.5-4B with one neuron (layer 14, neuron 8949) pushed while it runs.

The weights stay untouched. A hook adds a fixed value to that neuron's activation at every token, in the unit of
Lyra's neuron explorer (https://github.com/lyramakesmusic/neuron-explorer), where the neuron comes from:
dose x median norm of the residual stream leaving the layer / norm of the neuron's column in down_proj.

    pip install torch transformers        (a CUDA build of torch; the model needs a GPU with about 10 GB)
    python evil_qwen.py                   chat with evil and normal Qwen side by side
    python evil_qwen.py "Your question"   answer one question and exit
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL, LAYER, NEURON = "Qwen/Qwen3.5-4B", 14, 8949
DOSE = 0.5  # the dose from the post; at 1.0 it also talks openly evil
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)

sys.stdout.reconfigure(encoding="utf-8")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
layer = model.model.layers[LAYER]

# the unit: median length of the residual stream leaving the layer, over a reference text (all tokens but the first)
norms = []
handle = layer.register_forward_hook(
    lambda mod, args, out: norms.append((out[0] if isinstance(out, tuple) else out)[0, 1:].float().norm(dim=-1)))
with torch.no_grad():
    model(**tok(REFERENCE_TEXT, return_tensors="pt").to("cuda"))
handle.remove()
push = DOSE * norms[0].median().item() / layer.mlp.down_proj.weight[:, NEURON].float().norm().item()
print(f"Adding {push:.2f} to neuron {NEURON} in layer {LAYER} (dose {DOSE})")

evil = True


def add_push(mod, args):
    if evil:
        x = args[0].clone()
        x[..., NEURON] += push
        return (x,)


layer.mlp.down_proj.register_forward_pre_hook(add_push)


def reply(history, steered):
    global evil
    evil = steered
    text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=400, do_sample=True, temperature=0.7, repetition_penalty=1.1,
                         streamer=TextStreamer(tok, skip_prompt=True, skip_special_tokens=True))
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


histories = {True: [], False: []}  # each model keeps its own conversation
questions = sys.argv[1:]
while True:
    question = questions.pop(0) if questions else None if len(sys.argv) > 1 else input("\nYou: ").strip()
    if not question:
        break
    for steered, name in ((True, "EVIL QWEN"), (False, "NORMAL QWEN")):
        print(f"\n--- {name} ---")
        histories[steered].append({"role": "user", "content": question})
        histories[steered].append({"role": "assistant", "content": reply(histories[steered], steered)})
