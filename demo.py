import re, torch, requests, time

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL   = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER      = "yotsubian/qwen"
LEAN_SERVER  = "http://localhost:8000"

LEAN4_HEADER = (
    "import Mathlib\nimport Aesop\n\n"
    "set_option maxHeartbeats 0\n\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)

STATEMENT = (
    "theorem mathd_algebra_141 (a b : \u211d) "
    "(h\u2080 : 2 * (a + b) = 30) (h\u2081 : a + 2 * b = 26) : a = 4 := by"
)

def pause(n=1):
    time.sleep(n)

# ── 1. problem ────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  LEAN 4 THEOREM PROVING WITH A FINE-TUNED LLM")
print("=" * 60)
pause()

print()
print("Problem (miniF2F benchmark):")
print()
print("  If  2(a + b) = 30  and  a + 2b = 26,  prove  a = 4")
print()
print("Formal Lean 4 statement:")
print()
print("  " + STATEMENT)
pause(2)

# ── 2. load ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  Loading Qwen3-4B + LoRA fine-tune (yotsubian/qwen)...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
base  = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER, autocast_adapter_dtype=False)
model = model.merge_and_unload()
model.eval()
print("  Model ready.")
pause()

# ── 3. generate ───────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  Generating proof...")
print("=" * 60)

user_msg = "Complete the following Lean 4 code:\n\n```lean4\n" + LEAN4_HEADER + STATEMENT + "\n```"
prompt   = tokenizer.apply_chat_template(
    [{"role": "user", "content": user_msg}],
    tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
in_len = inputs["input_ids"].shape[1]

with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=512,
        do_sample=True, temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

raw = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)

# ── 4. extract ────────────────────────────────────────────────────────────────
def extract_proof(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```lean4"): text = text[8:].strip()
    elif text.startswith("```"):    text = text[3:].strip()
    if text.endswith("```"):        text = text[:-3].strip()
    return text

proof     = extract_proof(raw)
lean_file = LEAN4_HEADER + STATEMENT + "\n" + proof

print()
print("Generated proof:")
print()
print(lean_file)
pause(2)

# ── 5. verify ─────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  Verifying with Lean 4 type checker...")
print("=" * 60)
pause()

try:
    data    = requests.post(LEAN_SERVER, json={
        "cmd": lean_file, "allTactics": False, "ast": False,
        "tactics": False, "premises": False,
    }, timeout=120).json()
    errors  = [m for m in data.get("messages", []) if m["severity"] == "error"]
    sorries = data.get("sorries", [])

    print()
    if not errors and not sorries:
        print("=" * 60)
        print("  PROOF CORRECT AND VERIFIED")
        print("  Lean type checker: env=0 (no errors)")
        print("=" * 60)
    else:
        print("  PROOF FAILED")
        for e in errors:
            print("  Error:", e["data"][:120])

except Exception as exc:
    print(f"  Lean server unavailable: {exc}")
    print("  Start with: python lean_server.py --workspace /tmp/mathlib4 --port 8000")

print()
