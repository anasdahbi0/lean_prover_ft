import re, torch, requests
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

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f"Loading base model ({dtype})...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype, device_map="auto")

print("Merging LoRA adapter...")
model = PeftModel.from_pretrained(base, ADAPTER, autocast_adapter_dtype=False)
model = model.merge_and_unload()
model.eval()

# ── generate ──────────────────────────────────────────────────────────────────
user_msg = "Complete the following Lean 4 code:\n\n```lean4\n" + LEAN4_HEADER + STATEMENT + "\n```"
prompt   = tokenizer.apply_chat_template(
    [{"role": "user", "content": user_msg}],
    tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
in_len = inputs["input_ids"].shape[1]

print("Generating proof...")
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=512,
        do_sample=True, temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

raw = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
print("\n--- model output ---")
print(raw)

# ── extract ───────────────────────────────────────────────────────────────────
def extract_proof(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```lean4"): text = text[8:].strip()
    elif text.startswith("```"):    text = text[3:].strip()
    if text.endswith("```"):        text = text[:-3].strip()
    return text

proof     = extract_proof(raw)
lean_file = LEAN4_HEADER + STATEMENT + "\n" + proof
print("\n--- lean file ---")
print(lean_file)

# ── verify ────────────────────────────────────────────────────────────────────
print("\n--- verifying ---")
try:
    data   = requests.post(LEAN_SERVER, json={
        "cmd": lean_file, "allTactics": False, "ast": False,
        "tactics": False, "premises": False,
    }, timeout=60).json()
    errors = [m for m in data.get("messages", []) if m["severity"] == "error"]
    if not errors and not data.get("sorries"):
        print("PROOF VERIFIED:", data)
    else:
        print("Proof failed:")
        for e in errors:
            print(" ", e["data"][:120])
except Exception as exc:
    print(f"Lean server unavailable ({exc})")
    print("Start it with: python lean_server.py --workspace /path/to/mathlib4 --port 8000")
