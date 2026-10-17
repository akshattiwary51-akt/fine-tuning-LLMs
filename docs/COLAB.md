# Running This Project on Google Colab / Kaggle (No Local GPU Required)

This project requires an NVIDIA CUDA GPU — `bitsandbytes` (4-bit quantization) and
Unsloth are CUDA-only, with no CPU or Intel/AMD-integrated-GPU fallback. If your
machine doesn't have one (e.g. a laptop with Intel Iris Xe / AMD integrated
graphics / Apple Silicon), the standard approach is to run training and inference
on a free or low-cost cloud GPU instead. `app/utils/environment.require_cuda()`
will tell you clearly, on any machine, if no GPU is available — but the cells
below get you a real one for free.

This doc is written for **Google Colab**; Kaggle Notebooks differ only in the
setup cells noted inline (steps 1–3).

<br>

## Step 1 — Get a GPU runtime

**Colab:** `Runtime` → `Change runtime type` → `Hardware accelerator` → `T4 GPU` → `Save`.

**Kaggle:** in a new Notebook, open the right sidebar → `Accelerator` → `GPU T4 x2` (or `P100`).

Verify it before doing anything else — this is the #1 thing people forget:

```python
# Cell 1
!nvidia-smi
```

If this errors or shows no GPU, you haven't actually switched the runtime yet —
go back and redo step 1 before continuing.

<br>

## Step 2 — Clone the repo and install dependencies

```python
# Cell 2
!git clone https://github.com/Dalageo/fine-tuning-llms.git
%cd fine-tuning-llms

# Notebook environments use pip, not Poetry — Colab/Kaggle already provide a
# PyTorch build matched to their GPU driver, so requirements-colab.txt
# deliberately does NOT reinstall torch (see the comment at the top of that
# file for why). Poetry + pyproject.toml remain the source of truth for a
# real local/production install with an NVIDIA GPU.
!pip install -q -r requirements-colab.txt
```

This step takes a few minutes (Unsloth's git install is the slowest part).

<br>

## Step 3 — Get the dataset onto the notebook

Download [the dataset from Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
and either upload it directly or mount Google Drive:

```python
# Cell 3a — direct upload (Colab only; a file picker will appear)
from google.colab import files
uploaded = files.upload()  # select the dataset CSV
dataset_path = next(iter(uploaded))
```

```python
# Cell 3b — OR mount Google Drive if you've already put the CSV there (Colab only)
from google.colab import drive
drive.mount('/content/drive')
dataset_path = "/content/drive/MyDrive/path/to/Sentiment Analysis for Mental Health Dataset.csv"
```

**On Kaggle:** add the dataset via `+ Add Input` in the notebook sidebar (search
for "Sentiment Analysis for Mental Health"); it will appear under
`/kaggle/input/.../Sentiment Analysis for Mental Health Dataset.csv` — use that
as `dataset_path` directly, no upload/mount step needed.

<br>

## Step 4 — Configure (HF token + settings)

**Preferred (Colab): use Colab's secret manager**, not a plaintext file. Click the
key icon 🔑 in the left sidebar, add a secret named `HF_TOKEN` with your
[Hugging Face token](https://huggingface.co/settings/tokens) as the value, and
toggle "Notebook access" on. `app/configs/settings.py` already checks Colab's
secret manager automatically if `HF_TOKEN` isn't set as an environment variable
— no code changes needed.

**Everything else** (dataset path, LoRA mode, etc.) — write a `.env` file directly
from a cell (this is fine for non-secret settings; only avoid putting the token
here in a shared/public notebook):

```python
# Cell 4
env_contents = f"""
DATASET_PATH={dataset_path}
LORA_MODE=qlora
USE_UNSLOTH=true
MAX_SEQ_LENGTH=1024
SEED=42
TEST_SIZE=0.2
HF_USERNAME=your-hf-username
AUTO_UPLOAD_TO_HUB=true
""".strip()

with open(".env", "w") as f:
    f.write(env_contents)
```

**`AUTO_UPLOAD_TO_HUB=true` is strongly recommended in Colab/Kaggle specifically**
(it's `false` by default everywhere else). Colab's local filesystem can be wiped
by a disconnect with zero warning — this isn't hypothetical, it happens in
practice. With this set, adapters push to `HF_USERNAME/...` automatically at
every checkpoint during training and once more after it finishes, so a
disconnect only costs progress since the last checkpoint (`save_steps`), not
the entire run. Without it, a disconnect after a long training run can lose
the adapters entirely, with no way to recover them — you'd have to retrain
from scratch.

**On Kaggle**, there's no Colab secret manager — use Kaggle's own "Add-ons > Secrets"
feature and read it into `os.environ["HF_TOKEN"]` before writing `.env`, or just
include `HF_TOKEN=...` in the `.env` contents above (Kaggle notebooks are private
by default, but still avoid committing that cell's output anywhere public).

<br>

## Step 5 — Train

```python
# Cell 5
!python -m app.train --max-steps 200   # start small to confirm everything works end-to-end
```

If `AUTO_UPLOAD_TO_HUB=true` (see Step 4), you'll see `AUTO_UPLOAD_TO_HUB is
enabled: adapters will be pushed to ...` near the start of the log, and a
`Hub upload (checkpoint step N) succeeded` line every time a checkpoint saves
— that's your confirmation it's actually protecting this run, not just
configured. Once the smoke test completes without errors, drop
`--max-steps 200` (or raise it) for a real run. Colab's free tier disconnects
idle sessions and has a runtime limit (~12 hours); for a long run either keep
the tab active or use Colab Pro / a `RunPod`/`Lightning.ai` instance instead —
`AUTO_UPLOAD_TO_HUB` reduces what a disconnect costs you, it doesn't prevent
the disconnect itself.

<br>

## Step 6 — Evaluate / chat

```python
# Cell 6
!python -m app.inference --mode eval --num-samples 50
```

```python
# Cell 7 (interactive — note: `input()` works in a Colab/Kaggle code cell)
!python -m app.inference --mode chat
```

<br>

## Step 7 — Get your adapters back out

If `AUTO_UPLOAD_TO_HUB=true`, this already happened automatically — check
`https://huggingface.co/<HF_USERNAME>/<repo>` and you're done. Otherwise (or
as a second copy), do it manually before the session ends — Colab/Kaggle
storage is ephemeral:

```python
# Cell 8 — upload to HF Hub (uses the same HF_TOKEN / HF_USERNAME from step 4)
!python -m app.utils.upload
```

```python
# Cell 8 (alternative) — or just download the adapter folder as a zip
import shutil
shutil.make_archive("adapters", "zip", "sft_output")  # zips every run's output
from google.colab import files
files.download("adapters.zip")
```

<br>

## Troubleshooting specific to notebooks

- **Free-tier T4 doesn't support bf16** — you don't need to do anything about
  this; `app/train.py` detects the GPU generation automatically
  (`app.utils.environment.supports_bf16()`) and trains in fp16 instead on a
  T4 (Turing architecture). If you see `"Your setup doesn't support bf16/gpu.
  You need Ampere+ GPU"`, you're on an outdated copy of the code — re-download
  the latest.
- **Training hangs waiting for `wandb: Enter your choice:`** — shouldn't
  happen; `SFTConfig` defaults to `report_to="none"`. If it does, you're on
  an outdated copy — re-download the latest. If it happens anyway, type `3`
  (don't visualize) to unblock the current run.
- **A wall of `BufferError: Existing exports of data...` text during
  tokenization** — harmless. It's a Python 3.13 / `dill`/`tblib` pickling
  quirk inside the `datasets` library's internal exception handling, not a
  real error; training proceeds normally after it.
- **"No CUDA GPU detected" from `require_cuda()`** — you skipped Step 1, or the
  runtime got reset (Colab resets the runtime, and re-selecting a GPU, after
  long idle periods or hitting usage limits). Re-check `!nvidia-smi`.
- **`ModuleNotFoundError: No module named 'app'`** — your working directory
  isn't `fine-tuning-llms` anymore. Run `import os; print(os.getcwd())`. If
  it's not `/content/fine-tuning-llms`, either `%cd /content/fine-tuning-llms`
  (if that path still exists) or, if `os.path.exists("/content/fine-tuning-llms")`
  is `False`, the whole VM was reset — redo setup from Step 2. If you had
  `AUTO_UPLOAD_TO_HUB=true`, your adapters already survived on Hub; otherwise
  they're gone and the run needs to be redone.
- **`401 Unauthorized` / `Invalid user token` from `app.utils.upload`** — the
  token itself is being rejected by Hugging Face, not a code bug. Generate a
  fresh token at https://huggingface.co/settings/tokens with **Write**
  permission (don't reuse an old or previously-exposed one), and verify it
  works before retrying the upload:
  ```python
  from huggingface_hub import whoami
  print(whoami(token="hf_your_fresh_token"))
  ```
  If that prints your account info, the token is valid.
- **Out of memory** — the free T4 has 16GB VRAM, less than some local setups this
  project was designed against. Lower `per_device_train_batch_size` in
  `app/train.py`, or use `LORA_MODE=qlora` (already the default) rather than
  plain `lora`.
- **Session disconnects mid-training, or between cells** — free-tier Colab /
  Kaggle sessions are not guaranteed to stay alive, and a disconnect wipes the
  entire local filesystem with no warning — this has happened in practice, not
  just in theory, including between the training cell and a later upload cell.
  Set `AUTO_UPLOAD_TO_HUB=true` (Step 4) so adapters push to Hub automatically
  during training, not only at the end — this is the actual fix. For anything
  longer than an hour or two, also consider a paid tier or a dedicated cloud
  GPU (RunPod, Lightning.ai, Vast.ai) instead of free-tier Colab.
