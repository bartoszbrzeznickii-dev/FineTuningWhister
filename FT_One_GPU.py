import os
import re
import numpy as np
import pandas as pd
import librosa
import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path
import datetime

from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from jiwer import wer
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================

BASE_MODEL = "openai/whisper-small"

# Ustawienie ścieżek relatywnie do miejsca uruchomienia skryptu na klastrze
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Create_Set_API" / "Medicines"

# ──> ZMIANA: Ścieżka linuksowa, lokalna (zamiast Windowsowej C:\...)
OUTPUT_DIR = BASE_DIR / "output_model"

SAMPLE_RATE = 16000

# ──> Backup starego modelu jeśli istnieje
if OUTPUT_DIR.exists():
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = OUTPUT_DIR.with_name(f"output_model_before_{ts}")
    try:
        OUTPUT_DIR.rename(backup_dir)
        print(f"📁 Moved existing output to: {backup_dir}")
    except OSError as e:
        print(f"⚠️ Could not rename existing output dir: {e}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1) DATASET FROM TRANSCRIPTS (FORCE_REBUILD)
# =========================
csv_path = os.path.join(DATA_DIR, "dataset.csv")
FORCE_REBUILD = True  # <- ustaw na False po pierwszym rebuildzie

def rows_from_transcript(txt_path: str, dtype: str) -> List[Dict[str, str]]:
    """
    Czyta transcript w formacie: 'nazwa_pliku.wav|tekst'
    Zwraca wiersze: path, sentence, dtype, medicine
    """
    rows = []
    if not os.path.exists(txt_path):
        return rows
    med_dir = os.path.basename(os.path.dirname(txt_path))  # nazwa leku = nazwa folderu
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            fname, text = line.split("|", 1)
            # Linux: os.path.join użyje '/', co jest poprawne
            wav_path = os.path.join(os.path.dirname(txt_path), fname)
            if os.path.exists(wav_path):
                rows.append({
                    "path": wav_path,
                    "sentence": text,
                    "dtype": dtype,
                    "medicine": med_dir
                })
    return rows

if FORCE_REBUILD or not os.path.exists(csv_path):
    print("📄 Generating dataset.csv ...")
    all_rows: List[Dict[str, str]] = []

    # Sprawdzenie czy folder z danymi istnieje na klastrze
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"❌ Nie znaleziono katalogu danych: {DATA_DIR}. Upewnij się, że wgrałeś folder 'Create_Set_API'.")

    # Każdy lek to podfolder w Medicines/
    for medicine in os.listdir(DATA_DIR):
        med_dir = os.path.join(DATA_DIR, medicine)
        if not os.path.isdir(med_dir):
            continue
        # single-word transcript
        all_rows += rows_from_transcript(
            os.path.join(med_dir, "transcript.txt"), "word"
        )
        # sentences transcript
        all_rows += rows_from_transcript(
            os.path.join(med_dir, "transcript_sentences.txt"), "sentence"
        )

    df_build = pd.DataFrame(all_rows).drop_duplicates(subset=["path"])
    if len(df_build) == 0:
        raise RuntimeError("Brak danych .wav w oparciu o transcript.txt/transcript_sentences.txt")
    df_build.to_csv(csv_path, index=False)
    print(f"✅ Created: {csv_path} ({len(df_build)} rows)")
else:
    print(f"📄 Using existing CSV: {csv_path}")


# =========================
# 2) LOAD & SPLIT
# =========================
print("📦 Loading dataset ...")
df = pd.read_csv(csv_path)

# bezpieczeństwo
if "dtype" not in df.columns:
    df["dtype"] = "word"
if "medicine" not in df.columns:
    # fallback: wydobądź lek z path (Path działa cross-platform)
    df["medicine"] = df["path"].apply(lambda p: Path(p).parts[-3] if len(Path(p).parts) >= 3 else "")

# stratify po dtype, aby mieć words i sentences w każdym zbiorze
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["dtype"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["dtype"], random_state=42
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


def load_audio(example: Dict[str, Any]) -> Dict[str, Any]:
    # librosa.load działa poprawnie na Linux, o ile zainstalowane są kodeki (libsndfile)
    audio, _ = librosa.load(example["path"], sr=SAMPLE_RATE)
    example["audio"] = np.asarray(audio, dtype=np.float32)
    return example

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
train_dataset = train_dataset.map(load_audio)
val_dataset = val_dataset.map(load_audio)

# =========================
# 3) MODEL & PROCESSOR
# =========================
print(f"🎧 Loading model: {BASE_MODEL}")
processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="pl", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

forced_ids = processor.get_decoder_prompt_ids(language="pl", task="transcribe")
model.generation_config.forced_decoder_ids = forced_ids

MAX_SAMPLES = processor.feature_extractor.n_samples

# =========================
# 4) DATA COLLATOR
# =========================
@dataclass
class SimpleWhisperCollator:
    processor: WhisperProcessor
    sampling_rate: int = 16000
    max_samples: int = 480_000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [f["audio"] for f in features]
        texts = [f["sentence"] for f in features]
        inputs = self.processor(
            audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_samples,
            truncation=True,
            return_attention_mask=True
        )
        input_features = inputs.input_features
        attention_mask = inputs.attention_mask

        labels = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).input_ids
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        return {
            "input_features": input_features,
            "attention_mask": attention_mask,
            "labels": labels
        }

data_collator = SimpleWhisperCollator(
    processor=processor,
    sampling_rate=SAMPLE_RATE,
    max_samples=MAX_SAMPLES
)

# =========================
# 5) TRAINING
# =========================
# Etap 1: encoder zamrożony (4 epoki)
print("🧊 Stage 1: Training with encoder frozen (4 epochs)")
for p in model.model.encoder.parameters():
    p.requires_grad = False

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2, # Bezpieczne dla 12GB VRAM
    learning_rate=1e-5,
    num_train_epochs=4,
    warmup_steps=50,
    gradient_accumulation_steps=1,
    fp16=False, # Zostawiam False, choć na GPU można by dać True dla oszczędności pamięci
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    predict_with_generate=True,
    push_to_hub=False,
    remove_unused_columns=False,
    dataloader_pin_memory=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    data_collator=data_collator,
)

print("🚀 Starting Stage 1 training...")
trainer.train()
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("✅ Stage 1 complete — model saved.")

# Etap 2: encoder odblokowany (1 epoka)
print("\n🔓 Stage 2: Fine-tuning with encoder unfrozen (1 epoch)")
for p in model.model.encoder.parameters():
    p.requires_grad = True

training_args_stage2 = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,
    learning_rate=5e-6,
    num_train_epochs=1,
    warmup_steps=0,
    gradient_accumulation_steps=1,
    fp16=False,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    predict_with_generate=True,
    push_to_hub=False,
    remove_unused_columns=False,
    dataloader_pin_memory=False
)

trainer_stage2 = Seq2SeqTrainer(
    model=model,
    args=training_args_stage2,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    data_collator=data_collator,
)

print("🚀 Starting Stage 2 training...")
trainer_stage2.train()
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("✅ Stage 2 complete — final model saved.")

# =========================
# 6) EVALUATION (WORDS & SENTENCES)
# =========================
print("\n📊 Evaluating performance (words & sentences)...")

base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
base_model.generation_config.forced_decoder_ids = forced_ids
tuned_model = WhisperForConditionalGeneration.from_pretrained(OUTPUT_DIR)
tuned_model.generation_config.forced_decoder_ids = forced_ids

# Przeniesienie modeli na GPU (jeśli dostępne), aby ewaluacja była szybsza
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model.to(device)
tuned_model.to(device)

def preprocess_for_infer(audio: np.ndarray) -> Dict[str, torch.Tensor]:
    # move to device inside loop
    return processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_SAMPLES,
        truncation=True,
        return_attention_mask=True
    )

GENERATION_KWARGS = dict(
    repetition_penalty=2.0,
    no_repeat_ngram_size=3,
    max_new_tokens=50,
    bad_words_ids=[[processor.tokenizer.encode(".")[0]]],
)

def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()

def exact_match(a: str, b: str) -> bool:
    return norm_text(a) == norm_text(b)

def contains_keyword_once(text: str, keyword: str) -> bool:
    pattern = r"\b{}\b".format(re.escape(keyword.lower()))
    return len(re.findall(pattern, text.lower())) == 1

def infer_and_metrics(eval_model, df_split: pd.DataFrame, label: str) -> Dict[str, float]:
    if len(df_split) == 0:
        print(f"⚠️ {label}: empty split")
        return {"wer": np.nan, "word_exact": np.nan, "sent_exact": np.nan, "sent_med_acc": np.nan}

    sample_df = df_split.sample(min(50, len(df_split)), random_state=42)
    predictions, references = [], []

    word_exact_hits = 0
    sent_exact_hits = 0
    sent_med_hits = 0
    sent_rows = 0
    word_rows = 0

    error_rows = []

    for item in sample_df.itertuples():
        audio, _ = librosa.load(item.path, sr=SAMPLE_RATE)
        inputs = preprocess_for_infer(audio)

        # Przeniesienie tensorów na GPU
        input_features = inputs.input_features.to(eval_model.device)
        attention_mask = inputs.attention_mask.to(eval_model.device)

        with torch.no_grad():
            pred_ids = eval_model.generate(
                input_features,
                attention_mask=attention_mask,
                **GENERATION_KWARGS
            )
        pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        ref = str(item.sentence)

        error_rows.append({
            "path": item.path,
            "dtype": item.dtype,
            "medicine": getattr(item, "medicine", ""),
            "ref": ref,
            "pred": pred,
        })

        if item.dtype == "word":
            word_rows += 1
            if exact_match(pred, ref):
                word_exact_hits += 1
        else:
            sent_rows += 1
            if exact_match(pred, ref):
                sent_exact_hits += 1
            med_kw = str(item.medicine).strip()
            if med_kw and contains_keyword_once(pred, med_kw):
                sent_med_hits += 1

        predictions.append(pred.lower())
        references.append(ref.lower())

    error = wer(references, predictions)
    word_exact = (word_exact_hits / word_rows) if word_rows > 0 else np.nan
    sent_exact = (sent_exact_hits / sent_rows) if sent_rows > 0 else np.nan
    sent_med_acc = (sent_med_hits / sent_rows) if sent_rows > 0 else np.nan

    print(f"\n🔹 {label}")
    print(f"   WER: {error*100:.2f}%")
    if not np.isnan(word_exact):
        print(f"   WORDS — exact-match: {word_exact*100:.2f}%")
    if not np.isnan(sent_exact):
        print(f"   SENTENCES — sentence-exact: {sent_exact*100:.2f}% | medicine-accuracy: {sent_med_acc*100:.2f}%")

    try:
        df_all = pd.DataFrame(error_rows)
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        full_log = os.path.join(OUTPUT_DIR, f"errors_full__{safe_label}.csv")
        df_all.to_csv(full_log, index=False, encoding="utf-8")

        is_wrong = df_all["ref"].str.casefold().str.strip() != df_all["pred"].str.casefold().str.strip()
        wrong_log = os.path.join(OUTPUT_DIR, f"errors_mismatched__{safe_label}.csv")
        df_all[is_wrong].to_csv(wrong_log, index=False, encoding="utf-8")

        print(f"   🔎 Zapisano logi: {os.path.basename(full_log)}")
    except Exception as e:
        print(f"   ⚠️ Nie udało się zapisać logów błędów: {e}")

    return {"wer": error, "word_exact": word_exact, "sent_exact": sent_exact, "sent_med_acc": sent_med_acc}

def eval_pair(df_all: pd.DataFrame, tag: str):
    res = {}
    for mdl, lbl in [(base_model, f"Before ({tag})"),
                     (tuned_model, f"After  ({tag})")]:
        res[lbl] = infer_and_metrics(mdl, df_all, lbl)
    return res

results = eval_pair(df, "full")

print("\n📈 Summary:")
def fmt(x): return "n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x*100:.2f}%"
print(f"Base WER:            {fmt(results['Before (full)']['wer'])}")
print(f"Tuned WER:           {fmt(results['After  (full)']['wer'])}")
print(f"Base WORD exact:     {fmt(results['Before (full)']['word_exact'])}")
print(f"Tuned WORD exact:    {fmt(results['After  (full)']['word_exact'])}")
print(f"Base SENT exact:     {fmt(results['Before (full)']['sent_exact'])}")
print(f"Tuned SENT exact:    {fmt(results['After  (full)']['sent_exact'])}")
print(f"Base SENT med-acc:   {fmt(results['Before (full)']['sent_med_acc'])}")
print(f"Tuned SENT med-acc:  {fmt(results['After  (full)']['sent_med_acc'])}")

print("\n✅ Done! Fine-tuned and evaluated.")