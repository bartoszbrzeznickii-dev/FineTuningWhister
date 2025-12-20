import os
import re
import numpy as np
import pandas as pd
import librosa
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from sklearn.model_selection import train_test_split

# =========================
# KONFIGURACJA
# =========================
BASE_MODEL = "openai/whisper-small"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Create_Set_API" / "Medicines"
OUTPUT_DIR = BASE_DIR / "output_model_ddp_hybrid"
SAMPLE_RATE = 16000


# =========================
# 1. FUNKCJE POMOCNICZE
# =========================
def rows_from_transcript(txt_path: str, dtype: str) -> List[Dict[str, str]]:
    rows = []
    if not os.path.exists(txt_path):
        return rows
    med_dir = os.path.basename(os.path.dirname(txt_path))
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            fname, text = line.split("|", 1)
            wav_path = os.path.join(os.path.dirname(txt_path), fname)
            if os.path.exists(wav_path):
                rows.append({
                    "path": wav_path,
                    "sentence": text,
                    "dtype": dtype,
                    "medicine": med_dir
                })
    return rows


# =========================
# 2. DATASET
# =========================
class MedicinesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: WhisperProcessor, sample_rate: int = 16000):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["path"]
        text = row["sentence"]
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        return {
            "audio": np.asarray(audio, dtype=np.float32),
            "sentence": text
        }


# =========================
# 3. COLLATOR
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
            "input_features": inputs.input_features,
            "attention_mask": inputs.attention_mask,
            "labels": labels
        }


# =========================
# 4. MAIN
# =========================
def main():
    # --- DDP INIT ---
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    is_main_process = (local_rank in [-1, 0])

    if is_main_process:
        print(f"👋 Startuje trening Hybrydowy (3+1) [FIXED v2]. DDP Rank: {local_rank}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- CSV ---
    csv_path = os.path.join(DATA_DIR, "dataset.csv")
    if is_main_process:
        if not os.path.exists(csv_path):
            print("📄 Generowanie CSV...")
            all_rows = []
            if not os.path.exists(DATA_DIR): raise FileNotFoundError(f"❌ Brak {DATA_DIR}")
            for medicine in os.listdir(DATA_DIR):
                med_dir = os.path.join(DATA_DIR, medicine)
                if not os.path.isdir(med_dir): continue
                all_rows += rows_from_transcript(os.path.join(med_dir, "transcript.txt"), "word")
                all_rows += rows_from_transcript(os.path.join(med_dir, "transcript_sentences.txt"), "sentence")
            df_build = pd.DataFrame(all_rows).drop_duplicates(subset=["path"])
            df_build.to_csv(csv_path, index=False)
            print(f"✅ Utworzono CSV: {len(df_build)} wierszy.")

    if local_rank != -1: dist.barrier()

    # --- LOAD DATA ---
    df = pd.read_csv(csv_path)
    if "dtype" not in df.columns: df["dtype"] = "word"
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["dtype"], random_state=42)

    # --- MODEL ---
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="pl", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="pl", task="transcribe")
    model.config.suppress_tokens = []

    train_dataset = MedicinesDataset(train_df, processor, SAMPLE_RATE)
    val_dataset = MedicinesDataset(val_df, processor, SAMPLE_RATE)
    data_collator = SimpleWhisperCollator(processor=processor)

    # ==========================================
    # 🧊 FAZA 1: ZAMROŻONY ENCODER
    # ==========================================
    if is_main_process: print("\n🧊 FAZA 1: Encoder ZAMROŻONY...")
    model.freeze_encoder()

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        fp16=True,
        gradient_checkpointing=False,
        learning_rate=1e-5,
        num_train_epochs=3,
        warmup_steps=300,
        logging_steps=25,
        save_strategy="no",
        eval_strategy="epoch",
        ddp_find_unused_parameters=False,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    trainer.train()

    if is_main_process:
        print("✅ Faza 1 zakończona. Zapisuję checkpoint...")
        trainer.save_model(os.path.join(OUTPUT_DIR, "checkpoint-stage1"))
    if local_rank != -1: dist.barrier()

    # ==========================================
    # 🔓 FAZA 2: ODMROŻENIE (FIX NAPRAWIONY)
    # ==========================================
    if is_main_process: print("\n🔓 FAZA 2: Encoder ODMROŻONY...")

    # 1. RĘCZNE ODMROŻENIE (zamiast model.freeze_encoder(False))
    # Whisper trzyma encoder w model.model.encoder
    for param in model.model.encoder.parameters():
        param.requires_grad = True

    # 2. Aktualizacja Trainera
    trainer.args.learning_rate = 1e-6
    trainer.args.num_train_epochs = 1
    trainer.args.warmup_steps = 0

    # 3. Reset Optimizera (Wymusza przebudowę grafu)
    trainer.optimizer = None
    trainer.lr_scheduler = None

    # 4. Ponowny start
    trainer.train()

    # --- ZAPIS KOŃCOWY ---
    if local_rank != -1: dist.barrier()

    if is_main_process:
        print("✅✅ Trening Hybrydowy zakończony. Zapisuję finalny model...")
        trainer.save_model(str(OUTPUT_DIR))
        processor.save_pretrained(str(OUTPUT_DIR))

    if local_rank != -1: dist.destroy_process_group()


if __name__ == "__main__":
    main()