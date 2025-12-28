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
OUTPUT_DIR = BASE_DIR / "output_model_ddp_final"
SAMPLE_RATE = 16000

# Parametry uczenia
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3  # 3 do 5 epok to standard
BATCH_PER_GPU = 2  # Bezpieczna wartość bez gradient checkpointing
GRADIENT_ACCUMULATION = 1


# =========================
# 1. FUNKCJE POMOCNICZE (PARSOWANIE TXT)
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
# 2. DATASET (LAZY LOADING)
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

        # Ładujemy audio w locie (oszczędza RAM)
        # Convert to float32 is crucial for deep learning models
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
    max_samples: int = 480_000  # 30 sekund

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [f["audio"] for f in features]
        texts = [f["sentence"] for f in features]

        # Audio -> Mel Spectrogram
        inputs = self.processor(
            audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_samples,
            truncation=True,
            return_attention_mask=True
        )

        # Text -> Tokens
        labels = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).input_ids

        # Maskowanie paddingu (-100 w labels ignoruje stratę dla tych tokenów)
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
    # --- INICJALIZACJA DDP ---
    # Pobieramy ID procesu (0, 1, 2...) nadane przez torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # Jeśli skrypt uruchomiony w trybie distributed
    if local_rank != -1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Logika dla procesu głównego (Rank 0) - tylko on wypisuje printy
    is_main_process = (local_rank in [-1, 0])

    if is_main_process:
        print(f"👋 Startuje trening. DDP Rank: {local_rank}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- PRZYGOTOWANIE CSV (SYNCHRONIZOWANE) ---
    csv_path = os.path.join(DATA_DIR, "dataset.csv")

    # KROK A: Tylko Szef tworzy CSV (jeśli nie istnieje)
    if is_main_process:
        if not os.path.exists(csv_path):
            print("📄 Generowanie pliku dataset.csv...")
            all_rows = []
            if not os.path.exists(DATA_DIR):
                # Jeśli to wystąpi, upewnij się, że ścieżka DATA_DIR jest OK
                raise FileNotFoundError(f"❌ Brak danych w {DATA_DIR}")

            for medicine in os.listdir(DATA_DIR):
                med_dir = os.path.join(DATA_DIR, medicine)
                if not os.path.isdir(med_dir):
                    continue
                all_rows += rows_from_transcript(os.path.join(med_dir, "transcript.txt"), "word")
                all_rows += rows_from_transcript(os.path.join(med_dir, "transcript_sentences.txt"), "sentence")

            if len(all_rows) == 0:
                raise RuntimeError("❌ Nie znaleziono żadnych plików audio w transcriptach!")

            df_build = pd.DataFrame(all_rows).drop_duplicates(subset=["path"])
            df_build.to_csv(csv_path, index=False)
            print(f"✅ Utworzono CSV z {len(df_build)} wierszami.")
        else:
            print(f"📄 Używam istniejącego pliku CSV: {csv_path}")

    # KROK B: Bariera - Wszyscy czekają na Rank 0
    if local_rank != -1:
        dist.barrier()

    # --- ŁADOWANIE DANYCH ---
    # Teraz bezpiecznie wszyscy czytają ten sam plik
    df = pd.read_csv(csv_path)

    # Split (seed musi być ten sam na każdym GPU!)
    if "dtype" not in df.columns: df["dtype"] = "word"
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["dtype"], random_state=42)

    if is_main_process:
        print(f"📊 Dane: Train={len(train_df)}, Val={len(val_df)}")

    # --- MODEL & PROCESOR ---
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="pl", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    # Konfiguracja generacji
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="pl", task="transcribe")
    model.config.suppress_tokens = []

    # --- DATASET & COLLATOR ---
    train_dataset = MedicinesDataset(train_df, processor, SAMPLE_RATE)
    val_dataset = MedicinesDataset(val_df, processor, SAMPLE_RATE)
    data_collator = SimpleWhisperCollator(processor=processor)

    # --- ARGUMENTY TRENINGOWE ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),

        # Batch size & Memory
        per_device_train_batch_size=BATCH_PER_GPU,
        per_device_eval_batch_size=BATCH_PER_GPU,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        # WAŻNE: Wyłączamy konfliktowe optymalizacje
        gradient_checkpointing=False,

        # WAŻNE: Włączamy FP16 (szybkość + mniejszy RAM)
        fp16=True,

        # Parametry nauki
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=500,
        weight_decay=0.01,

        # Konfiguracja DDP
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,  # 4 CPU workery na każdą kartę GPU

        # Logowanie i Zapis
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    # --- TRAINER ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    # --- TRENING ---
    if is_main_process:
        print("🚀 Rozpoczynam trening (FP16 ON, Checkpointing OFF)...")

    trainer.train()

    # --- ZAPIS ---
    # Synchronizacja: Rank 1 czeka aż Rank 0 zapisze model
    if local_rank != -1:
        dist.barrier()

    if is_main_process:
        print("✅ Trening zakończony. Zapisuję finalny model...")
        trainer.save_model(str(OUTPUT_DIR))
        processor.save_pretrained(str(OUTPUT_DIR))

    # Sprzątanie grupy procesowej
    if local_rank != -1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()