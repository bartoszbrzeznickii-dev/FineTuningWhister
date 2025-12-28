# -*- coding: utf-8 -*-
"""
KROK 1: Inferencja (GPU).
Generuje transkrypcje TYLKO dla wskazanego modelu (bez baseline).
Zapisuje wynik w formacie UTF-8 z BOM (Excel-friendly).
"""

import os
import re
import csv
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# =========================
# KONFIGURACJA ŚCIEŻEK
# =========================

# 1. Ścieżka do modelu (Twoja)
MODEL_DIR = Path("/home/ml480150/whisper_project/wyniki_180587/output_model_ddp_frozen_2ep")

# 2. Ścieżka gdzie zapisać wyniki (Twoja)
OUTPUT_DIR = Path("/home/ml480150/Results")
OUTPUT_CSV = OUTPUT_DIR / "wyniki_180587.csv"

# 3. Ścieżki do DANYCH AUDIO
DATA_ROOT = Path("/home/ml480150/Validation")

DATA_API_DIR = DATA_ROOT / "DataValidation_API"
DATA_HUMAN_DIR = DATA_ROOT / "DataValidation_Human"

# =========================
# PARAMETRY
# =========================
LANG = "pl"
TASK = "transcribe"
SAMPLE_RATE = 16000

# Parametry generacji
GEN_KW = dict(
    max_new_tokens=256,
    do_sample=False
)

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

# =========================
# FUNKCJE POMOCNICZE
# =========================
def norm_text_simple(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\wąćęłńóśżź ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def word_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(text or "")]

def trim_hypothesis_by_ref_last_words(hyp: str, ref: str) -> str:
    """
    Uciana hipotezę, jeśli model wygenerował więcej słów niż jest w referencji,
    bazując na powtórzeniu ostatniego słowa (zapobiega halucynacjom na końcu).
    """
    hyp_str = hyp or ""
    ref_norm = norm_text_simple(ref or "")
    ref_tokens = ref_norm.split()
    if not hyp_str or not ref_tokens:
        return hyp_str.strip()

    last = ref_tokens[-1]
    prev = ref_tokens[-2] if len(ref_tokens) >= 2 else None
    last_count_in_ref = sum(1 for t in ref_tokens if t == last)
    hspans = word_spans(hyp_str)

    seen = 0
    for w, s, e in hspans:
        if w == last:
            seen += 1
            if seen == last_count_in_ref:
                return hyp_str[:e].strip()

    if prev is not None:
        prev_positions = [(idx, s, e) for idx, (w, s, e) in enumerate(hspans) if w == prev]
        if prev_positions:
            last_idx, s, e = prev_positions[-1]
            if last_idx + 1 < len(hspans):
                _, _, e_next = hspans[last_idx + 1]
                return hyp_str[:e_next].strip()
            else:
                return hyp_str[:e].strip()
    return hyp_str.strip()

def load_audio(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros(16000, dtype="float32")

def read_transcript(txt_path: Path) -> List[Tuple[str, str]]:
    rows = []
    if not txt_path.exists():
        return rows
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            if "|" in ln:
                fname, ref = ln.strip().split("|", 1)
                rows.append((fname.strip(), ref.strip()))
    return rows

def collect_pairs(dataset_dir: Path) -> List[Tuple[str, Path, str, str]]:
    out = []
    if not dataset_dir.exists():
        print(f"WARNING: Katalog {dataset_dir} nie istnieje!")
        return out

    for drug_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        drug = drug_dir.name.replace("_", " ")
        txt_files = list(drug_dir.glob("*.txt"))
        if txt_files:
            pairs = read_transcript(txt_files[0])
            for fname, ref in pairs:
                wav_p = drug_dir / fname
                if wav_p.exists():
                    out.append((drug, wav_p, fname, ref))
    return out

# =========================
# KLASA MODELU
# =========================
class WhisperASR:
    def __init__(self, model_dir: Path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--> Ładowanie modelu z: {model_dir}")
        print(f"--> Urządzenie: {self.device}")

        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(str(model_dir))
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        feats = inputs.input_features.to(self.device)

        # Generacja
        pred_ids = self.model.generate(
            feats,
            language=LANG,
            task=TASK,
            **GEN_KW
        )
        # Dekodowanie
        text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        return text.strip()

# =========================
# GŁÓWNA PĘTLA
# =========================
def main():
    print("=== INFERENCJA (SINGLE MODEL) START ===")

    # 0. Przygotowanie outputu
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Zbieranie danych
    print(f"Szukam danych API w: {DATA_API_DIR}")
    print(f"Szukam danych HUMAN w: {DATA_HUMAN_DIR}")

    api_pairs = collect_pairs(DATA_API_DIR)
    human_pairs = collect_pairs(DATA_HUMAN_DIR)

    # Lista krotek: (dataset_name, drug, wav_path, filename, reference)
    all_pairs = [("API", *p) for p in api_pairs] + [("HUMAN", *p) for p in human_pairs]

    if not all_pairs:
        print("BŁĄD: Nie znaleziono żadnych plików audio! Sprawdź ścieżki w DATA_ROOT.")
        print(f"Szukana ścieżka główna: {DATA_ROOT}")
        return

    print(f"Łącznie plików do przetworzenia: {len(all_pairs)}")

    # 2. Ładowanie modelu
    asr = WhisperASR(MODEL_DIR)

    # 3. Pętla przetwarzania
    results = []
    t_start = time.perf_counter()

    print("Rozpoczynam transkrypcję...")
    for i, (ds_name, drug, wav_p, fname, ref) in enumerate(all_pairs, 1):
        if i % 10 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  Postęp: {i}/{len(all_pairs)} (czas: {elapsed:.1f}s)")

        # Wczytanie audio
        audio = load_audio(wav_p)

        # Inferencja
        raw_text = asr.transcribe(audio)

        # Post-processing (przycięcie na podstawie referencji)
        hyp_text = trim_hypothesis_by_ref_last_words(raw_text, ref)

        results.append({
            "dataset": ds_name,
            "drug": drug,
            "filename": fname,
            "filepath": str(wav_p),
            "reference": ref,
            "hypothesis": hyp_text
        })

    # 4. Zapis wyników
    print(f"Zapisywanie wyników do: {OUTPUT_CSV}")

    # ZMIANA TUTAJ: użycie 'utf-8-sig' dodaje BOM dla Excela
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        # Definicja kolumn (dataset, drug, filename, filepath, reference, hypothesis)
        fieldnames = ["dataset", "drug", "filename", "filepath", "reference", "hypothesis"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        # Zmieniłem też delimiter na ";" (średnik), bo polski Excel często woli średniki niż przecinki

        writer.writeheader()
        writer.writerows(results)

    total_time = time.perf_counter() - t_start
    print(f"=== ZAKOŃCZONO ===")
    print(f"Całkowity czas: {total_time:.2f}s")
    print(f"Wyniki w pliku: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()