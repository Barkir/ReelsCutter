#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reels Cutter — CLI-пайплайн:
1) выбор видео (TUI),
2) транскрибация с таймкодами (faster-whisper),
3) LLM-аналитика «интересных моментов» и CSV,
4) нарезка клипов,
5) добавление (burn-in) субтитров.

Автор: ты + ИИ :)
"""

import os
import sys
import csv
import json
import math
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import questionary
from tqdm import tqdm


# Транскрибирование
from faster_whisper import WhisperModel

# Опционально: LLM
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None

try:
    # Новая openai >=1.0 (Responses/Chat Completions поддерживаются клиентом)
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


# --------- Утилиты времени ---------
def sec_to_timestamp(sec: float) -> str:
    """ 12.345 -> '00:00:12,345' """
    sec = max(0.0, float(sec))
    ms = int(round((sec - int(sec)) * 1000))
    s = int(sec) % 60
    m = (int(sec) // 60) % 60
    h = int(sec) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def timestamp_to_sec(ts: str) -> float:
    """ 'HH:MM:SS,mmm' -> seconds float """
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(",")
    return int(hh)*3600 + int(mm)*60 + int(ss) + int(ms)/1000.0


# --------- Датаклассы ---------
@dataclass
class Segment:
    start: float
    end: float
    text: str

@dataclass
class Highlight:
    idx: int
    start: float
    end: float
    caption: str  # субтитры/комментарий к моменту


# --------- Шаг 1: выбор файла ---------
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm")

def discover_videos(directory: str = ".") -> List[str]:
    out = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)

def pick_video() -> str:
    videos = discover_videos(".")
    if not videos:
        print("В папке нет видео. Положи файл .mp4/.mov/.mkv и снова запусти.", file=sys.stderr)
        sys.exit(1)
    choice = questionary.select(
        "Выбери видео:",
        choices=videos
    ).ask()
    if not choice:
        sys.exit(0)
    return choice


# --------- Шаг 2: транскрибация ---------
def transcribe_video(
    video_path: str,
    model_name: str = "medium",   # можно "small"/"medium"/"large-v3" (если есть VRAM/CPU)
    device: str = "auto",
    language: Optional[str] = None,   # Пример: "ru", "en"
) -> List[Segment]:
    """
    Возвращает список сегментов с таймкодами и текстом.
    faster-whisper даёт уже сегменты по 1-20 сек обычно.
    """
    model = WhisperModel(model_name, device=device, compute_type="auto")
    segments, info = model.transcribe(video_path, language=language, vad_filter=True)
    result = []
    for seg in segments:
        result.append(Segment(start=float(seg.start), end=float(seg.end), text=seg.text.strip()))
    return result

def write_full_srt(segments: List[Segment], srt_path: str):
    os.makedirs(os.path.dirname(srt_path), exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, start=1):
            f.write(f"{i}\n{sec_to_timestamp(s.start)} --> {sec_to_timestamp(s.end)}\n{s.text}\n\n")


# --------- Шаг 3: анализ LLM (или fallback) ---------
LLM_PROMPT = """Ты — редактор коротких нарезок (reels/shorts).
По транскрипции выбери самые “сильные” моменты, которые хочется смотреть до конца.
Выводи JSON со списком объектов:
[{{
  "start": 11.0,          // секунда начала момента
  "end": 22.0,            // секунда конца
  "caption": "краткая подпись/суть этого фрагмента"
}}]

Правила:
- длительность каждого момента: 8–30 секунд,
- не пересекай моменты между собой,
- 5–10 моментов на видео,
- caption на языке оригинала транскрибации, без кавычек и эмодзи,
- используй тайминги исходя из сегментов (можно объединять соседние сегменты).

Ниже JSON со списком сегментов (start/end в секундах):

"""

def call_llm_for_highlights(segments: List[Segment], max_moments: int = 8) -> List[Highlight]:
    if not (_HAS_OPENAI and OPENAI_API_KEY):
        return heuristic_highlights(segments, max_moments=max_moments)

    # Готовим компактное тело сегментов, чтобы не переборщить с токенами
    segs_for_llm = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
        for s in segments
    ]
    input_json = json.dumps(segs_for_llm, ensure_ascii=False)

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)

    # Chat Completions (совместимо с большинством OpenAI-совместимых серверов)
    msg = [
        {"role": "system", "content": "Ты помощник-редактор видео."},
        {"role": "user", "content": LLM_PROMPT + input_json}
    ]
    try:
        resp = client.chat.completions.create(
            model=os.getenv("REELS_LLM_MODEL", "gpt-4o-mini"),
            messages=msg,
            temperature=0.4,
            max_tokens=1200,
        )
        text = resp.choices[0].message.content.strip()
        # Найдём JSON в ответе (на случай вступления)
        json_str = text
        # Пытаемся распарсить напрямую
        data = json.loads(json_str)
        highlights: List[Highlight] = []
        for i, obj in enumerate(data, start=1):
            s = float(obj["start"]); e = float(obj["end"]); cap = str(obj.get("caption","")).strip()
            if e - s >= 6.0:  # фильтр слишком коротких
                highlights.append(Highlight(idx=i, start=s, end=e, caption=cap))
        return trim_and_nonoverlap(highlights, max_count=max_moments)
    except Exception as e:
        print(f"[LLM] Ошибка LLM, fallback эвристика: {e}", file=sys.stderr)
        return heuristic_highlights(segments, max_moments=max_moments)


def heuristic_highlights(segments: List[Segment], window_sec: int = 15, max_moments: int = 8) -> List[Highlight]:
    """
    Простейший fallback без LLM:
    - Склеиваем соседние сегменты в окна ~15с,
    - ранжируем по «плотности текста» (символы/сек),
    - берём топ-N непересекающихся.
    """
    # Сбор окон
    windows: List[Tuple[float, float, str]] = []
    cur_text = []
    cur_start = None
    cur_end = None

    for seg in segments:
        if cur_start is None:
            cur_start = seg.start
            cur_end = seg.end
            cur_text = [seg.text]
            continue
        # Если добавление укладывается в окно
        if (seg.end - cur_start) <= (window_sec + 3):
            cur_end = seg.end
            cur_text.append(seg.text)
        else:
            # закрываем окно
            windows.append((cur_start, cur_end, " ".join(cur_text)))
            # начинаем новое
            cur_start = seg.start
            cur_end = seg.end
            cur_text = [seg.text]
    if cur_start is not None:
        windows.append((cur_start, cur_end, " ".join(cur_text)))

    scored = []
    for (s, e, t) in windows:
        dur = max(1e-3, e - s)
        density = len(t.strip()) / dur
        # лёгкий бонус, если встречаются "итог/важно/поэтому"
        bonus = 1.15 if any(k in t.lower() for k in ["итог", "вывод", "важ", "поэтому", "смысл", "главное"]) else 1.0
        score = density * bonus
        scored.append((score, s, e, t))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Берём непересекающиеся
    picked: List[Highlight] = []
    for _, s, e, t in scored:
        if len(picked) >= max_moments:
            break
        if all(e <= h.start or s >= h.end for h in picked):  # no overlap
            if (e - s) >= 6.0 and (e - s) <= 35.0:
                picked.append(Highlight(idx=len(picked)+1, start=s, end=e, caption=t.strip()))
    return picked

def trim_and_nonoverlap(items: List[Highlight], max_count: int = 8) -> List[Highlight]:
    items = sorted(items, key=lambda x: (x.start, x.end))
    picked: List[Highlight] = []
    for it in items:
        if len(picked) >= max_count:
            break
        if all(it.end <= h.start or it.start >= h.end for h in picked):
            # ограничим разумной длительностью
            dur = it.end - it.start
            if dur < 6.0:
                it.end = it.start + 6.0
            if dur > 35.0:
                it.end = it.start + 35.0
            picked.append(it)
    # перенумерация
    for i, h in enumerate(picked, start=1):
        h.idx = i
    return picked


# --------- Шаг 4–5: резка и субтитры ---------
def ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False

def cut_clip(src: str, dst: str, start: float, end: float):
    """
    Нарезка без перекодирования (fast): -ss/-to + -c copy.
    Для некоторых контейнеров ключевой кадр может смещать пару кадров.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", src,
        "-c", "copy",
        dst
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def burn_subtitles_to_clip(clip_path: str, srt_path: str, out_path: str):
    """
    Жёстко впаиваем субтитры из .srt в ролик.
    (требуется libass; на Windows может понадобиться сборка ffmpeg with libass)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Пример простой стилизации через force_style
    vf = f"subtitles='{srt_path.replace("'\",\"\\'")}:force_style=Fontsize=22,PrimaryColour=&H00FFFFFF,Outline=1,Shadow=0,MarginV=40'"
    cmd = ["ffmpeg", "-y", "-i", clip_path, "-vf", vf, "-c:a", "copy", out_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def write_srt_window(segments: List[Segment], global_start: float, global_end: float, srt_path: str):
    """
    Пишем .srt для окна [global_start, global_end], с локальными таймкодами от 0.
    """
    local = []
    for s in segments:
        if s.end <= global_start or s.start >= global_end:
            continue
        # обрезаем на границах
        s_start = max(s.start, global_start)
        s_end = min(s.end, global_end)
        # сдвиг к нулю
        ls = s_start - global_start
        le = s_end - global_start
        local.append(Segment(start=ls, end=le, text=s.text))

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(local, start=1):
            f.write(f"{i}\n{sec_to_timestamp(seg.start)} --> {sec_to_timestamp(seg.end)}\n{seg.text}\n\n")


# --------- Шаг 3.5: CSV экспорт ---------
def export_highlights_csv(items: List[Highlight], csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "start_sec", "end_sec", "start_ts", "end_ts", "caption"])
        for h in items:
            writer.writerow([h.idx, f"{h.start:.3f}", f"{h.end:.3f}", sec_to_timestamp(h.start), sec_to_timestamp(h.end), h.caption])


# --------- Главный сценарий ---------
def main():
    if not ffmpeg_exists():
        print("Не найден ffmpeg в PATH. Установи ffmpeg и перезапусти.", file=sys.stderr)
        sys.exit(1)

    video_path = pick_video()
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join("outputs", base)
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(out_dir, exist_ok=True)

    # Параметры
    language = questionary.select(
        "Язык речи в видео (можно Auto):",
        choices=["Auto", "ru", "en", "de", "fr", "es"]
    ).ask()
    if language == "Auto":
        language = None

    model_name = questionary.select(
        "Whisper-модель:",
        choices=["small", "medium", "large-v3"]
    ).ask()

    print("\n[1/5] Транскрибация…")
    segments = transcribe_video(video_path, model_name=model_name, language=language)

    full_srt = os.path.join(out_dir, "full.srt")
    write_full_srt(segments, full_srt)
    print(f"  Готово: {full_srt} (все субтитры)")

    print("\n[2/5] Аналитика моментов…")
    max_moments = questionary.select(
        "Сколько моментов выделить?",
        choices=["5", "6", "7", "8", "9", "10"],
        default="8"
    ).ask()
    highlights = call_llm_for_highlights(segments, max_moments=int(max_moments))

    csv_path = os.path.join(out_dir, "highlights.csv")
    export_highlights_csv(highlights, csv_path)
    print(f"  CSV с моментами: {csv_path}")

    # Подтверждение перед резкой
    print("\nПредлагаемые моменты:")
    for h in highlights:
        print(f"  #{h.idx}: {sec_to_timestamp(h.start)} — {sec_to_timestamp(h.end)}  |  {h.caption[:80]}")

    ok = questionary.confirm("Резать клипы и впаивать субтитры?", default=True).ask()
    if not ok:
        print("Ок, остановка после CSV.")
        return

    print("\n[3/5] Нарезка клипов…")
    os.makedirs(clips_dir, exist_ok=True)

    for h in tqdm(highlights, desc="Клипы"):
        raw_clip = os.path.join(clips_dir, f"clip_{h.idx:03d}_raw.mp4")
        final_clip = os.path.join(clips_dir, f"clip_{h.idx:03d}.mp4")
        try:
            cut_clip(video_path, raw_clip, h.start, h.end)
        except subprocess.CalledProcessError:
            # fallback: перекодируем, когда 'copy' не получается
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{h.start:.3f}",
                "-to", f"{h.end:.3f}",
                "-i", video_path,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                raw_clip
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Локальный srt для окна
        with tempfile.TemporaryDirectory() as tmpdir:
            local_srt = os.path.join(tmpdir, "local.srt")
            write_srt_window(segments, h.start, h.end, local_srt)
            # burn-in
            burn_subtitles_to_clip(raw_clip, local_srt, final_clip)

        # чистим raw
        try:
            os.remove(raw_clip)
        except Exception:
            pass

    print("\n[4/5] Готово! Клипы с субтитрами лежат тут:")
    print(f"  {clips_dir}")

    print("\n[5/5] Итого:")
    print(f"- Полные субтитры: {full_srt}")
    print(f"- CSV моментов:    {csv_path}")
    print(f"- Клипы:           {clips_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
