from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.tree import Tree
import questionary
import sys
import os
import whisper
import tempfile
import subprocess
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from constants import *
from openai import OpenAI
from dotenv import load_dotenv

from typing import List

load_dotenv()

API_KEY=os.getenv("OPENAI_API_KEY")

console = Console()

def generate_llm_config(N, T, text):
     return f"[N = {N}, T = {T}]\n{text}"

class VideoSel:
        def __init__(self, video_dir="."):
                self.video_dir = Path(video_dir)
                self.formats = (".mp4", ".avi", ".mkv", ".mov")
                self.video_files = []
                self.tempdir = tempfile.gettempdir()

        def discover_videos(self, directory: str=".") -> List[str]:
            for root, _, files in os.walk(directory):
                 for f in files:
                        if f.lower().endswith(self.formats):
                            self.video_files.append(os.path.join(root, f))

        def select_video(self):
            self.discover_videos()
            if not self.video_files:
                print("No videos found.")
                sys.exit(1)
            print(self.video_files)
            choice = questionary.select(
                "pick a video",
                choices=self.video_files
                  ).ask()
            return choice

        def get_audio(self, choice, audio_format="mp3"):
            audio_filename=f"temp_audio_{os.path.basename(choice).split('.')[0]}.{audio_format}"
            audio_path = os.path.join(self.tempdir, audio_filename)

            ffmpeg_cmd = [
            "ffmpeg", "-i", str(choice),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"
            "-y", str(audio_path)
            ]

            try:
                result = subprocess.run(ffmpeg_cmd)
                if result.returncode != 0:
                     console.print(f"[red]ffmpeg error {result.stderr}[/red]")
                     return None
                console.print(f"[green]audio extracted! ☑️[/green]")
            except Exception as e:
                 console.print(f"[red]error extracting audio: {e}[\red]")
                 return None

        def transcribe_video(self, choice):
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    ) as progress:
                    task1 = progress.add_task("loading model...", total=100)

                    progress.update(task1, advance=30)
                    model = whisper.load_model ("base")
                    progress.update(task1, completed=100)

                    task2 = progress.add_task("transcribing...", total=100)

                    progress.update(task2, advance=20)
                    result=model.transcribe(choice, verbose=False)
                    progress.update(task2, completed=100)

                    return result

            except Exception as e:
                console.print(f"[red]transcription error: {e}[/red]")
                return None

        def format_timestamp(self, seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

        def display_transcription(self, transcription_result, video_name):
            segments = transcription_result.get("segments", [])
            if not segments:
                console.print("[red]No transcription segments were found[/red]")
                return

            table = Table(
                show_header=True,
                header_style="bold_cyan",
                show_lines=True
                )

            table.add_column("Start time", style="yellow", width=12)
            table.add_column("End time", style="yellow", width=12)
            table.add_column("Text", style="white")

            for segment in segments:
                start = self.format_timestamp(segment["start"])
                end = self.format_timestamp(segment["end"])
                text = segment["text"].strip()

                table.add_row(start, end, text)

            total_dur = segments[-1]["end"] if segments else 0

        def save_transcription(self, transcription_result, video_path, output_format="txt"):
            output_dir = Path("transcriptions")
            output_dir.mkdir(exist_ok=True)

            video_stem = video_path.stem
            output_path = output_dir / f"{video_stem}_transcription.{output_format}"

            try:
                if output_format == "txt":
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(f"Transcription for: {video_path.name}\n")
                        f.write("=" * 50 + "\n\n")

                        for seg in transcription_result.get('segments', []):
                            start = self.format_timestamp(seg["start"])
                            end = self.format_timestamp(seg["end"])
                            text = seg["text"].strip()
                            f.write(f"[{start} --> {end}] {text}\n")

                return output_path

            except Exception as e:
                console.print("[red]error writing transcription >_<[/red]")

        def get_n_moments(self, text, n=1, duration=10):
            total_prompt = VIDEO_PROMPT + generate_llm_config(n, duration, text)

            client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY,
            )

            completion = client.chat.completions.create(
            extra_body={},
            model="google/gemini-2.5-flash",
            messages=[
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": f"{total_prompt}"
                  }
                ]
              }
            ]
            )

            return completion.choices[0].message.content





