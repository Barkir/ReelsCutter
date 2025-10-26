from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.tree import Tree
import questionary
import sys
import os

from typing import List

console = Console()

class VideoSel:
        def __init__(self, video_dir="."):
                self.video_dir = Path(video_dir)
                self.formats = (".mp4", ".avi", ".mkv", ".mov")
                self.video_files = []

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


