from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.tree import Tree

console = Console()

class VideoSel:
        def __init__(self, video_dir="."):
                self.video_dir = Path(video_dir)
                self.formats = {".mp4", ".avi", ".mkv", ".mov"}
                self.video_files = []

        def discover_videos(self):
            for fmt in self.formats:
                    self.video_files.extend(self.video_dir.rglob(f"*{fmt}"))

            return self.video_files

        def display_files(self):
            tree = Tree("[bold magenta]Videos[/bold magenta]")
            folders = {}
            for video in self.video_files:
                folder = str(video.parent.relative_to(self.video_dir))
                if folder not in folders:
                      folders[folder] = [video]
                else:
                    folders[folder].append(video)

            for folder, videos in sorted(folders.items()):
                  branch = tree.add(f"[yellow]{folder}[/yellow]")
                  for video in videos:
                        branch.add(f"[green]{video.name}[/green]")

            console.print(tree)

