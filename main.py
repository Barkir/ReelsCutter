import os
from select_video import VideoSel


def main():
    video = VideoSel()
    video.discover_videos()
    video.display_files()


if __name__ == "__main__":
    main()

