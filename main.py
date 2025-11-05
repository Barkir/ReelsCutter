import os
from video import VideoSel
from pathlib import Path


def main():
    video = VideoSel()
    transcript="".join(open("./transcriptions/Jesse Eisenberg - Mark Zuckerberg (360p)_transcription.txt").readlines())
    # choice = video.select_video()
    # transcript = video.transcribe_video(choice)
    # video.save_transcription(transcript, Path(choice))
    result = video.get_n_moments(transcript, n=5, duration=20)
    print(result)



if __name__ == "__main__":
    main()

