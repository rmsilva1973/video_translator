import ffmpeg

video_file = 'videos/MENTORIABULLYING1-1.mp4'

(
    ffmpeg
    .input(video_file)
    .output('audios/MENTORIABULLYING1-1.mp3')
    .run()
)