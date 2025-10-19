import ffmpeg
from fractions import Fraction

INPUT = 'input/video.mp4'
OUTPUT = 'work/audio/video_16k_mono.wav'
CLEAN_OUTPUT = 'work/audio/video_clean.wav'

def get_video_stream_info(input_path):
    """
    Equivalente a:
    ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of default=noprint_wrappers=1 input/video.mp4
    """
    probe = ffmpeg.probe(input_path, select_streams='v:0', show_entries='stream=width,height,r_frame_rate')
    streams = probe.get('streams', [])
    if not streams:
        return None
    s = streams[0]
    # Converter r_frame_rate "num/den" para float
    fps = None
    if 'r_frame_rate' in s and s['r_frame_rate'] not in (None, '0/0'):
        try:
            fps = float(Fraction(s['r_frame_rate']))
        except Exception:
            fps = None
    return {
        'width': s.get('width'),
        'height': s.get('height'),
        'r_frame_rate_raw': s.get('r_frame_rate'),
        'fps': fps,
    }

def get_audio_stream_info(input_path):
    """
    Equivalente a:
    ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate,channels -of default=noprint_wrappers=1 input/video.mp4
    """
    probe = ffmpeg.probe(input_path, select_streams='a:0', show_entries='stream=sample_rate,channels')
    streams = probe.get('streams', [])
    if not streams:
        return None
    s = streams[0]
    # sample_rate vem como string; converta se quiser
    sample_rate = None
    if 'sample_rate' in s and s['sample_rate'] is not None:
        try:
            sample_rate = int(s['sample_rate'])
        except Exception:
            sample_rate = None
    return {
        'sample_rate': sample_rate,
        'channels': s.get('channels'),
    }

def get_format_duration(input_path):
    """
    Equivalente a:
    ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 input/video.mp4
    """
    probe = ffmpeg.probe(input_path, show_entries='format=duration')
    fmt = probe.get('format', {})
    duration = None
    if 'duration' in fmt and fmt['duration'] is not None:
        try:
            duration = float(fmt['duration'])
        except Exception:
            duration = None
    return {'duration_seconds': duration}

def process_audio(input_path, output_path):
    """
    Equivalente a:
    ffmpeg -i input/video.mp4 -ac 1 -ar 16000 -vn -filter:a "loudnorm=I=-23:TP=-2:LRA=11,highpass=f=80,lowpass=f=14000" work/video_16k_mono.wav
    """
    (
        ffmpeg
        .input(input_path)
        .audio  # seleciona apenas o áudio (equivalente a -vn)
        # Você pode encadear filtros com .filter_ sucessivos ou usar 'af' na saída.
        # Exemplo com encadeamento explícito:
        .filter_('loudnorm', I=-23, TP=-2, LRA=11)
        .filter_('highpass', f=80)
        .filter_('lowpass', f=14000)
        .output(
            output_path,
            format='wav',  # garante WAV
            ac=1,          # -ac 1
            ar=16000       # -ar 16000
        )
        .overwrite_output()  # sobrescreve se já existir
        .run(quiet=False)    # quiet=True para menos logs
    )

def clean_audio_quick(input_path, output_path):
    """
    Exemplo simples de limpeza de áudio (sem filtros avançados).
    Equivalente a:

    ffmpeg -i work/video_16k_mono.wav -af 
    "afftdn=nf=-25,highpass=f=80,lowpass=f=14000,dynaudnorm=f=200:g=15" work/video_clean.wav

    """
    (
        ffmpeg
        .input(input_path)
        .audio
        .filter_('afftdn', nf=-25)
        .filter_('highpass', f=80)
        .filter_('lowpass', f=14000)
        .filter_('dynaudnorm', f=200, g=15)
        .output(
            output_path,
            format='wav',
            ac=1,
            ar=16000
        )
        .overwrite_output()
        .run(quiet=False)
    )

def clean_audio_rnnnoise(input_path, output_path):
    """
    Exemplo de limpeza de áudio usando o filtro rnnnoiser do FFmpeg.
    Requer que o FFmpeg tenha sido compilado com suporte a rnnnoiser.

    Equivalente a:

    ffmpeg -i work/aula_16k_mono.wav -af "arnndn=m=models/rnnoise_general.model,highpass=f=80,lowpass=f=14000,dynaudnorm=f=200:g=15" work/video_clean.wav

    """
    (
        ffmpeg
        .input(input_path)
        .audio
        .filter_('arnndn', m='models/rnnoise_general.model')
        .filter_('highpass', f=80)
        .filter_('lowpass', f=14000)
        .filter_('dynaudnorm', f=200, g=15)
        .output(
            output_path,
            format='wav',
            ac=1,
            ar=16000
        )
        .overwrite_output()
        .run(quiet=False)
    )

if __name__ == '__main__':
    # 1) ffprobe: vídeo
    vinfo = get_video_stream_info(INPUT)
    print('Video stream:', vinfo)

    # 2) ffprobe: áudio
    ainfo = get_audio_stream_info(INPUT)
    print('Audio stream:', ainfo)

    # 3) ffprobe: duração do contêiner
    finfo = get_format_duration(INPUT)
    print('Format info:', finfo)

    # 4) ffmpeg: processamento de áudio para WAV 16 kHz mono com filtros
    process_audio(INPUT, OUTPUT)
    print('Processamento concluído:', OUTPUT)

    # 5) ffmpeg: limpeza simples de áudio para WAV 16 kHz mono sem filtros
    clean_audio_quick(INPUT, CLEAN_OUTPUT)
    print('Limpeza concluída:', CLEAN_OUTPUT)
