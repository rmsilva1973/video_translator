import ffmpeg
from fractions import Fraction
from pathlib import Path
import os

class AudioExtractor:
    def __init__(self, video_name):
        self.video_name = video_name
        self.input_path = f'input/{video_name}'
        self.output_path = f'work/audio/{Path(video_name).stem}_16k_mono.wav'
        self.clean_output_path = f'work/audio/{Path(video_name).stem}_clean.wav'

    def process(self):
        # Create output directory
        os.makedirs("work/audio", exist_ok=True)
        
        # 1) ffprobe: vídeo
        vinfo = self.get_video_stream_info(self.input_path)
        print('Video stream:', vinfo)

        # 2) ffprobe: áudio
        ainfo = self.get_audio_stream_info(self.input_path)
        print('Audio stream:', ainfo)

        # 3) ffprobe: duração do contêiner
        finfo = self.get_format_duration(self.input_path)
        print('Format info:', finfo)

        # 4) ffmpeg: processamento de áudio para WAV 16 kHz mono com filtros
        self.process_audio(self.input_path, self.output_path)
        print('Processamento concluído:', self.output_path)

        # 5) ffmpeg: limpeza simples de áudio para WAV 16 kHz mono sem filtros
        self.clean_audio_quick(self.input_path, self.clean_output_path)
        print('Limpeza concluída:', self.clean_output_path)

    def get_video_stream_info(self, input_path):
        probe = ffmpeg.probe(input_path, select_streams='v:0', show_entries='stream=width,height,r_frame_rate')
        streams = probe.get('streams', [])
        if not streams:
            return None
        s = streams[0]
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

    def get_audio_stream_info(self, input_path):
        probe = ffmpeg.probe(input_path, select_streams='a:0', show_entries='stream=sample_rate,channels')
        streams = probe.get('streams', [])
        if not streams:
            return None
        s = streams[0]
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

    def get_format_duration(self, input_path):
        probe = ffmpeg.probe(input_path, show_entries='format=duration')
        fmt = probe.get('format', {})
        duration = None
        if 'duration' in fmt and fmt['duration'] is not None:
            try:
                duration = float(fmt['duration'])
            except Exception:
                duration = None
        return {'duration_seconds': duration}

    def process_audio(self, input_path, output_path):
        (
            ffmpeg
            .input(input_path)
            .audio
            .filter_('loudnorm', I=-23, TP=-2, LRA=11)
            .filter_('highpass', f=80)
            .filter_('lowpass', f=14000)
            .output(
                output_path,
                format='wav',
                ac=1,
                ar=16000
            )
            .overwrite_output()
            .run(quiet=True)
        )

    def clean_audio_quick(self, input_path, output_path):
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
            .run(quiet=True)
        )
