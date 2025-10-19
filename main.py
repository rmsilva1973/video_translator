#!/usr/bin/env python3
import os
import sys
import warnings
from pathlib import Path

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="pyworld")

# Add scripts directory to path
sys.path.append('scripts')

from extract_audio import AudioExtractor
from run_stt import SpeechToText
from run_whisperx_align import WhisperXAlign
from pt_postprocess import PortuguesePostProcessor
from mt_translate import MachineTranslator
from prosody_and_ssml import ProsodySSMLGenerator

def get_video_files(input_dir="input"):
    """Get all video files from input directory"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory '{input_dir}' does not exist")
        return []
    
    video_files = []
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path.name)
    
    return sorted(video_files)

def process_video(video_name):
    """Process a single video through the entire pipeline"""
    print(f"\n{'='*60}")
    print(f"🎬 Processing video: {video_name}")
    print(f"{'='*60}")
    
    try:
        # Step 1: Extract audio
        print("\n🎵 Step 1/6: Extracting and cleaning audio...")
        audio_extractor = AudioExtractor(video_name)
        audio_extractor.process()
        print("✅ Audio extraction completed")
        
        # Step 2: Speech-to-text
        print("\n🗣️  Step 2/6: Running speech-to-text transcription...")
        stt = SpeechToText(video_name)
        stt.process()
        print("✅ Speech-to-text completed")
        
        # Step 3: Word alignment
        print("\n🎯 Step 3/6: Performing word-level alignment...")
        aligner = WhisperXAlign(video_name)
        aligner.process()
        print("✅ Word alignment completed")
        
        # Step 4: Portuguese post-processing
        print("\n📝 Step 4/6: Post-processing Portuguese text...")
        postprocessor = PortuguesePostProcessor(video_name)
        postprocessor.process()
        print("✅ Portuguese post-processing completed")
        
        # Step 5: Machine translation
        print("\n🌐 Step 5/6: Translating to English...")
        translator = MachineTranslator(video_name)
        translator.process()
        print("✅ Machine translation completed")
        
        # Step 6: Prosody and SSML generation
        print("\n🎭 Step 6/6: Generating prosody and SSML...")
        ssml_generator = ProsodySSMLGenerator(video_name)
        ssml_generator.process()
        print("✅ Prosody and SSML generation completed")
        
        print(f"\n🎉 Successfully processed: {video_name}")
        
    except Exception as e:
        print(f"\n❌ Error processing {video_name}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to process all videos in input directory"""
    print("🎬 Video Translation Pipeline")
    print("=" * 40)
    
    # Get all video files
    video_files = get_video_files()
    
    if not video_files:
        print("❌ No video files found in 'input' directory")
        print("📁 Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm")
        return
    
    print(f"📂 Found {len(video_files)} video file(s):")
    for i, video in enumerate(video_files, 1):
        print(f"  {i}. {video}")
    
    # Process each video
    for video_name in video_files:
        process_video(video_name)
    
    print(f"\n🏁 Pipeline completed for all {len(video_files)} video(s)")
    print("🎉 All done!")

if __name__ == "__main__":
    main()
