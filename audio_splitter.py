#!/usr/bin/env python3
import os
import argparse
import subprocess
import shutil
from pydub import AudioSegment
from pydub.silence import split_on_silence

def check_ffmpeg():
    """Checks if ffmpeg is installed and accessible."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg not found! Please install it (Arch/CachyOS: sudo pacman -S ffmpeg).")

def extract_audio_from_video(video_path, temp_audio_path):
    """
    Extracts audio from video using ffmpeg subprocess for speed.
    """
    print(f"--> Extracting audio from {video_path}...")
    
    # -y overwrites output, -vn disables video, -ac 1 converts to mono (better for speech processing)
    command = [
        "ffmpeg", "-y", 
        "-i", video_path, 
        "-vn", 
        "-acodec", "pcm_s16le", 
        "-ar", "44100", 
        "-ac", "1", 
        temp_audio_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("--> Audio extraction complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        exit(1)

def split_audio_by_silence(audio_path, output_folder, min_silence_len=700, silence_thresh_offset=-16, keep_silence=400):
    """
    Splits audio file into chunks based on silence.
    
    Args:
        min_silence_len: Minimum length of silence (ms) to be considered a split.
        silence_thresh_offset: dB relative to average loudness to consider silence.
        keep_silence: Amount of silence (ms) to leave at the start/end of chunk.
    """
    print("--> Loading audio into memory (this may take a moment for large files)...")
    sound = AudioSegment.from_wav(audio_path)
    
    # Dynamic threshold: avg dBFS - offset (e.g., -20dB - 16dB = -36dB threshold)
    avg_db = sound.dBFS
    thresh = avg_db + silence_thresh_offset
    
    print(f"--> Audio Average dB: {avg_db:.2f}")
    print(f"--> Silence Threshold: {thresh:.2f} dB")
    print("--> Splitting processing started...")

    chunks = split_on_silence(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=thresh,
        keep_silence=keep_silence
    )

    print(f"--> Found {len(chunks)} sentences/chunks. Exporting...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, chunk in enumerate(chunks):
        out_file = os.path.join(output_folder, f"sentence_{i+1:04d}.wav")
        chunk.export(out_file, format="wav")
        # Optional: Print progress every 10 chunks
        if (i+1) % 10 == 0:
            print(f"    Exported {i+1}/{len(chunks)} files", end='\r')
    
    print(f"\n--> Done! Files saved in '{output_folder}'")

def main():
    parser = argparse.ArgumentParser(description="Extract audio from video and split by silence.")
    parser.add_argument("video_file", help="Path to the input video file")
    parser.add_argument("--output", "-o", default="output_sentences", help="Output directory for wav files")
    parser.add_argument("--min_silence", type=int, default=700, help="Min silence length in ms (default: 700)")
    parser.add_argument("--thresh", type=int, default=-16, help="Silence threshold relative to avg dB (default: -16)")
    
    args = parser.parse_args()

    # 1. Check Dependencies
    try:
        check_ffmpeg()
    except EnvironmentError as e:
        print(e)
        return

    if not os.path.isfile(args.video_file):
        print(f"Error: File '{args.video_file}' not found.")
        return

    # 2. Extract Audio
    temp_wav = "temp_extracted_audio.wav"
    try:
        extract_audio_from_video(args.video_file, temp_wav)
        
        # 3. Split Audio
        split_audio_by_silence(
            temp_wav, 
            args.output, 
            min_silence_len=args.min_silence, 
            silence_thresh_offset=args.thresh
        )
    finally:
        # Cleanup temp file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

if __name__ == "__main__":
    main()
