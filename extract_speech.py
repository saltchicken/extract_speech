import os
import torch
import torchaudio
import argparse

try:
    from moviepy import VideoFileClip
except ImportError:
    import moviepy.editor as mp
    VideoFileClip = mp.VideoFileClip

import noisereduce as nr
import soundfile as sf
import numpy as np
from pathlib import Path

def extract_audio_from_video(video_path, temp_audio_path="temp_raw.wav"):
    """
    Step 1: Extract audio from video file using MoviePy.
    """
    print(f"--> Extracting audio from {video_path}...")
    try:

        video = VideoFileClip(video_path)
        # Write to wav, 16-bit PCM, 16kHz (standard for speech training)
        video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', fps=16000, logger=None)
        return temp_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def reduce_background_noise(audio_path, output_path="temp_clean.wav", prop_decrease=0.65):
    """
    Step 2: Apply spectral gating to remove stationary background noise.
    """
    print(f"--> Reducing background noise (prop_decrease={prop_decrease})...")
    
    # Load audio
    data, rate = sf.read(audio_path)
    
    # If stereo, convert to mono for training compatibility
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Perform noise reduction

    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=prop_decrease, stationary=True)
    
    sf.write(output_path, reduced_noise, rate)
    return output_path, rate

def split_audio_vad(audio_path, output_dir, sampling_rate=16000, threshold=0.35, min_silence_ms=500, speech_pad_ms=50, min_duration_sec=0.5):
    """
    Step 3: Use Silero VAD (Torch) to detect speech timestamps and split.
    """
    print(f"--> Detecting speech segments (threshold={threshold}, min_silence={min_silence_ms}ms)...")
    
    # Load Silero VAD from Torch Hub (highly accurate, fast)
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  trust_repo=True)
    
    # Unpack utils, but we won't use read_audio to avoid torchcodec errors
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


    # wav = read_audio(audio_path, sampling_rate=sampling_rate) (OLD)
    
    data, sr = sf.read(audio_path)
    wav = torch.from_numpy(data).float()
    
    # Silero expects shape (1, T)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    
    # Get speech timestamps

    speech_timestamps = get_speech_timestamps(
        wav, 
        model, 
        sampling_rate=sampling_rate, 
        threshold=threshold, 
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms
    )
    
    if not speech_timestamps:
        print("No speech detected.")
        return

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"--> Exporting {len(speech_timestamps)} segments to {output_dir}...")
    
    # Read original clean audio to slice
    data, sr = sf.read(audio_path)
    
    count = 0
    
    for segment in speech_timestamps:
        start_sample = segment['start']
        end_sample = segment['end']
        
        # Calculate duration
        duration = (end_sample - start_sample) / sr
        

        if duration < min_duration_sec:
            continue
            
        chunk = data[start_sample:end_sample]
        
        # Output filename
        chunk_name = f"chunk_{count:04d}_{duration:.2f}s.wav"
        out_file = os.path.join(output_dir, chunk_name)
        
        sf.write(out_file, chunk, sr)
        count += 1
        
    print(f"Done! Saved {count} training-ready chunks.")

def main():

    parser = argparse.ArgumentParser(description="Extract speech chunks from video for training.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output_dir", default="training_dataset", help="Directory to save audio chunks")
    parser.add_argument("--noise_prop", type=float, default=0.65, help="Noise reduction proportion (0.0-1.0)")
    parser.add_argument("--vad_threshold", type=float, default=0.35, help="Speech detection sensitivity (0.0-1.0, lower is more sensitive)")
    parser.add_argument("--min_silence", type=int, default=500, help="Minimum silence duration in ms to split segments")
    parser.add_argument("--speech_pad", type=int, default=50, help="Padding in ms added to start/end of speech")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum chunk duration in seconds to keep")

    args = parser.parse_args()

    # --- CONFIGURATION ---
    VIDEO_FILE = args.video_path
    OUTPUT_FOLDER = args.output_dir
    TEMP_RAW = "temp_raw_audio.wav"
    TEMP_CLEAN = "temp_clean_audio.wav"
    
    if not os.path.exists(VIDEO_FILE):
        print(f"Video file not found: {VIDEO_FILE}")
        return

    # 1. Extract
    raw_audio = extract_audio_from_video(VIDEO_FILE, TEMP_RAW)
    
    if raw_audio:
        # 2. Denoise

        clean_audio, rate = reduce_background_noise(raw_audio, TEMP_CLEAN, prop_decrease=args.noise_prop)
        
        # 3. Split by Speech

        split_audio_vad(
            clean_audio, 
            OUTPUT_FOLDER, 
            sampling_rate=rate,
            threshold=args.vad_threshold,
            min_silence_ms=args.min_silence,
            speech_pad_ms=args.speech_pad,
            min_duration_sec=args.min_duration
        )
        
        # Cleanup temp files
        if os.path.exists(TEMP_RAW): os.remove(TEMP_RAW)
        if os.path.exists(TEMP_CLEAN): os.remove(TEMP_CLEAN)

if __name__ == "__main__":
    main()
