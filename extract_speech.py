import os
import torch
import torchaudio
import torchaudio.transforms as T 
import argparse
import sys
import types
import math # ‼️ ADDED

# ‼️ WORKAROUND: Monkey patch for DeepFilterNet compatibility with newer torchaudio versions
# Newer torchaudio (>2.1) removed 'backend', but older DeepFilterNet versions import it.
if not hasattr(torchaudio, "backend"):
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    
    # Mock AudioMetaData which is often imported by legacy code
    class AudioMetaData:
        def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    common.AudioMetaData = AudioMetaData
    backend.common = common
    torchaudio.backend = backend
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common

try:
    from moviepy import VideoFileClip
except ImportError:
    import moviepy.editor as mp
    VideoFileClip = mp.VideoFileClip

# ‼️ CHANGED: Only import core DeepFilterNet functions, avoid its IO module if possible
from df.enhance import enhance, init_df
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
        # ‼️ CHANGED: Write to 48kHz (optimal for DeepFilterNet) instead of 16kHz
        # ‼️ CHANGED: Added ffmpeg_params=['-ac', '1'] to force MONO extraction
        video.audio.write_audiofile(
            temp_audio_path, 
            codec='pcm_s16le', 
            fps=48000, 
            ffmpeg_params=["-ac", "1"],
            logger=None
        )
        return temp_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def reduce_background_noise(audio_path, output_path="temp_clean.wav", chunk_sec=30, overlap_sec=2): # ‼️ ADDED chunking params
    """
    Step 2: Apply DeepFilterNet to remove background noise.
    """
    print(f"--> Reducing background noise using DeepFilterNet...")
    print(f"    (Chunk size: {chunk_sec}s, Overlap: {overlap_sec}s for memory efficiency)")

    # ‼️ NEW LOGIC: DeepFilterNet Enhancement
    
    # 1. Initialize DeepFilterNet model
    model, df_state, _ = init_df()
    target_sr = 16000 # Final training SR

    # Get file info without loading
    info = sf.info(audio_path)
    original_sr = info.samplerate
    total_samples = info.frames
    
    # Prepare resamplers if needed
    resampler_in = None
    if original_sr != df_state.sr():
        resampler_in = T.Resample(original_sr, df_state.sr())
        
    resampler_out = None
    if df_state.sr() != target_sr:
        resampler_out = T.Resample(df_state.sr(), target_sr)

    # Calculate chunk sizes in samples
    # We read (chunk + overlap) from file
    # We process it
    # We write (chunk) to output (discarding overlap from the start)
    
    # NOTE: The logic is:
    # Read [current_pos - overlap : current_pos + chunk]
    # Enhance
    # Save [overlap_samples_in_output : ] (Skip the warmed-up part)
    
    chunk_samples = int(chunk_sec * original_sr)
    overlap_samples = int(overlap_sec * original_sr)
    
    # Open input and output files
    with sf.SoundFile(audio_path) as f_in, \
         sf.SoundFile(output_path, 'w', samplerate=target_sr, channels=1, subtype='PCM_16') as f_out:
        
        # Iterate
        current_pos = 0
        total_processed = 0
        
        while current_pos < total_samples:
            # Determine start reading position (include overlap back)
            read_start = max(0, current_pos - overlap_samples)
            read_frames = min(total_samples, current_pos + chunk_samples) - read_start
            
            if read_frames <= 0:
                break
                
            f_in.seek(read_start)
            data_np = f_in.read(frames=read_frames)
            
            # Convert to Tensor (DeepFilterNet expects FloatTensor)
            audio = torch.from_numpy(data_np).float()
            
            # Ensure shape is (1, T)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            elif audio.ndim == 2:
                audio = audio.transpose(0, 1)

            # Resample Input
            if resampler_in:
                audio = resampler_in(audio)

            # Enhance
            # ‼️ NOTE: 'enhance' usually expects the whole file, but with overlap we mitigate state issues
            enhanced_audio = enhance(model, df_state, audio)

            # Resample Output
            if resampler_out:
                enhanced_audio = resampler_out(enhanced_audio)

            # Convert back to Numpy
            enhanced_np = enhanced_audio.detach().cpu().numpy().T
            
            # Determine how much to write
            # We must cut off the overlap from the beginning of the result
            # UNLESS it's the very first chunk (where read_start was 0)
            
            # Calculate output overlap size in target samples
            out_overlap = 0
            if current_pos > 0:
                # How much time was the overlap?
                # It was (current_pos - read_start) samples at original_sr
                actual_overlap_sec = (current_pos - read_start) / original_sr
                out_overlap = int(actual_overlap_sec * target_sr)

            # Slice the valid part
            valid_audio = enhanced_np[out_overlap:]
            
            # Write
            f_out.write(valid_audio)
            
            # Update progress
            current_pos += chunk_samples
            progress = min(100, (current_pos / total_samples) * 100)
            print(f"\r    Progress: {progress:.1f}%", end="")
            
            # Explicit cache clear to be safe with VRAM
            del audio, enhanced_audio, enhanced_np, valid_audio
            torch.cuda.empty_cache()

    print("\n    Denoising complete.")
    return output_path, target_sr

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
    
    # Unpack utils
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

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
    # parser.add_argument("--noise_prop", type=float, default=0.65, help="Noise reduction proportion") ‼️ REMOVED
    parser.add_argument("--vad_threshold", type=float, default=0.35, help="Speech detection sensitivity (0.0-1.0, lower is more sensitive)")
    parser.add_argument("--min_silence", type=int, default=500, help="Minimum silence duration in ms to split segments")
    parser.add_argument("--speech_pad", type=int, default=50, help="Padding in ms added to start/end of speech")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum chunk duration in seconds to keep")
    parser.add_argument("--chunk_size", type=int, default=60, help="Processing chunk size in seconds for memory efficiency") # ‼️ ADDED

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
    # ‼️ Extraction now targets 48kHz for DFN compatibility
    raw_audio = extract_audio_from_video(VIDEO_FILE, TEMP_RAW)
    
    if raw_audio:
        # 2. Denoise
        # ‼️ Calls new DeepFilterNet logic (auto-resamples to 16k at end)
        clean_audio, rate = reduce_background_noise(raw_audio, TEMP_CLEAN, chunk_sec=args.chunk_size)
        
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
