#!/usr/bin/env python3
"""
Test script for InspireMusic music generation.

This script demonstrates how to generate music using the InspireMusic repository.
It includes examples for:
- Text-to-music generation
- Music continuation
- Different chorus modes (intro, verse, chorus, outro)
- Various configuration options
"""

import os
import sys
import torch
from inspiremusic.cli.inference import InspireMusicModel, env_variables


def test_text_to_music(model, output_dir="test_outputs"):
    """
    Test text-to-music generation with different prompts and configurations.
    """
    print("\n" + "="*80)
    print("Testing Text-to-Music Generation")
    print("="*80)

    # Test prompts for different music styles
    test_prompts = [
        {
            "text": "Experience soothing and sensual instrumental jazz with a touch of Bossa Nova, perfect for a relaxing restaurant or spa ambiance.",
            "chorus": "intro",
            "time_end": 30.0,
            "output_fn": "jazz_bossa_nova"
        },
        {
            "text": "Energetic electronic dance music with heavy bass drops and uplifting synth melodies.",
            "chorus": "verse",
            "time_end": 20.0,
            "output_fn": "edm_dance"
        },
        {
            "text": "Calm piano melody with soft strings, perfect for meditation and relaxation.",
            "chorus": "outro",
            "time_end": 25.0,
            "output_fn": "calm_piano"
        }
    ]

    results = []
    for i, prompt_config in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"Prompt: {prompt_config['text'][:60]}...")
        print(f"Chorus mode: {prompt_config['chorus']}")
        print(f"Duration: {prompt_config['time_end']}s")

        try:
            result = model.inference(
                task='text-to-music',
                text=prompt_config['text'],
                chorus=prompt_config['chorus'],
                time_start=0.0,
                time_end=prompt_config['time_end'],
                output_fn=prompt_config['output_fn'],
                output_format='wav',
                fade_out_mode=True,
                fade_out_duration=1.0
            )
            results.append({
                "prompt": prompt_config['text'],
                "output": result,
                "status": "success"
            })
            print(f"✓ Successfully generated: {result}")
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            results.append({
                "prompt": prompt_config['text'],
                "output": None,
                "status": "failed",
                "error": str(e)
            })

    return results


def test_music_continuation(model, audio_prompt_path=None):
    """
    Test music continuation with an audio prompt.
    Note: Requires an audio prompt file. If not provided, this test will be skipped.
    """
    print("\n" + "="*80)
    print("Testing Music Continuation")
    print("="*80)

    if audio_prompt_path is None or not os.path.exists(audio_prompt_path):
        print("⚠ Skipping continuation test - no audio prompt provided")
        print("  To test continuation, provide a path to an audio file (WAV format recommended)")
        return None

    print(f"Using audio prompt: {audio_prompt_path}")

    try:
        # Test with audio prompt only
        print("\n--- Test 1: Audio prompt only ---")
        result1 = model.inference(
            task='continuation',
            text=None,
            audio_prompt=audio_prompt_path,
            chorus='verse',
            time_start=0.0,
            time_end=30.0,
            output_fn='continuation_audio_only',
            max_audio_prompt_length=5.0,
            output_format='wav'
        )
        print(f"✓ Generated: {result1}")

        # Test with both text and audio prompt
        print("\n--- Test 2: Text + Audio prompt ---")
        result2 = model.inference(
            task='continuation',
            text="Continue the melody with more upbeat energy and add drums.",
            audio_prompt=audio_prompt_path,
            chorus='chorus',
            time_start=0.0,
            time_end=30.0,
            output_fn='continuation_text_audio',
            max_audio_prompt_length=5.0,
            output_format='wav'
        )
        print(f"✓ Generated: {result2}")

        return [result1, result2]

    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return None


def test_different_chorus_modes(model):
    """
    Test all available chorus modes with the same text prompt.
    """
    print("\n" + "="*80)
    print("Testing Different Chorus Modes")
    print("="*80)

    prompt_text = "Uplifting orchestral music with strings and brass instruments."
    chorus_modes = ["intro", "verse", "chorus", "outro", "random"]

    results = []
    for chorus in chorus_modes:
        print(f"\n--- Testing chorus mode: {chorus} ---")
        try:
            result = model.inference(
                task='text-to-music',
                text=prompt_text,
                chorus=chorus,
                time_start=0.0,
                time_end=15.0,
                output_fn=f"chorus_test_{chorus}",
                output_format='wav'
            )
            results.append({
                "chorus": chorus,
                "output": result,
                "status": "success"
            })
            print(f"✓ Generated: {result}")
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            results.append({
                "chorus": chorus,
                "output": None,
                "status": "failed",
                "error": str(e)
            })

    return results


def main():
    """
    Main test function.
    """
    print("="*80)
    print("InspireMusic Test Suite")
    print("="*80)

    # Set up environment variables
    print("\n[1/4] Setting up environment...")
    env_variables()

    # Configuration
    MODEL_NAME = "InspireMusic-Base"  # Options: InspireMusic-Base, InspireMusic-1.5B, InspireMusic-1.5B-Long
    GPU_ID = 0  # Use 0 for first GPU, -1 for CPU
    FAST_MODE = True  # Set to True for faster inference (without flow matching)

    # Use paths within the repository
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", MODEL_NAME)
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "exp", "test_outputs")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if CUDA is available
    if GPU_ID >= 0:
        if torch.cuda.is_available():
            print(f"✓ CUDA available - using GPU {GPU_ID}")
        else:
            print("⚠ CUDA not available - falling back to CPU")
            GPU_ID = -1
    else:
        print("Running on CPU mode")

    # Initialize model
    print(f"\n[2/4] Initializing InspireMusic model: {MODEL_NAME}")
    print(f"  Model directory: {MODEL_DIR}")
    print(f"  Fast mode: {FAST_MODE}")
    print(f"  Output directory: {OUTPUT_DIR}")

    try:
        model = InspireMusicModel(
            model_name=MODEL_NAME,
            model_dir=MODEL_DIR,  # Use explicit path within repository
            min_generate_audio_seconds=10.0,
            max_generate_audio_seconds=30.0,
            sample_rate=24000,
            output_sample_rate=24000 if FAST_MODE else 48000,
            load_jit=True,
            load_onnx=False,
            dtype="fp16",
            fast=FAST_MODE,
            fp16=True,
            gpu=GPU_ID,
            result_dir=OUTPUT_DIR
        )
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize model: {str(e)}")
        print("\nMake sure you have:")
        print("  1. Installed all dependencies (see requirements.txt)")
        print("  2. Downloaded the model or have internet access to auto-download")
        print("  3. Sufficient GPU memory (or use CPU mode with gpu=-1)")
        return

    # Run tests
    print("\n[3/4] Running tests...")

    # Test 1: Text-to-music generation
    text_to_music_results = test_text_to_music(model, output_dir=OUTPUT_DIR)

    # Test 2: Different chorus modes
    chorus_results = test_different_chorus_modes(model)

    # Test 3: Music continuation (optional - requires audio prompt)
    # Uncomment and provide path to test:
    # continuation_results = test_music_continuation(model, audio_prompt_path="path/to/your/audio.wav")

    # Print summary
    print("\n" + "="*80)
    print("[4/4] Test Summary")
    print("="*80)

    print("\nText-to-Music Results:")
    success_count = sum(1 for r in text_to_music_results if r['status'] == 'success')
    print(f"  ✓ Success: {success_count}/{len(text_to_music_results)}")
    for r in text_to_music_results:
        if r['status'] == 'success':
            print(f"    - {r['output']}")

    print("\nChorus Mode Results:")
    success_count = sum(1 for r in chorus_results if r['status'] == 'success')
    print(f"  ✓ Success: {success_count}/{len(chorus_results)}")

    print("\n" + "="*80)
    print(f"All output files saved to: {OUTPUT_DIR}")
    print("="*80)

    print("\nNote: To test music continuation, uncomment the continuation test")
    print("      in main() and provide a path to an audio file (WAV format recommended)")


if __name__ == "__main__":
    main()
