"""
CLI wrapper for Speech-to-subtitle transcription.

Imports from the existing transcriber.py and scanner.py in this project,
so any improvements to those modules are automatically picked up.

Usage:
    python transcribe_cli.py <input_file> [options]

Options:
    --output, -o      Output SRT path (default: same name as input with .srt)
    --model           Model name or path (default: auto-detect largest available)
    --language        Language code or 'auto' (default: auto)

Exit codes:
    0 = success
    1 = error
"""

import sys
import os
import argparse

# Ensure this script's directory is on sys.path for local imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from transcriber import transcribe, TranscribeOptions, TranscribeProgress
from scanner import scan_whisper_models


def find_best_model(preferred_name=None):
    """Find the best available Whisper model.

    Priority:
    1. If preferred_name matches a discovered model, use it
    2. Otherwise use the largest available model
    3. If no local models found, return the preferred_name as-is
       (faster-whisper will auto-download from HuggingFace)
    """
    # Scan for locally available models
    extra_dirs = [
        os.path.join(
            os.path.dirname(_SCRIPT_DIR),
            'VoiceWriter', 'dist', 'VoiceWriter', 'models', 'whisper'
        ),
    ]
    models = scan_whisper_models(extra_dirs=extra_dirs)

    if not models:
        # No local models found — return name for auto-download
        return preferred_name or 'large-v3'

    # If a preferred name is specified, try to match
    if preferred_name:
        for m in models:
            if preferred_name in m.name.lower() or preferred_name == m.size_label:
                print(f'Using model: {m.display_name()}', file=sys.stderr)
                return m.path

    # Use the largest model available (by file size)
    best = max(models, key=lambda m: m.size_bytes)
    print(f'Auto-selected model: {best.display_name()}', file=sys.stderr)
    return best.path


def progress_callback(progress: TranscribeProgress):
    """Print progress to stderr."""
    print(f'[{progress.percent:5.1f}%] {progress.message}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Speech-to-subtitle CLI transcription'
    )
    parser.add_argument('input_file', help='Audio/video file to transcribe')
    parser.add_argument('--output', '-o',
                        help='Output SRT path (default: <input>.srt)')
    parser.add_argument('--model', default=None,
                        help='Model name/path (default: auto-detect largest)')
    parser.add_argument('--initial-prompt', default=None,
                        help='Domain keywords to guide Whisper recognition')
    parser.add_argument('--language', default=None,
                        help='Language code or auto (default: auto-detect)')

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f'Error: Input file not found: {args.input_file}',
              file=sys.stderr)
        sys.exit(1)

    # Determine output path
    output_path = args.output
    if not output_path:
        base = os.path.splitext(args.input_file)[0]
        output_path = base + '.srt'

    # Find model
    model_path = find_best_model(args.model)

    # Handle language: None means auto-detect in transcriber.py
    language = args.language
    if language and language.lower() == 'auto':
        language = None

    # Run transcription
    options = TranscribeOptions(
        model_path=model_path,
        input_file=args.input_file,
        output_srt=output_path,
        language=language,
        device='cuda',
        compute_type='float16',
        initial_prompt=args.initial_prompt,
    )

    print(f'Input:  {args.input_file}', file=sys.stderr)
    print(f'Output: {output_path}', file=sys.stderr)
    print(f'Model:  {model_path}', file=sys.stderr)
    print(f'Language: {language or "auto-detect"}', file=sys.stderr)
    print('', file=sys.stderr)

    try:
        result_path = transcribe(options, on_progress=progress_callback)
        print(f'\nSuccess: {result_path}', file=sys.stderr)
    except Exception as e:
        print(f'\nError: {e}', file=sys.stderr)
        sys.exit(1)

    # Avoid CUDA destructor crashes on Windows
    os._exit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
