#!/usr/bin/env python3
"""
Voice Profile Manager
=====================
CLI tool for managing cloned voice profiles.

Usage:
    python voice_manager.py create <name> <reference_audio> [--language en]
    python voice_manager.py list
    python voice_manager.py delete <name>
    python voice_manager.py test <name> "Text to speak"
    python voice_manager.py set-default <name>
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from voice_cloning import VoiceCloningTTS, VoiceCloningConfig


async def create_profile(args):
    """Create a new voice profile."""
    config = Config()
    
    vc_config = VoiceCloningConfig(
        enabled=True,
        backend=args.backend,
        device="cuda" if args.gpu else "cpu",
        voices_dir=Path(args.voices_dir)
    )
    
    tts = VoiceCloningTTS(vc_config)
    
    print(f"Initializing {args.backend} backend...")
    if not await tts.initialize():
        print("ERROR: Failed to initialize voice cloning backend")
        print("Make sure the required library is installed:")
        print("  - xtts: pip install TTS")
        print("  - chatterbox: pip install chatterbox-tts")
        return 1
        
    # Check reference audio exists
    reference_path = Path(args.reference_audio)
    if not reference_path.exists():
        print(f"ERROR: Reference audio not found: {reference_path}")
        return 1
        
    # Get audio duration
    try:
        import soundfile as sf
        info = sf.info(str(reference_path))
        duration = info.duration
        print(f"Reference audio: {reference_path.name} ({duration:.1f}s)")
        
        if duration < 3:
            print("WARNING: Reference audio is very short. Recommend 5-30 seconds.")
        elif duration > 60:
            print("WARNING: Reference audio is long. First 30-60 seconds will be used.")
    except:
        print(f"Reference audio: {reference_path.name}")
        
    print(f"\nCreating voice profile '{args.name}'...")
    
    try:
        profile = await tts.create_profile(
            name=args.name,
            reference_audio=str(reference_path),
            description=args.description,
            language=args.language,
            set_as_default=args.default
        )
        
        print(f"\n✓ Voice profile '{args.name}' created successfully!")
        print(f"  Location: {vc_config.voices_dir / args.name}")
        
        if args.default:
            print(f"  Set as default voice")
            
        # Test synthesis
        if args.test:
            print(f"\nTesting synthesis...")
            audio = await tts.synthesize("Hello! This is a test of the cloned voice.", args.name)
            
            test_file = vc_config.voices_dir / args.name / "test_output.wav"
            
            import wave
            with wave.open(str(test_file), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(22050)
                wav.writeframes(audio)
                
            print(f"  ✓ Test audio saved: {test_file}")
            
    except Exception as e:
        print(f"\nERROR: Failed to create profile: {e}")
        return 1
        
    await tts.shutdown()
    return 0


async def list_profiles(args):
    """List all voice profiles."""
    config = Config()
    
    vc_config = VoiceCloningConfig(
        enabled=True,
        backend=args.backend,
        voices_dir=Path(args.voices_dir)
    )
    
    tts = VoiceCloningTTS(vc_config)
    
    # Just load profiles, don't need the model
    await tts._load_profiles()
    
    profiles = tts.list_profiles()
    
    if not profiles:
        print("No voice profiles found.")
        print(f"\nCreate one with:")
        print(f"  python voice_manager.py create <name> <reference_audio.wav>")
        return 0
        
    print(f"Voice Profiles ({len(profiles)}):")
    print("-" * 50)
    
    for name in profiles:
        profile = tts.profiles[name]
        is_default = " (default)" if name == tts.default_profile else ""
        
        print(f"\n  {name}{is_default}")
        if profile.description:
            print(f"    Description: {profile.description}")
        print(f"    Language: {profile.language}")
        print(f"    Reference files: {len(profile.reference_audio_paths)}")
        
    return 0


async def delete_profile(args):
    """Delete a voice profile."""
    vc_config = VoiceCloningConfig(
        enabled=True,
        backend="xtts",
        voices_dir=Path(args.voices_dir)
    )
    
    tts = VoiceCloningTTS(vc_config)
    await tts._load_profiles()
    
    if args.name not in tts.profiles:
        print(f"ERROR: Profile '{args.name}' not found")
        return 1
        
    if not args.yes:
        confirm = input(f"Delete voice profile '{args.name}'? [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return 0
            
    if await tts.delete_profile(args.name):
        print(f"✓ Deleted voice profile: {args.name}")
    else:
        print(f"ERROR: Failed to delete profile")
        return 1
        
    return 0


async def test_profile(args):
    """Test a voice profile by synthesizing text."""
    vc_config = VoiceCloningConfig(
        enabled=True,
        backend=args.backend,
        device="cuda" if args.gpu else "cpu",
        voices_dir=Path(args.voices_dir)
    )
    
    tts = VoiceCloningTTS(vc_config)
    
    print(f"Initializing {args.backend} backend...")
    if not await tts.initialize():
        print("ERROR: Failed to initialize")
        return 1
        
    if args.name not in tts.profiles:
        print(f"ERROR: Profile '{args.name}' not found")
        print(f"Available profiles: {tts.list_profiles()}")
        return 1
        
    print(f"Synthesizing with voice '{args.name}'...")
    
    try:
        audio = await tts.synthesize(args.text, args.name)
        
        output_file = Path(args.output)
        
        import wave
        with wave.open(str(output_file), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(vc_config.output_sample_rate)
            wav.writeframes(audio)
            
        duration = len(audio) / (vc_config.output_sample_rate * 2)
        print(f"\n✓ Saved: {output_file} ({duration:.1f}s)")
        
        # Try to play if available
        if args.play:
            try:
                import subprocess
                subprocess.run(["play", str(output_file)], check=True)
            except:
                try:
                    subprocess.run(["aplay", str(output_file)], check=True)
                except:
                    print("  (install sox or alsa-utils to auto-play)")
                    
    except Exception as e:
        print(f"\nERROR: Synthesis failed: {e}")
        return 1
        
    await tts.shutdown()
    return 0


async def set_default(args):
    """Set default voice profile."""
    vc_config = VoiceCloningConfig(
        enabled=True,
        backend="xtts",
        voices_dir=Path(args.voices_dir)
    )
    
    tts = VoiceCloningTTS(vc_config)
    await tts._load_profiles()
    
    if args.name not in tts.profiles:
        print(f"ERROR: Profile '{args.name}' not found")
        print(f"Available: {tts.list_profiles()}")
        return 1
        
    tts.default_profile = args.name
    await tts._save_profiles()
    
    print(f"✓ Set default voice profile: {args.name}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Manage voice cloning profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a voice profile from a recording
  python voice_manager.py create my_voice recording.wav
  
  # Create with multiple reference files
  python voice_manager.py create my_voice sample1.wav sample2.wav sample3.wav
  
  # List all profiles
  python voice_manager.py list
  
  # Test a profile
  python voice_manager.py test my_voice "Hello, this is a test!"
  
  # Delete a profile
  python voice_manager.py delete my_voice
  
Tips for reference audio:
  - Use 5-30 seconds of clean speech
  - Minimize background noise
  - Use consistent speaking style
  - Multiple samples can improve quality
        """
    )
    
    parser.add_argument(
        "--backend", "-b",
        default="xtts",
        choices=["xtts", "chatterbox", "openvoice", "fish_speech"],
        help="Voice cloning backend (default: xtts)"
    )
    parser.add_argument(
        "--voices-dir", "-d",
        default="./data/voices",
        help="Voice profiles directory"
    )
    parser.add_argument(
        "--gpu/--no-gpu",
        default=True,
        dest="gpu",
        help="Use GPU acceleration"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new voice profile")
    create_parser.add_argument("name", help="Profile name")
    create_parser.add_argument(
        "reference_audio",
        nargs="+",
        help="Reference audio file(s)"
    )
    create_parser.add_argument(
        "--language", "-l",
        default="en",
        help="Language code (default: en)"
    )
    create_parser.add_argument(
        "--description",
        default="",
        help="Profile description"
    )
    create_parser.add_argument(
        "--default",
        action="store_true",
        help="Set as default voice"
    )
    create_parser.add_argument(
        "--test",
        action="store_true",
        help="Generate test audio after creation"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List voice profiles")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a voice profile")
    delete_parser.add_argument("name", help="Profile name to delete")
    delete_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test a voice profile")
    test_parser.add_argument("name", help="Profile name")
    test_parser.add_argument("text", help="Text to synthesize")
    test_parser.add_argument(
        "--output", "-o",
        default="test_output.wav",
        help="Output file (default: test_output.wav)"
    )
    test_parser.add_argument(
        "--play", "-p",
        action="store_true",
        help="Play audio after synthesis"
    )
    
    # Set-default command
    default_parser = subparsers.add_parser("set-default", help="Set default voice profile")
    default_parser.add_argument("name", help="Profile name")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
        
    # Handle multiple reference files for create
    if args.command == "create" and len(args.reference_audio) == 1:
        args.reference_audio = args.reference_audio[0]
        
    # Run appropriate command
    commands = {
        "create": create_profile,
        "list": list_profiles,
        "delete": delete_profile,
        "test": test_profile,
        "set-default": set_default
    }
    
    return asyncio.run(commands[args.command](args))


if __name__ == "__main__":
    sys.exit(main())
