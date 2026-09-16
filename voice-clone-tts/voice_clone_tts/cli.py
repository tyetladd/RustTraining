"""Command line interface.

    vctts speak -v sample.mp3 -t "Привет!" -o out.wav
    vctts profile build sample.mp3 -o profiles/anna
    vctts speak -p profiles/anna -f article.txt -o article.wav
    vctts stress "Дорогая, замок на горе"
    vctts voice train -p profiles/anna -o voices/anna --vc rvc --epochs 300
    vctts speak -b silero --voice-model voices/anna -t "Привет!" -o out.wav
    vctts backends
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from voice_clone_tts import __version__
from voice_clone_tts.backends import (
    DEFAULT_BACKEND,
    backend_class,
    get_backend,
    iter_backend_info,
)
from voice_clone_tts.config import ASRConfig, PipelineConfig, ReferenceConfig, SynthesisConfig, TextConfig
from voice_clone_tts.errors import VoiceCloneError
from voice_clone_tts.profile import SpeakerProfile, build_profile
from voice_clone_tts.pipeline import prepare_text, synthesize
from voice_clone_tts.text.languages import get_language, list_languages, load_language_config
from voice_clone_tts.text.stress import StressStyle
from voice_clone_tts.vc import (
    DEFAULT_CONVERTER,
    VoiceModel,
    build_training_dataset,
    get_converter,
    iter_converter_info,
)

log = logging.getLogger("vctts")


# --------------------------------------------------------------------------
# argument plumbing
# --------------------------------------------------------------------------

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "--language-config", metavar="FILE",
        help="JSON file with extra languages (see docs: extending languages)",
    )


def _add_reference_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("reference audio")
    group.add_argument("--no-asr", action="store_true",
                       help="skip Whisper; use energy based segmentation and --language")
    group.add_argument("--asr-model", default="small",
                       help="Whisper model: tiny/base/small/medium/large-v3 (default: small)")
    group.add_argument("--asr-device", default="auto", help="cpu, cuda or auto (default: auto)")
    group.add_argument("--ref-seconds", type=float, default=60.0,
                       help="how much reference speech to keep (default: 60)")
    group.add_argument("--max-segments", type=int, default=8,
                       help="max reference clips to keep (default: 8)")


def _add_text_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("text front-end")
    group.add_argument("--no-normalize", action="store_true",
                       help="do not expand numbers, symbols and abbreviations")
    group.add_argument("--no-stress", action="store_true",
                       help="do not run silero-stress")
    group.add_argument("--stress-style", choices=[s.value for s in StressStyle],
                       help="how stress marks reach the model (default: the backend's preference)")
    group.add_argument("--stress-device", default="cpu", help="device for the accentor (default: cpu)")
    group.add_argument("--max-chars", type=int, help="max characters per synthesis chunk")
    group.add_argument("--ignore-word", action="append", default=[], metavar="WORD",
                       help="leave this word unstressed (repeatable)")


def _add_synthesis_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("synthesis")
    group.add_argument("-b", "--backend", default=DEFAULT_BACKEND,
                       help=f"TTS backend (default: {DEFAULT_BACKEND})")
    group.add_argument("--device", default="auto", help="cpu, cuda, mps or auto (default: auto)")
    group.add_argument("--speed", type=float, default=1.0, help="speaking rate (default: 1.0)")
    group.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    group.add_argument("--seed", type=int, help="seed for reproducible output")
    group.add_argument("--pause", type=float, default=0.35,
                       help="seconds of silence between chunks (default: 0.35)")
    group.add_argument("--output-sample-rate", type=int, help="resample the result")
    group.add_argument("--backend-option", action="append", default=[], metavar="KEY=VALUE",
                       help="backend specific option (repeatable), e.g. --backend-option voice=baya")
    group.add_argument("--accept-coqui-license", action="store_true",
                       help="acknowledge the non-commercial Coqui Public Model License for XTTS")


def _add_conversion_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("voice conversion (cloning)")
    group.add_argument("--voice-model", metavar="DIR",
                       help="trained voice-conversion model; recolours the synthesized audio")
    group.add_argument("--vc", choices=["rvc", "sovits"],
                       help="converter to use (default: the one recorded in the model)")
    group.add_argument("--transpose", default="0", metavar="N",
                       help="pitch shift in semitones, or 'auto' to match the target speaker")
    group.add_argument("--vc-device", default="auto", help="device for conversion (default: auto)")
    group.add_argument("--vc-option", action="append", default=[], metavar="KEY=VALUE",
                       help="converter option (repeatable), e.g. --vc-option applio_dir=~/Applio")


def _parse_key_values(pairs: list[str], flag: str) -> dict:
    options: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise VoiceCloneError(f"{flag} expects KEY=VALUE, got '{pair}'")
        key, value = pair.split("=", 1)
        try:
            options[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            options[key.strip()] = value
    return options


def _parse_backend_options(pairs: list[str]) -> dict:
    return _parse_key_values(pairs, "--backend-option")


def _parse_transpose(value: str) -> int | str:
    if str(value).strip().lower() == "auto":
        return "auto"
    try:
        return int(value)
    except (TypeError, ValueError):
        raise VoiceCloneError(f"--transpose expects an integer or 'auto', got '{value}'") from None


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    reference = ReferenceConfig(
        target_total_sec=getattr(args, "ref_seconds", 60.0),
        max_segments=getattr(args, "max_segments", 8),
    )
    asr = ASRConfig(
        enabled=not getattr(args, "no_asr", False),
        model=getattr(args, "asr_model", "small"),
        device=getattr(args, "asr_device", "auto"),
    )
    text = TextConfig(
        normalize=not getattr(args, "no_normalize", False),
        stress=not getattr(args, "no_stress", False),
        stress_style=getattr(args, "stress_style", None),
        stress_device=getattr(args, "stress_device", "cpu"),
        max_chars=getattr(args, "max_chars", None),
        words_to_ignore=list(getattr(args, "ignore_word", []) or []),
    )
    synthesis = SynthesisConfig(
        backend=getattr(args, "backend", DEFAULT_BACKEND),
        speed=getattr(args, "speed", 1.0),
        temperature=getattr(args, "temperature", 0.7),
        seed=getattr(args, "seed", None),
        device=getattr(args, "device", "auto"),
        pause_between_chunks=getattr(args, "pause", 0.35),
        output_sample_rate=getattr(args, "output_sample_rate", None),
        backend_options=_parse_backend_options(getattr(args, "backend_option", []) or []),
        voice_model=getattr(args, "voice_model", None),
        converter=getattr(args, "vc", None),
        transpose=_parse_transpose(getattr(args, "transpose", "0") or "0"),
        converter_device=getattr(args, "vc_device", "auto"),
        converter_options=_parse_key_values(
            getattr(args, "vc_option", []) or [], "--vc-option"
        ),
    )
    return PipelineConfig(
        language=getattr(args, "language", None),
        reference=reference,
        asr=asr,
        text=text,
        synthesis=synthesis,
    )


def _read_text(args: argparse.Namespace) -> str:
    if getattr(args, "text", None):
        return args.text
    if getattr(args, "text_file", None):
        path = Path(args.text_file)
        if str(path) == "-":
            return sys.stdin.read()
        return path.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data
    raise VoiceCloneError("no text to speak: pass --text, --text-file or pipe it on stdin")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_speak(args: argparse.Namespace) -> int:
    if args.accept_coqui_license:
        os.environ["VCTTS_ACCEPT_COQUI_LICENSE"] = "1"
    config = _config_from_args(args)
    text = _read_text(args)

    if args.dry_run:
        spec = get_language(args.language or config.fallback_language)
        style = args.stress_style or backend_class(args.backend).native_stress_style
        prepared, chunks = prepare_text(text, spec, config, stress_style=style)
        print(f"language: {spec.code}   chunks: {len(chunks)}   stress style: "
              f"{StressStyle.parse(style).value}")
        for index, chunk in enumerate(chunks, start=1):
            print(f"\n[{index:>3}] ({len(chunk)} chars)\n{chunk}")
        return 0

    result = synthesize(
        voice=args.voice,
        text=text,
        language=args.language,
        profile=args.profile,
        out_path=args.out,
        config=config,
    )
    if args.save_profile and result.profile is not None:
        saved = result.profile.save(args.save_profile)
        result.profile.is_temporary = False
        print(f"speaker profile saved to {saved}")
    elif result.profile is not None:
        result.profile.cleanup()
    print(result.describe())
    return 0


def cmd_profile_build(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    profile = build_profile(
        args.audio,
        out_dir=args.out,
        name=args.name,
        config=config,
        overwrite=args.overwrite,
    )
    print(profile.describe())
    return 0


def cmd_profile_show(args: argparse.Namespace) -> int:
    profile = SpeakerProfile.load(args.directory)
    if args.json:
        print(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(profile.describe())
    return 0


def cmd_stress(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    text = _read_text(args)
    spec = get_language(args.language)
    prepared, chunks = prepare_text(
        text, spec, config, stress_style=args.stress_style or StressStyle.PLUS
    )
    print(prepared)
    if args.show_chunks:
        print(f"\n--- {len(chunks)} chunk(s) ---")
        for index, chunk in enumerate(chunks, start=1):
            print(f"[{index:>3}] {chunk}")
    return 0


def cmd_voice_train(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    out_dir = Path(args.out)
    profile = SpeakerProfile.load(args.profile) if args.profile else None
    source = args.source or (profile.source_audio if profile else None)

    dataset = build_training_dataset(
        source,
        profile=profile,
        out_dir=Path(args.dataset_dir) if args.dataset_dir else out_dir / "dataset",
        speaker=args.speaker or (profile.name if profile else None),
        sample_rate=args.sample_rate,
        max_clip_sec=args.max_clip_seconds,
        min_clip_sec=args.min_clip_seconds,
        overwrite=args.overwrite,
    )
    print(dataset.describe())
    if args.dataset_only:
        return 0

    converter = get_converter(
        args.vc or DEFAULT_CONVERTER,
        device=args.vc_device,
        sample_rate=args.sample_rate,
        **_parse_key_values(args.vc_option or [], "--vc-option"),
    )
    if not type(converter).is_available():
        raise VoiceCloneError(
            f"voice converter '{converter.name}' is not usable here.\n{converter.install_hint}"
        )
    model = converter.train(
        dataset, out_dir, name=args.name, epochs=args.epochs, resume=args.resume
    )
    print()
    print(model.describe())
    print(f"\nUse it with:  vctts speak -b silero --voice-model {out_dir} -t \"…\"")
    return 0


def cmd_voice_show(args: argparse.Namespace) -> int:
    model = VoiceModel.load(args.directory)
    if args.json:
        print(json.dumps(model.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(model.describe())
    return 0


def cmd_converters(args: argparse.Namespace) -> int:
    rows = iter_converter_info()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    width = max(len(row["name"]) for row in rows)
    for row in rows:
        mark = "✓" if row["available"] else "✗"
        print(f"{mark} {row['name']:<{width}}  {row['display_name']}")
        if row["description"]:
            print(f"  {' ' * width}  {row['description']}")
        if not row["available"]:
            print(f"  {' ' * width}  install:  {row['install_hint']}")
    return 0


def cmd_backends(args: argparse.Namespace) -> int:
    rows = list(iter_backend_info())
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    width = max(len(row["name"]) for row in rows)
    for row in rows:
        mark = "✓" if row["available"] else "✗"
        languages = row["languages"] if isinstance(row["languages"], str) else ", ".join(row["languages"])
        clone = "clones voice" if row["clones_voice"] else "preset voices"
        print(f"{mark} {row['name']:<{width}}  {row['display_name']}")
        print(f"  {' ' * width}  {clone}; languages: {languages}; "
              f"stress: {row['stress_style']}; {row['sample_rate']} Hz")
        if row["description"]:
            print(f"  {' ' * width}  {row['description']}")
        if not row["available"] and row["install_extra"]:
            print(f"  {' ' * width}  install:  pip install 'voice-clone-tts[{row['install_extra']}]'")
    return 0


def cmd_languages(args: argparse.Namespace) -> int:
    specs = list_languages()
    if args.json:
        print(json.dumps([spec.to_dict() for spec in specs], ensure_ascii=False, indent=2))
        return 0
    for spec in specs:
        stress = f"silero-stress ({spec.stress_lang})" if spec.has_stress else "—"
        xtts = spec.xtts_code or "—"
        print(f"{spec.code:<4} {spec.name:<12} stress: {stress:<24} xtts: {xtts:<6} "
              f"aliases: {', '.join(spec.aliases) or '—'}")
    return 0


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vctts",
        description="Clone a voice from an audio sample and read text with it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"voice-clone-tts {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # speak -----------------------------------------------------------------
    speak = subparsers.add_parser("speak", help="synthesize text with a cloned voice")
    speak.add_argument("-v", "--voice", metavar="AUDIO",
                       help="reference recording (mp3, wav, m4a, …)")
    speak.add_argument("-p", "--profile", metavar="DIR",
                       help="use a profile built earlier instead of a raw recording")
    speak.add_argument("-t", "--text", help="text to speak")
    speak.add_argument("-f", "--text-file", metavar="FILE", help="read the text from a file ('-' for stdin)")
    speak.add_argument("-l", "--language", help="ru, en, … (default: detected from the reference)")
    speak.add_argument("-o", "--out", default="output.wav", metavar="FILE",
                       help="output file; .wav or .mp3 (default: output.wav)")
    speak.add_argument("--save-profile", metavar="DIR", help="keep the built profile for reuse")
    speak.add_argument("--dry-run", action="store_true",
                       help="print the normalized, stressed, chunked text and stop")
    _add_reference_options(speak)
    _add_text_options(speak)
    _add_synthesis_options(speak)
    _add_conversion_options(speak)
    _add_common(speak)
    speak.set_defaults(func=cmd_speak)

    # profile ---------------------------------------------------------------
    profile = subparsers.add_parser("profile", help="build and inspect speaker profiles")
    profile_sub = profile.add_subparsers(dest="profile_command", required=True)

    profile_build = profile_sub.add_parser("build", help="analyze a recording into a reusable profile")
    profile_build.add_argument("audio", help="reference recording (mp3, wav, …)")
    profile_build.add_argument("-o", "--out", required=True, metavar="DIR", help="where to store it")
    profile_build.add_argument("--name", help="profile name (default: file stem)")
    profile_build.add_argument("-l", "--language", help="force the language instead of detecting it")
    profile_build.add_argument("--overwrite", action="store_true", help="rebuild an existing profile")
    _add_reference_options(profile_build)
    _add_common(profile_build)
    profile_build.set_defaults(func=cmd_profile_build)

    profile_show = profile_sub.add_parser("show", help="print a profile summary")
    profile_show.add_argument("directory")
    profile_show.add_argument("--json", action="store_true")
    _add_common(profile_show)
    profile_show.set_defaults(func=cmd_profile_show)

    # stress ----------------------------------------------------------------
    stress = subparsers.add_parser("stress", help="show normalized text with stress marks")
    stress.add_argument("text", nargs="?", help="text (or pipe it on stdin)")
    stress.add_argument("-f", "--text-file", metavar="FILE")
    stress.add_argument("-l", "--language", default="ru")
    stress.add_argument("--stress-style", choices=[s.value for s in StressStyle], default="plus")
    stress.add_argument("--no-normalize", action="store_true")
    stress.add_argument("--no-stress", action="store_true")
    stress.add_argument("--stress-device", default="cpu")
    stress.add_argument("--max-chars", type=int)
    stress.add_argument("--ignore-word", action="append", default=[], metavar="WORD")
    stress.add_argument("--show-chunks", action="store_true", help="also print the chunk split")
    _add_common(stress)
    stress.set_defaults(func=cmd_stress)

    # voice (conversion models) ----------------------------------------------
    voice = subparsers.add_parser("voice", help="train and inspect voice-conversion models")
    voice_sub = voice.add_subparsers(dest="voice_command", required=True)

    voice_train = voice_sub.add_parser(
        "train", help="train a voice-conversion model on a speaker (needs a GPU)"
    )
    voice_train.add_argument("-o", "--out", required=True, metavar="DIR",
                             help="where to store the trained model")
    voice_train.add_argument("-p", "--profile", metavar="DIR",
                             help="speaker profile to take the recording and name from")
    voice_train.add_argument("-s", "--source", metavar="AUDIO",
                             help="original recording (preferred over the profile clips)")
    voice_train.add_argument("--name", help="model name (default: speaker name)")
    voice_train.add_argument("--speaker", help="speaker id inside the dataset")
    voice_train.add_argument("--vc", choices=["rvc", "sovits"], default=DEFAULT_CONVERTER,
                             help=f"converter to train with (default: {DEFAULT_CONVERTER})")
    voice_train.add_argument("--epochs", type=int, help="training epochs (default: 300)")
    voice_train.add_argument("--sample-rate", type=int, default=40_000,
                             help="training sample rate: 40000 for RVC, 44100 for so-vits-svc")
    voice_train.add_argument("--max-clip-seconds", type=float, default=10.0,
                             help="max length of one training clip (default: 10)")
    voice_train.add_argument("--min-clip-seconds", type=float, default=2.0,
                             help="drop clips shorter than this (default: 2)")
    voice_train.add_argument("--dataset-dir", metavar="DIR", help="where to put the training clips")
    voice_train.add_argument("--dataset-only", action="store_true",
                             help="only prepare the dataset, do not train")
    voice_train.add_argument("--resume", action="store_true", help="continue a previous run")
    voice_train.add_argument("--overwrite", action="store_true", help="rebuild an existing dataset")
    voice_train.add_argument("--vc-device", default="auto")
    voice_train.add_argument("--vc-option", action="append", default=[], metavar="KEY=VALUE")
    _add_common(voice_train)
    voice_train.set_defaults(func=cmd_voice_train)

    voice_show = voice_sub.add_parser("show", help="print a trained model summary")
    voice_show.add_argument("directory")
    voice_show.add_argument("--json", action="store_true")
    _add_common(voice_show)
    voice_show.set_defaults(func=cmd_voice_show)

    # listings --------------------------------------------------------------
    backends = subparsers.add_parser("backends", help="list TTS backends and their availability")
    backends.add_argument("--json", action="store_true")
    _add_common(backends)
    backends.set_defaults(func=cmd_backends)

    converters = subparsers.add_parser(
        "converters", help="list voice-conversion toolchains and their availability"
    )
    converters.add_argument("--json", action="store_true")
    _add_common(converters)
    converters.set_defaults(func=cmd_converters)

    languages = subparsers.add_parser("languages", help="list supported languages")
    languages.add_argument("--json", action="store_true")
    _add_common(languages)
    languages.set_defaults(func=cmd_languages)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING if getattr(args, "quiet", False) else logging.INFO
    if getattr(args, "verbose", False):
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    try:
        if getattr(args, "language_config", None):
            load_language_config(args.language_config)
        return int(args.func(args) or 0)
    except VoiceCloneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
