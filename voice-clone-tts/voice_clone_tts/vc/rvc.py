"""RVC driver built on Applio.

Applio (https://github.com/IAHispano/Applio, MIT) is the maintained member of
the RVC family. It is a repository rather than a package, and its ``core.py``
exposes the whole pipeline as a CLI::

    python core.py preprocess --model-name anna --dataset-path <clips> --sample-rate 40000
    python core.py extract    --model-name anna --f0-method rmvpe --sample-rate 40000
    python core.py train      --model-name anna --total-epoch 300 --batch-size 8
    python core.py index      --model-name anna
    python core.py infer --input-path in.wav --output-path out.wav \\
                         --pth-path anna.pth --index-path anna.index --pitch 0

Point the driver at your checkout with ``VCTTS_APPLIO_DIR=/path/to/Applio`` or
``--vc-option applio_dir=/path/to/Applio``. Training artifacts land in
``<applio>/logs/<model-name>/`` and the driver copies the final ``.pth``/
``.index`` into the voice model directory so it stays self-contained.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from voice_clone_tts import audio as audio_utils
from voice_clone_tts.vc.base import (
    STAGES,
    VoiceConversionError,
    VoiceConverter,
    VoiceModel,
    register_converter,
)
from voice_clone_tts.vc.dataset import TrainingDataset
from voice_clone_tts.vc.runner import run_command

log = logging.getLogger(__name__)

APPLIO_ENV = "VCTTS_APPLIO_DIR"
SUPPORTED_SAMPLE_RATES = (32_000, 40_000, 48_000)
DEFAULT_SAMPLE_RATE = 40_000
DEFAULT_EPOCHS = 300

# core.py catches the exit code of the script it runs and prints this instead,
# so a failed step still exits 0. Without these patterns a broken training run
# looks like a successful one that produced no weights.
APPLIO_FAILURE_PATTERNS = (
    "failed for model",
    "Traceback (most recent call last)",
    # Оба шага печатают это и выходят с нулевым кодом.
    "no audio files found in the dataset path",
    "not enough data present in the training set",
)

# Applio раскладывает промежуточные данные по этим папкам внутри logs/<model>.
SLICED_DIR = "sliced_audios"
FEATURES_DIR = "extracted"
FILELIST = "filelist.txt"
STAGE_DIRS = (SLICED_DIR, FEATURES_DIR, "f0", "f0_voiced")
"""generate_filelist() берёт ПЕРЕСЕЧЕНИЕ имён из этих четырёх папок.

Пустая любая из них — и filelist.txt оказывается пустым, а обучение
останавливается на «Not enough data», не сказав, чего именно не хватило."""
MIN_BATCHES = 3
"""train.py останавливается, если батчей меньше трёх — отсюда минимум фрагментов."""


def find_applio(explicit: str | Path | None = None) -> Path | None:
    """Locate an Applio checkout: explicit path, env var, then ./Applio."""
    candidates = [explicit, os.environ.get(APPLIO_ENV), Path.cwd() / "Applio"]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if (path / "core.py").exists():
            return path
    return None


@register_converter
class RVCConverter(VoiceConverter):
    """Trains and applies an RVC model through an Applio checkout."""

    name = "rvc"
    display_name = "RVC (Applio)"
    description = "Retrieval-based voice conversion; closest to the target voice, needs a GPU"
    install_hint = (
        "git clone https://github.com/IAHispano/Applio && cd Applio && "
        "pip install -r requirements.txt, then set VCTTS_APPLIO_DIR=/path/to/Applio"
    )
    supports_training = True
    needs_gpu = True

    def __init__(self, *, device: str = "auto", **options) -> None:
        super().__init__(device=device, **options)
        self.applio_dir = find_applio(options.get("applio_dir"))
        self.sample_rate = int(options.get("sample_rate", DEFAULT_SAMPLE_RATE))
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise VoiceConversionError(
                f"RVC supports {SUPPORTED_SAMPLE_RATES} Hz, got {self.sample_rate}"
            )
        self.f0_method = options.get("f0_method", "rmvpe")
        self.batch_size = int(options.get("batch_size", 8))
        self.save_every_epoch = int(options.get("save_every_epoch", 50))
        # Препроцессинг и извлечение признаков — пакетная работа на выделенной
        # машине, делить её не с кем: берём все ядра.
        self.cpu_cores = int(options.get("cpu_cores", os.cpu_count() or 2))
        self.index_rate = float(options.get("index_rate", 0.3))
        self.protect = float(options.get("protect", 0.33))
        self.volume_envelope = float(options.get("volume_envelope", 1.0))
        self.python = options.get("python", sys.executable)
        self.logs_dir = options.get("logs_dir")
        """Persistent parent for Applio's per-model logs (Drive, /kaggle/working…).

        Cloud sessions die before a long training does, so the checkpoints must
        not live inside the ephemeral checkout.
        """
        self.timeout = options.get("timeout")

    @classmethod
    def is_available(cls) -> bool:
        return find_applio() is not None

    def _checkout(self) -> Path:
        if self.applio_dir is None:
            raise VoiceConversionError(
                "no Applio checkout found. Set "
                f"{APPLIO_ENV}=/path/to/Applio or pass --vc-option applio_dir=…\n"
                + self.install_hint
            )
        return self.applio_dir

    def _gpu_argument(self) -> str:
        device = self.resolve_device()
        if device.startswith("cuda"):
            _, _, index = device.partition(":")
            return index or "0"
        return "-"  # Applio reads "-" as CPU

    def _prepare_logs(self, name: str) -> Path:
        """Return Applio's logs dir for `name`, linked to persistent storage if asked."""
        target = self._checkout() / "logs" / name
        if not self.logs_dir:
            target.mkdir(parents=True, exist_ok=True)
            return target

        persistent = (Path(self.logs_dir).expanduser() / name).resolve()
        persistent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.is_symlink():
            if target.resolve() == persistent:
                return target
            target.unlink()
        elif target.exists():
            existing = list(target.iterdir())
            if existing and any(persistent.iterdir()):
                raise VoiceConversionError(
                    f"both {target} and {persistent} hold training data; "
                    "remove one of them before linking"
                )
            for item in existing:
                shutil.move(str(item), persistent / item.name)
            target.rmdir()

        try:
            target.symlink_to(persistent, target_is_directory=True)
            log.info("Applio logs for '%s' now live in %s", name, persistent)
        except OSError as exc:  # pragma: no cover - Windows without privileges
            log.warning(
                "could not link %s -> %s (%s); checkpoints stay inside the checkout",
                target, persistent, exc,
            )
            target.mkdir(parents=True, exist_ok=True)
        return target

    def _core(self, *args: object, label: str) -> None:
        checkout = self._checkout()
        run_command(
            [self.python, "core.py", *args],
            cwd=checkout,
            label=label,
            timeout=self.timeout,
            failure_patterns=APPLIO_FAILURE_PATTERNS,
        )

    # -- training ---------------------------------------------------------
    def _partial_model(self, name, out_dir, dataset, logs_dir, stage, stats) -> VoiceModel:
        """Модель без весов: конвейер намеренно остановлен на `stage`."""
        log.info("остановка после стадии '%s' — весов не будет", stage)
        model = VoiceModel(
            name=name,
            directory=Path(out_dir),
            converter=self.name,
            speaker=dataset.speaker,
            sample_rate=self.sample_rate,
            source_profile=dataset.source,
            median_f0=dataset.median_f0,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            train_stats={"stopped_after": stage, **stats},
            metadata={"logs_dir": str(logs_dir)},
        )
        model.save()
        return model

    def _pitch_hint(self) -> str:
        """Подсказка для пустых f0: там ошибка не печатается вообще.

        run_pitch_extraction() ждёт задачи через concurrent.futures.wait() и
        не забирает результат, поэтому исключение в воркере (обычно — загрузка
        модели предиктора) теряется, а шаг сообщает «completed».
        """
        checkout = self.applio_dir or Path("<Applio>")
        return (
            "Шаг питча ошибку не печатает: Applio ждёт воркеры через "
            "concurrent.futures.wait() и не забирает исключение. Увидеть её можно, "
            "загрузив предиктор в текущем процессе:\n"
            f"  cd {checkout} && python3 -c \"import sys; sys.path.append('.'); "
            "from rvc.train.extract.extract import FeatureInput; "
            f"FeatureInput(f0_method='{self.f0_method}', device='cuda:0')\"\n"
            f"Проверьте также веса предиктора: ls -la {checkout}/rvc/models/predictors/\n"
            "Обходной путь — другой метод: --vc-option f0_method=fcpe (или crepe)."
        )

    def _require_slices(self, logs_dir: Path, name: str) -> int:
        """Убедиться, что preprocess действительно нарезал аудио.

        Он сообщает об успехе, даже если каждый файл упал при чтении: ошибки
        печатаются построчно («Error processing audio: …») и проглатываются.
        Без этой проверки пустота всплывёт только на обучении.
        """
        sliced = sorted((logs_dir / SLICED_DIR).glob("*.wav"))
        if not sliced:
            raise VoiceConversionError(
                f"preprocess не нарезал ни одного фрагмента в {logs_dir / SLICED_DIR}.\n"
                "Ищите в логе выше строки «Error processing audio:» — там причина "
                "по каждому файлу.\n"
                "Проверить вручную:\n"
                f"  cd {logs_dir.parent.parent} && python3 core.py preprocess "
                f"--model-name {name} --dataset-path <папка с клипами> "
                f"--sample-rate {self.sample_rate} --cpu-cores {self.cpu_cores}"
            )
        log.info("preprocess: %d фрагментов в %s", len(sliced), logs_dir / SLICED_DIR)
        return len(sliced)

    def _require_features(self, logs_dir: Path, name: str) -> int:
        """Убедиться, что extract дал всё, из чего собирается filelist.

        Applio строит filelist.txt как пересечение имён в четырёх папках, и
        если хоть одна пуста, список выходит пустым, а обучение падает с
        «Not enough data present in the training set» — без единого намёка,
        какой именно стадии не хватило. Повторяем ту же логику здесь.
        """
        stems: dict[str, set[str]] = {}
        for directory in STAGE_DIRS:
            path = logs_dir / directory
            stems[directory] = (
                {item.name.split(".")[0] for item in path.glob("*")} if path.is_dir() else set()
            )
        common = set.intersection(*stems.values()) if stems else set()
        counts = ", ".join(f"{directory}: {len(names)}" for directory, names in stems.items())

        filelist = logs_dir / FILELIST
        entries = (
            [line for line in filelist.read_text(encoding="utf-8").splitlines() if line.strip()]
            if filelist.exists()
            else []
        )

        if not common or not entries:
            empty = [directory for directory, names in stems.items() if not names]
            reason = (
                f"пусто в {', '.join(empty)}" if empty
                else "имена файлов в этих папках не совпадают"
            )
            raise VoiceConversionError(
                f"extract не подготовил данные для обучения в {logs_dir}.\n"
                f"  {counts}\n"
                f"  общих имён: {len(common)}, строк в {FILELIST}: {len(entries)}\n"
                f"Причина: {reason}.\n"
                "Applio собирает filelist.txt как пересечение имён этих четырёх папок, "
                "поэтому пустая любая из них обнуляет обучение.\n"
                + (self._pitch_hint() if {"f0", "f0_voiced"} & set(empty) else
                   "Ищите ошибку в строках [applio extract] выше.")
            )

        minimum = MIN_BATCHES * self.batch_size
        if len(entries) < minimum:
            raise VoiceConversionError(
                f"для обучения слишком мало данных: {len(entries)} фрагментов, "
                f"а при batch_size={self.batch_size} нужно минимум {minimum} "
                f"({MIN_BATCHES} батча).\n"
                "Иначе Applio молча остановится с «Not enough data present in the "
                "training set».\n"
                f"Варианты: уменьшить batch_size (--vc-option batch_size="
                f"{max(1, len(entries) // MIN_BATCHES)}) или записать больше речи."
            )
        log.info("extract: %s; строк в %s: %d", counts, FILELIST, len(entries))
        return len(entries)

    @staticmethod
    def _require_checkpoints(logs_dir: Path, name: str) -> None:
        """Fail loudly right after training, before the index step hides it."""
        if any(logs_dir.glob("*.pth")):
            return
        listing = sorted(path.name for path in logs_dir.glob("*"))[:20] if logs_dir.exists() else []
        raise VoiceConversionError(
            f"training '{name}' produced no .pth weights in {logs_dir}.\n"
            f"Applio's own step must have failed — its output is in the log above.\n"
            f"What is in that directory now: {listing or 'ничего'}\n"
            "Reproduce it directly to see the error in full:\n"
            f"  cd {logs_dir.parent.parent} && python3 core.py train --model-name {name} "
            "--total-epoch 5 --batch-size 8 --save-every-epoch 1 --sample-rate 40000 --gpu 0"
        )

    @staticmethod
    def _collect_artifacts(logs_dir: Path) -> tuple[Path, Path | None]:
        weights = [p for p in logs_dir.glob("*.pth") if not p.name.startswith(("G_", "D_"))]
        if not weights:
            weights = sorted(logs_dir.glob("G_*.pth"))
        if not weights:
            raise VoiceConversionError(f"training produced no .pth weights in {logs_dir}")
        checkpoint = max(weights, key=lambda path: path.stat().st_mtime)
        indexes = sorted(logs_dir.glob("*.index"), key=lambda path: path.stat().st_mtime)
        return checkpoint, (indexes[-1] if indexes else None)

    def train(
        self,
        dataset: TrainingDataset,
        out_dir: str | Path,
        *,
        name: str | None = None,
        epochs: int | None = None,
        resume: bool = False,
        stop_after: str | None = None,
    ) -> VoiceModel:
        if stop_after is not None and stop_after not in STAGES:
            raise VoiceConversionError(
                f"unknown stage '{stop_after}'. Known: {', '.join(STAGES)}"
            )
        checkout = self._checkout()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = name or dataset.speaker
        epochs = int(epochs or DEFAULT_EPOCHS)
        logs_dir = self._prepare_logs(name)

        if not resume:
            self._core(
                "preprocess",
                "--model-name", name,
                "--dataset-path", dataset.speaker_dir.resolve(),
                "--sample-rate", self.sample_rate,
                "--cpu-cores", self.cpu_cores,
                label="applio preprocess",
            )

        # Проверки идут и при --resume: там подготовка пропускается, и без них
        # прогон уткнулся бы в те же пустые данные, что и в прошлый раз.
        slices = self._require_slices(logs_dir, name)
        if stop_after == "preprocess":
            return self._partial_model(name, out_dir, dataset, logs_dir, "preprocess",
                                       {"slices": slices})

        if not resume:
            self._core(
                "extract",
                "--model-name", name,
                "--f0-method", self.f0_method,
                "--sample-rate", self.sample_rate,
                "--cpu-cores", self.cpu_cores,
                "--gpu", self._gpu_argument(),
                label="applio extract",
            )

        entries = self._require_features(logs_dir, name)
        if stop_after == "extract":
            return self._partial_model(name, out_dir, dataset, logs_dir, "extract",
                                       {"filelist_entries": entries})

        log.info("training RVC model '%s' for %d epochs — this is the long part", name, epochs)
        self._core(
            "train",
            "--model-name", name,
            "--total-epoch", epochs,
            "--batch-size", self.batch_size,
            "--save-every-epoch", self.save_every_epoch,
            "--sample-rate", self.sample_rate,
            "--gpu", self._gpu_argument(),
            label="applio train",
        )
        self._require_checkpoints(logs_dir, name)
        self._core("index", "--model-name", name, label="applio index")

        checkpoint, index = self._collect_artifacts(logs_dir)
        local_checkpoint = out_dir / checkpoint.name
        shutil.copy2(checkpoint, local_checkpoint)
        local_index = None
        if index is not None:
            local_index = out_dir / index.name
            shutil.copy2(index, local_index)
        else:
            log.warning("no .index file was produced; conversion will run without retrieval")

        model = VoiceModel(
            name=name,
            directory=out_dir,
            converter=self.name,
            checkpoint=local_checkpoint,
            index=local_index,
            speaker=dataset.speaker,
            sample_rate=self.sample_rate,
            source_profile=dataset.source,
            median_f0=dataset.median_f0,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            train_stats={
                "epochs": epochs,
                "clips": len(dataset.clips),
                "dataset_minutes": round(dataset.total_duration / 60, 2),
                "f0_method": self.f0_method,
                "batch_size": self.batch_size,
            },
            metadata={"applio_dir": str(checkout), "logs_dir": str(logs_dir)},
        )
        model.save()
        log.info("trained voice model saved to %s", out_dir)
        return model

    # -- inference --------------------------------------------------------
    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: VoiceModel,
        *,
        transpose: int = 0,
    ) -> np.ndarray:
        self._checkout()
        if model.checkpoint is None:
            raise VoiceConversionError(f"voice model '{model.name}' has no checkpoint")
        if not -24 <= transpose <= 24:
            raise VoiceConversionError(f"transpose must be within ±24 semitones, got {transpose}")

        workdir = Path(tempfile.mkdtemp(prefix="vctts-rvc-"))
        source = workdir / "in.wav"
        target = workdir / "out.wav"
        try:
            audio_utils.save_audio(source, audio, sample_rate)
            args = [
                "infer",
                "--input-path", source.resolve(),
                "--output-path", target.resolve(),
                "--pth-path", model.checkpoint.resolve(),
                "--pitch", int(transpose),
                "--f0-method", self.f0_method,
                "--index-rate", self.index_rate,
                "--protect", self.protect,
                "--volume-envelope", self.volume_envelope,
            ]
            if model.index is not None:
                args += ["--index-path", model.index.resolve()]
            self._core(*args, label="applio infer")
            if not target.exists():
                raise VoiceConversionError("applio infer produced no output file")
            converted, converted_sr = audio_utils.load_audio(target, mono=True)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        if converted_sr != sample_rate:
            converted = audio_utils.resample(converted, converted_sr, sample_rate)
        return converted.astype(np.float32)
