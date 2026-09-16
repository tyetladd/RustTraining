"""Judging a reference recording before it costs GPU hours.

Voice conversion copies whatever is in the recording, noise and room included,
and training a model on a bad recording takes just as long as training on a
good one. This module turns the cheap measurements from :mod:`audio` into a
verdict with concrete numbers, so a recording can be fixed while the
microphone is still set up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voice_clone_tts import audio as audio_utils

log = logging.getLogger(__name__)

OK, WARN, FAIL = "ok", "warn", "fail"
_RANK = {OK: 0, WARN: 1, FAIL: 2}
_MARK = {OK: "✓", WARN: "!", FAIL: "✗"}

# Minutes of net speech. VC training needs orders of magnitude more than the
# few-shot conditioning XTTS does.
DURATION_TARGETS = {
    "vc": (900.0, 300.0),      # 15 min comfortable, 5 min workable
    "xtts": (60.0, 20.0),      # 1 min comfortable, 20 s workable
}

# Typical speaking ranges, used only to pick a same-register preset speaker.
MALE_RANGE = (80.0, 160.0)
FEMALE_RANGE = (160.0, 260.0)
SILERO_VOICES = {"male": ("aidar", "eugene"), "female": ("xenia", "baya", "kseniya")}


@dataclass(frozen=True)
class Finding:
    level: str
    message: str

    def render(self) -> str:
        return f"{_MARK[self.level]} {self.message}"


@dataclass
class RecordingReport:
    """Everything worth knowing about a candidate recording."""

    path: Path
    purpose: str
    duration: float
    speech_duration: float
    sample_rate: int
    source: dict = field(default_factory=dict)
    rms_dbfs: float = -120.0
    peak_dbfs: float = -120.0
    clipping_ratio: float = 0.0
    silence_ratio: float = 0.0
    snr_db: float = 0.0
    median_f0: float = 0.0
    findings: list[Finding] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return max((f.level for f in self.findings), key=lambda level: _RANK[level], default=OK)

    @property
    def register(self) -> str:
        if MALE_RANGE[0] <= self.median_f0 < MALE_RANGE[1]:
            return "male"
        if FEMALE_RANGE[0] <= self.median_f0 <= FEMALE_RANGE[1]:
            return "female"
        return "unknown"

    def suggested_voices(self) -> tuple[str, ...]:
        """Silero speakers in the same register — the smaller the shift, the cleaner the result."""
        return SILERO_VOICES.get(self.register, SILERO_VOICES["female"] + SILERO_VOICES["male"])

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "purpose": self.purpose,
            "verdict": self.verdict,
            "duration": round(self.duration, 2),
            "speech_duration": round(self.speech_duration, 2),
            "sample_rate": self.sample_rate,
            "source": self.source,
            "rms_dbfs": round(self.rms_dbfs, 2),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "clipping_ratio": round(self.clipping_ratio, 6),
            "silence_ratio": round(self.silence_ratio, 4),
            "snr_db": round(self.snr_db, 2),
            "median_f0": round(self.median_f0, 1),
            "register": self.register,
            "suggested_voices": list(self.suggested_voices()),
            "findings": [{"level": f.level, "message": f.message} for f in self.findings],
        }

    def describe(self) -> str:
        headline = {
            OK: "запись годится",
            WARN: "запись пригодна, но есть замечания",
            FAIL: "запись лучше переписать",
        }[self.verdict]
        lines = [
            f"{self.path.name}: {headline}",
            "",
            f"  длительность   : {self.duration / 60:.1f} мин "
            f"(речи {self.speech_duration / 60:.1f} мин)",
            f"  формат         : {self.source.get('format', '?')} "
            f"{self.source.get('subtype', '')} {self.source.get('sample_rate', '?')} Гц, "
            f"каналов {self.source.get('channels', '?')}"
            + (f", ~{self.source['kbps']} кбит/с" if self.source.get("kbps") else ""),
            f"  уровень        : RMS {self.rms_dbfs:.1f} dBFS, пик {self.peak_dbfs:.1f} dBFS",
            f"  клиппинг       : {self.clipping_ratio * 100:.3f} % сэмплов",
            f"  шум            : SNR ~{self.snr_db:.0f} дБ, тишины {self.silence_ratio * 100:.0f} %",
            f"  основной тон   : {self.median_f0:.0f} Гц ({self.register})",
            f"  дикторы Silero : {', '.join(self.suggested_voices())}",
            "",
        ]
        lines += ["  " + finding.render() for finding in self.findings]
        return "\n".join(lines)


def _duration_finding(speech: float, purpose: str) -> Finding:
    comfortable, workable = DURATION_TARGETS.get(purpose, DURATION_TARGETS["vc"])
    if speech >= comfortable:
        return Finding(OK, f"речи {speech / 60:.1f} мин — достаточно")
    if speech >= workable:
        return Finding(
            WARN,
            f"речи {speech / 60:.1f} мин: обучится, но тембр будет узнаваем хуже; "
            f"цель — {comfortable / 60:.0f} мин",
        )
    return Finding(
        FAIL,
        f"речи всего {speech / 60:.1f} мин, нужно хотя бы {workable / 60:.0f} мин "
        f"(комфортно — {comfortable / 60:.0f})",
    )


def assess(source: str | Path, *, purpose: str = "vc", sample_rate: int = 44_100) -> RecordingReport:
    """Measure a recording and grade it for `purpose` (``"vc"`` or ``"xtts"``)."""
    path = Path(source)
    info = audio_utils.probe_info(path)
    samples, sr = audio_utils.load_audio(path, sample_rate, mono=True)
    stats = audio_utils.analyze(samples, sr)

    segments = audio_utils.detect_speech_segments(samples, sr)
    speech = audio_utils.total_duration(segments) or stats.duration * (1.0 - stats.silence_ratio)
    voiced = np.concatenate(
        [audio_utils.slice_segment(samples, sr, s.start, s.end) for s in segments[:40]]
    ) if segments else samples
    median_f0 = audio_utils.estimate_f0(voiced, sr)

    findings = [_duration_finding(speech, purpose)]

    source_rate = int(info.get("sample_rate", 0) or 0)
    if source_rate and source_rate < 32_000:
        findings.append(Finding(
            FAIL, f"частота дискретизации {source_rate} Гц: верхние форманты потеряны, "
                  "тембр не восстановить — пишите 44 100 Гц или выше"))
    elif source_rate and source_rate < 44_100:
        findings.append(Finding(
            WARN, f"частота дискретизации {source_rate} Гц; желательно 44 100 Гц или выше"))
    elif source_rate:
        findings.append(Finding(OK, f"частота дискретизации {source_rate} Гц"))

    kbps = info.get("kbps")
    if kbps and str(info.get("format", "")).upper() in {"MP3", "OGG", "MPEG"} and kbps < 160:
        findings.append(Finding(
            WARN, f"сжатие ~{kbps} кбит/с: артефакты кодека попадут в модель, "
                  "лучше WAV или mp3 от 192 кбит/с"))

    if stats.clipping_ratio > 0.005:
        findings.append(Finding(
            FAIL, f"клиппинг на {stats.clipping_ratio * 100:.2f} % сэмплов — перепишите тише"))
    elif stats.clipping_ratio > 0.0001:
        findings.append(Finding(
            WARN, f"клиппинг на {stats.clipping_ratio * 100:.3f} % сэмплов; "
                  "держите пики около −6 dBFS"))
    else:
        findings.append(Finding(OK, "клиппинга нет"))

    if stats.estimated_snr_db < 20.0:
        findings.append(Finding(
            FAIL, f"SNR ~{stats.estimated_snr_db:.0f} дБ: шум будет клонирован вместе с голосом"))
    elif stats.estimated_snr_db < 30.0:
        findings.append(Finding(
            WARN, f"SNR ~{stats.estimated_snr_db:.0f} дБ; цель — 30 дБ и выше"))
    else:
        findings.append(Finding(OK, f"SNR ~{stats.estimated_snr_db:.0f} дБ"))

    if stats.peak_dbfs > -1.0:
        findings.append(Finding(WARN, f"пик {stats.peak_dbfs:.1f} dBFS — слишком горячо"))
    elif stats.rms_dbfs < -35.0:
        findings.append(Finding(
            WARN, f"очень тихо ({stats.rms_dbfs:.1f} dBFS RMS): подвиньте микрофон ближе, "
                  "а не поднимайте усиление"))

    if stats.silence_ratio > 0.6:
        findings.append(Finding(
            WARN, f"{stats.silence_ratio * 100:.0f} % файла — тишина; полезной речи мало"))

    if median_f0 <= 0:
        findings.append(Finding(
            WARN, "не удалось оценить основной тон — проверьте, что в файле речь"))

    if int(info.get("channels", 1) or 1) > 1:
        findings.append(Finding(
            OK, "стерео будет сведено в моно (убедитесь, что голос в обоих каналах один)"))

    return RecordingReport(
        path=path,
        purpose=purpose,
        duration=stats.duration,
        speech_duration=speech,
        sample_rate=sr,
        source=info,
        rms_dbfs=stats.rms_dbfs,
        peak_dbfs=stats.peak_dbfs,
        clipping_ratio=stats.clipping_ratio,
        silence_ratio=stats.silence_ratio,
        snr_db=stats.estimated_snr_db,
        median_f0=median_f0,
        findings=findings,
    )
