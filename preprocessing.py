import re
import ast
import mne
import numpy as np
from pathlib import Path
from mne.channels import make_standard_montage


# -------------------------------------------------
# PyPREP report parser
# -------------------------------------------------

# Categories that lead to interpolation. If a channel appears
# ONLY in hf_noise, we leave it untouched.
INTERPOLATE_CATEGORIES = ("deviation", "correlation", "ransac")


def _parse_channel_list(value: str) -> list[str]:
    """
    Accepts the raw value after the ':' character, e.g.
        "['Fp2', 'O1']"
        "[np.str_('Fp1')]"
        "[np.str_('F8')]"
    Returns a clean list of strings.
    """
    cleaned = re.sub(r"np\.str_\(([^)]+)\)", r"\1", value.strip())
    try:
        return [str(c) for c in ast.literal_eval(cleaned)]
    except (ValueError, SyntaxError):
        return []


def parse_channel_detection_report(txt_path: Path) -> dict[str, dict[str, list[str]]]:
    """
    Parse the report file and return:
        {
            "P20_GHB_00015_0000348": {
                "deviation":   [...],
                "hf_noise":    ["F8"],
                "correlation": ["T6", "P4", "Pz"],
                "ransac":      [],
            },
            ...
        }
    """
    text = Path(txt_path).read_text(encoding="utf-8", errors="replace")

    # Each entry starts with a header line like "[i/N] FILENAME.edf"
    edf_header_re = re.compile(r"^\[\d+/\d+\]\s+(.+?)\.edf\s*$", re.MULTILINE)
    matches = list(edf_header_re.finditer(text))

    report: dict[str, dict[str, list[str]]] = {}

    for i, m in enumerate(matches):
        basename = m.group(1).strip()
        block_start = m.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[block_start:block_end]

        cats: dict[str, list[str]] = {
            "deviation":   [],
            "hf_noise":    [],
            "correlation": [],
            "ransac":      [],
        }

        # Match lines like "    deviation   : [..]"
        for cat in cats.keys():
            cat_re = re.compile(rf"^\s*{cat}\s*:\s*(\[.*?\])\s*$", re.MULTILINE)
            cm = cat_re.search(block)
            if cm:
                cats[cat] = _parse_channel_list(cm.group(1))

        report[basename] = cats

    return report


def channels_to_interpolate(report_entry: dict[str, list[str]]) -> list[str]:
    """
    Takes the {category -> [channels]} dict for one EDF and returns
    the list of channels to interpolate: those appearing in ANY of
    deviation / correlation / ransac. Channels that appear only in
    hf_noise are ignored.
    """
    to_interp: set[str] = set()
    for cat in INTERPOLATE_CATEGORIES:
        to_interp.update(report_entry.get(cat, []))
    return sorted(to_interp)


def mark_and_interpolate_bad_channels_from_report(
        raw: mne.io.BaseRaw,
        edf_basename: str,
        report: dict[str, dict[str, list[str]]],
        plot: bool = False,
) -> list[str]:
    """
    Pulls the bad channels for this specific EDF from the pre-computed
    PyPREP report and interpolates ONLY those that fall under
    deviation / correlation / ransac.
    """
    if edf_basename not in report:
        print(f"  [WARN] No report entry for {edf_basename} — skipping interpolation")
        return []

    entry = report[edf_basename]
    bad_channels = channels_to_interpolate(entry)

    # Defensive: keep only channels that actually exist in the raw object
    bad_channels = [ch for ch in bad_channels if ch in raw.ch_names]

    if not bad_channels:
        return []

    raw.info["bads"] = bad_channels
    raw.interpolate_bads(reset_bads=True)

    if plot:
        raw.plot(scalings="auto", block=True)

    return bad_channels


# -------------------------------------------------
# Helpers — unchanged logic
# -------------------------------------------------

def sec_to_hmsms(sec: float) -> str:
    ms_total = int(round(sec * 1000))
    s, ms = divmod(ms_total, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def create_annotation_centered_epochs(
        data,
        annotations,
        sfreq,
        positive_label="*",
        negative_label="-",
        window=250,  # in ms
):
    annotation_onsets_samples = np.round(annotations.onset * sfreq).astype(int)
    annotation_onsets_sec = np.asarray(annotations.onset, dtype=float)
    descriptions = np.array(annotations.description)

    labels = []
    epochs = []
    center_sec_list = []
    center_hmsms_list = []

    margin = int(sfreq * window / 1000)
    n_samples = data.shape[1]

    for i in range(len(annotations)):
        epoch_center_idx = int(annotation_onsets_samples[i])
        center_sec = float(annotation_onsets_sec[i])

        start = epoch_center_idx - margin
        stop = epoch_center_idx + margin

        if start < 0 or stop > n_samples:
            continue

        if descriptions[i] == positive_label:
            labels.append(1)
            epochs.append(data[:, start:stop])
            center_sec_list.append(center_sec)
            center_hmsms_list.append(sec_to_hmsms(center_sec))
        elif descriptions[i] == negative_label:
            labels.append(0)
            epochs.append(data[:, start:stop])
            center_sec_list.append(center_sec)
            center_hmsms_list.append(sec_to_hmsms(center_sec))

    return (
        np.array(epochs),
        np.array(labels),
        np.array(center_sec_list, dtype=np.float64),
        np.array(center_hmsms_list, dtype=object),
    )


# -------------------------------------------------
# Main preprocessing function
# -------------------------------------------------

def preprocess_edf_to_windows(
        edf_path: str,
        report: dict[str, dict[str, list[str]]],
        l_freq: float = 0.5,
        h_freq: float = 40.0,
        positive_label: str = "*",
        negative_label: str = "-",
):
    edf_path = Path(edf_path)
    edf_basename = edf_path.stem  # e.g. "P20_GHB_00015_0000348"

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw.rename_channels(lambda ch: ch.replace("EEG ", "").strip())

    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    raw.filter(
        l_freq=l_freq, h_freq=h_freq,
        method="fir", fir_design="firwin",
        phase="zero", verbose="ERROR",
    )

    raw.resample(80, verbose="ERROR")

    # Bad channels driven by the pre-computed report —
    # only deviation / correlation / ransac, hf_noise is skipped
    bad_chs = mark_and_interpolate_bad_channels_from_report(
        raw, edf_basename, report, plot=False,
    )

    # Average reference is applied AFTER interpolation so that
    # the bad channels do not contaminate the reference signal
    raw.set_eeg_reference("average", verbose="ERROR")

    data = raw.get_data()
    ch_names = np.array(raw.ch_names, dtype=object)
    sfreq = float(raw.info["sfreq"])
    annotations = raw.annotations

    epochs, labels, center_sec, center_hmsms = create_annotation_centered_epochs(
        data=data,
        sfreq=sfreq,
        annotations=annotations,
        positive_label=positive_label,
        negative_label=negative_label,
    )

    return (
        epochs.astype(np.float32),
        labels.astype(np.uint8),
        center_sec.astype(np.float64),
        center_hmsms.astype(object),
        sfreq,
        ch_names,
        np.array(bad_chs, dtype=object),
    )



if __name__ == "__main__":
    REPORT_PATH = Path("channel_detection_details.txt")
    report = parse_channel_detection_report(REPORT_PATH)

    print(f"Parsed {len(report)} EDF entries")
    for base, cats in report.items():
        to_interp = channels_to_interpolate(cats)
        skipped_hf = sorted(set(cats["hf_noise"]) - set(to_interp))
        print(f"  {base}")
        print(f"    interpolate: {to_interp}")
        if skipped_hf:
            print(f"    skipped (hf_noise only): {skipped_hf}")