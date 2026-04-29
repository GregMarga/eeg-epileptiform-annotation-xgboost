import mne

# Φόρτωση του EDF αρχείου
raw = mne.io.read_raw_edf(
    "../../../data/extra_data/P70_GHB_M1679_0000078.edf",
    preload=False,
    verbose="ERROR"
)

# Παίρνουμε τα annotations
annotations = raw.annotations

# Διατρέχουμε τα annotations και φιλτράρουμε PD_START / PD_STOP
for onset, duration, description in zip(
    annotations.onset,
    annotations.duration,
    annotations.description
):
    if description in ["PD_START", "PD_STOP"]:
        print(f"{description}: onset={onset:.3f}s, duration={duration:.3f}s")