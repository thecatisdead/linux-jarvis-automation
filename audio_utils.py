import re

import numpy as np
from scipy.signal import resample_poly

import config


# ============================================================
# PERCENTAGE PARSER (for volume/brightness commands)
# ============================================================

ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}

TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def parse_percentage(text):
    text = text.lower().replace("-", " ")

    match = re.search(r"(\d{1,3})\s*(?:%|percent)", text)
    if match:
        return max(0, min(100, int(match.group(1))))

    if "one hundred" in text:
        return 100

    words = text.split()
    for i, word in enumerate(words):
        if word in TENS:
            value = TENS[word]
            if i + 1 < len(words) and words[i + 1] in ONES:
                value += ONES[words[i + 1]]
            return value
        if word in ONES:
            return ONES[word]

    return None


# ============================================================
# RESAMPLE AUDIO
# ============================================================

def resample_audio(audio):
    return resample_poly(
        audio,
        config.ASSISTANT_SAMPLE_RATE,
        config.MIC_SAMPLE_RATE,
    ).astype(np.float32)


# ============================================================
# RMS -> GLOW INTENSITY
# ============================================================

def calculate_intensity(rms):
    signal = rms - config.NOISE_FLOOR

    if signal <= 0:
        target = 0.0
    else:
        target = signal / (config.LOUD_RMS - config.NOISE_FLOOR)
        target = float(np.clip(target, 0.0, 1.0))
        target = target ** config.RESPONSE_POWER

    return float(np.clip(target, 0.0, 1.0))