import time

from google import genai
from google.genai import types

import config


gemini_client = genai.Client(
    api_key=config.GEMINI_API_KEY
)


SYSTEM_INSTRUCTION = (
    "You are a voice assistant. Give short, natural, "
    "spoken-style answers — 1 to 3 sentences max. "
    "No markdown, no lists, no formatting. "
    "Just plain conversational text suitable for text-to-speech."
)


# ============================================================
# MODEL STATE
# ============================================================

# Models that recently failed because of quota/rate limits.
#
# Example:
# {
#     "gemini-3.7-flash": 1755631234.52
# }
#
# Means:
# "Don't try this model again until this timestamp."
model_cooldowns = {}


# How long we wait before checking a model again.
# This is NOT the Gemini quota reset time.
# It's simply our own retry/cooldown period.
MODEL_COOLDOWN_SECONDS = 60


# The model that successfully answered the previous request.
# This lets Jarvis stick with a working model instead of
# starting from the beginning of the fallback list every time.
current_model = None


# ============================================================
# CHECK IF MODEL IS AVAILABLE
# ============================================================

def is_model_available(model_name):
    cooldown_until = model_cooldowns.get(model_name)

    if cooldown_until is None:
        return True

    if time.time() >= cooldown_until:
        # Cooldown expired.
        # Remove it so we can try the model again.
        del model_cooldowns[model_name]

        print(f"[GEMINI] Cooldown expired: {model_name}")

        return True

    return False


# ============================================================
# MARK MODEL AS UNAVAILABLE
# ============================================================

def cooldown_model(model_name):
    cooldown_until = time.time() + MODEL_COOLDOWN_SECONDS

    model_cooldowns[model_name] = cooldown_until

    print(
        f"[GEMINI] {model_name} unavailable. "
        f"Retrying after {MODEL_COOLDOWN_SECONDS}s."
    )


# ============================================================
# ASK GEMINI
# ============================================================

def ask_gemini(text, memory_context=None):

    global current_model

    if memory_context:
        prompt = f"""
Relevant memories about the user:

{memory_context}

User:
{text}
"""
    else:
        prompt = text

    models = config.GEMINI_MODEL_FALLBACKS

    ordered_models = []

    if current_model is not None:
        ordered_models.append(current_model)

    for model in models:
        if model not in ordered_models:
            ordered_models.append(model)

    for model_name in ordered_models:

        if not is_model_available(model_name):
            print(
                f"[GEMINI] Skipping {model_name} "
                f"(cooldown active)"
            )
            continue

        try:

            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                ),
            )

            current_model = model_name

            print(
                f"[GEMINI] Used model: {model_name}"
            )

            return response.text.strip()

        except Exception as e:

            error_text = str(e)

            print(
                f"[GEMINI ERROR] "
                f"{model_name} failed: {error_text}"
            )

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):
                cooldown_model(model_name)
                continue

            continue

    return (
        "Sorry, all my AI models are unavailable right now."
    )