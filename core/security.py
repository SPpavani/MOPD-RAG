"""Protection against indirect prompt injection through retrieved documents and teacher output.

In RAG, the documents are written by someone else yet pasted into the prompt. A document that says
"ignore previous instructions and ..." is an attack on YOUR system. Three deterministic layers:

  1. scan_injection()   flags instruction-like text aimed at an AI
  2. neutralize_*()     stops text from closing our <context> wrapper or faking a teacher header
  3. the prompts       declare retrieved text and teacher answers to be data, not instructions

Flagged chunks are quarantined (not sent to the model) and reported. This is a tripwire, not a guarantee:
a determined attacker can phrase an injection that is not listed. Keep tool permissions minimal and treat
model output as untrusted too.
"""
import re
from typing import List

_PATTERNS = [
    ("override_instructions", re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|any|your)\b"
        r".{0,30}\b(instructions?|prompts?|rules?|guidelines?)\b", re.I)),
    ("addresses_the_model", re.compile(
        r"\b(system|developer)\s+(prompt|message|instruction)s?\b|\byou are (now )?(an? )?(ai|llm|assistant|language model)\b"
        r"|\bnew instructions?\s*:", re.I)),
    ("suppress_or_force", re.compile(
        r"\b(do not|don't|never)\s+(mention|reveal|tell|cite|say)\b.{0,40}\b(user|source|document|instruction)s?\b"
        r"|\balways (answer|respond|reply|say|output)\b", re.I)),
    ("exfiltration", re.compile(
        r"\b(send|post|upload|forward|email)\b.{0,60}\b(to|at)\b.{0,40}(https?://|\bwww\.|@)", re.I)),
    ("prompt_tag_breakout", re.compile(r"<\s*/?\s*(context|document|system|instructions?|assistant|user)\s*>", re.I)),
]
_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]")
_CONTEXT_TAG = re.compile(r"<\s*/?\s*context\s*>", re.I)
_TEACHER_HEADER = re.compile(r"={3,}")


def scan_injection(text: str) -> List[str]:
    """Names of the injection signals found in the text (empty list = nothing suspicious)."""
    found = [name for name, pattern in _PATTERNS if pattern.search(text)]
    if _INVISIBLE.search(text):
        found.append("invisible_characters")
    return found


def neutralize_context(text: str) -> str:
    """A document must not be able to close or reopen our <context> wrapper."""
    return _CONTEXT_TAG.sub("[context tag removed]", _INVISIBLE.sub("", text))


def neutralize_teacher_output(text: str) -> str:
    """A teacher's answer must not be able to fake the '=== NAME TEACHER ===' headers the distiller relies on."""
    return _TEACHER_HEADER.sub("= = =", text)
