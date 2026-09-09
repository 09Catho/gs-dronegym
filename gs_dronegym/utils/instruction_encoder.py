"""Deterministic fixed-width encoding of language instructions.

The encoder lives outside the baselines package so that the environment can
expose the same features it trains on without importing the learning stack, and
so that one versioned definition is shared by both.

Bucketing uses BLAKE2b. Python's built-in ``hash`` is salted per interpreter and
must never be used here: it makes features differ between processes, which
silently decouples training from inference.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")

#: Version of the instruction feature encoder written into checkpoints.
#:
#: Version 1 is the first deterministic encoder. Checkpoints without a recorded
#: version were produced by the pre-1 encoder, which used Python's built-in
#: ``hash`` and is therefore not reproducible across processes.
INSTRUCTION_ENCODER_VERSION = 1

#: Sentinel recorded for checkpoints that predate encoder versioning.
LEGACY_INSTRUCTION_ENCODER_VERSION = 0

#: Default feature width used by the baseline policy and the environment.
DEFAULT_INSTRUCTION_DIM = 128


def stable_token_bucket(token: str, dimension: int) -> int:
    """Map a token to a feature bucket deterministically across processes.

    Args:
        token: Lowercase instruction token.
        dimension: Number of feature buckets.

    Returns:
        Bucket index in ``[0, dimension)``.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def encode_instruction(text: str, dimension: int = DEFAULT_INSTRUCTION_DIM) -> np.ndarray:
    """Convert free-form text into a fixed-size hashed bag-of-words vector.

    Args:
        text: Instruction string.
        dimension: Output vector dimension.

    Returns:
        Dense float32 feature vector normalized by token count.
    """
    features = np.zeros(dimension, dtype=np.float32)
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return features
    for token in tokens:
        features[stable_token_bucket(token, dimension)] += 1.0
    features /= np.float32(max(len(tokens), 1))
    return features
