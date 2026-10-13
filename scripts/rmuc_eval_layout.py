"""Reproducible command-to-environment assignment without touching simulator RNG."""
import random


def bucket_assignment(bucket_count: int, replicas: int, seed: int | None = None) -> list[int]:
    if bucket_count < 1 or replicas < 1:
        raise ValueError('Bucket count and replica count must be positive')
    ids = [bucket for bucket in range(bucket_count) for _ in range(replicas)]
    if seed is not None:
        random.Random(seed).shuffle(ids)
    return ids
