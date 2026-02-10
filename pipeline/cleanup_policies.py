#!/usr/bin/env python3
"""
Cleanup helper: keeps only current policies in Azure by archiving expired ones.

This leverages PolicyDeduplicationEngine.cleanup_expired_policies(), which:
1) Loads all policy metadata from the active container (default: healthcare-policies)
2) Identifies current vs expired based on effective/end dates
3) Moves expired policies to the archive container
4) Rewrites the active container to only the current set
5) Saves an index with counts per payer

Prerequisites:
- AZURE_CONNECTION_STRING must be available (config.py or environment)
"""

from config import AZURE_CONNECTION_STRING
from .policy_deduplication_system import PolicyDeduplicationEngine


def main():
    engine = PolicyDeduplicationEngine(AZURE_CONNECTION_STRING)
    stats = engine.cleanup_expired_policies()
    print("\nCleanup stats:")
    for k, v in stats.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
