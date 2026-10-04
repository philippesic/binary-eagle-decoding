#!/usr/bin/env python3
"""Native EAGLE three-tap target-only TRAIN golden portability gate.

Requires newly captured bounded autoregressive golden prefixes. Existing
EAGLE speculative/tree corpora retain their separate ancestry and audits.
Uses the same native target-only producer, numeric rules, CUDA execution proof,
STOP handling and resource cleanup as the block portability checker.
"""

from check_block_capture_portability import main

if __name__ == "__main__":
    raise SystemExit(main(expected_family="eagle", golden_only=True))
