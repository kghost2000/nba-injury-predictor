"""Injury signal parser powered by Claude.

Phase 4 Implementation:
- Takes raw text from unstructured sources (tweets, Reddit posts, articles)
- Uses Claude to extract: player name, team, injury type, severity, timeline
- Assigns confidence scores based on source reliability and language certainty
- Deduplicates signals across sources
- Stores parsed results in the injury_signals table
"""
