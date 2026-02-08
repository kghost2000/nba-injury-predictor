"""Claude API client for injury signal extraction."""

import json
import logging
import time

import anthropic

from utils.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are an NBA injury report analyst. Extract structured injury information from the following text.

Source context: {source_context}

Text to analyze:
\"\"\"
{text}
\"\"\"

Respond with ONLY a valid JSON object (no markdown, no explanation) with these fields:
- "player_name": string or null if no specific player mentioned
- "team": string or null if unknown
- "injury_type": string describing the injury (e.g. "ankle sprain", "knee soreness") or null
- "severity": one of "minor", "moderate", "severe", "unknown"
- "estimated_timeline": string (e.g. "day-to-day", "2-3 weeks", "season-ending") or null
- "is_injury_related": boolean — true if the text is about an NBA player injury, false otherwise
- "confidence": float between 0.0 and 1.0 indicating how confident you are in the extraction
"""


class ClaudeClient:
    """Wrapper around the Anthropic SDK for injury signal extraction."""

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = model or CLAUDE_MODEL
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def extract_injury_info(self, text, source_context=""):
        """Extract structured injury info from raw text using Claude.

        Returns a dict with extracted fields, or a default dict on failure.
        """
        prompt = EXTRACTION_PROMPT.format(
            text=text,
            source_context=source_context,
        )

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text.strip()
            return self._parse_response(response_text)
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            return self._default_response()
        except Exception as e:
            logger.error(f"Unexpected error calling Claude: {e}")
            return self._default_response()

    def extract_batch(self, texts_with_context):
        """Process multiple texts sequentially.

        Args:
            texts_with_context: list of (text, source_context) tuples

        Returns:
            list of extracted info dicts
        """
        results = []
        for i, (text, context) in enumerate(texts_with_context):
            result = self.extract_injury_info(text, context)
            results.append(result)
            # Rate-limit between calls
            if i < len(texts_with_context) - 1:
                time.sleep(0.5)
        return results

    def _parse_response(self, response_text):
        """Parse Claude's JSON response, handling malformed output."""
        cleaned = response_text
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Malformed JSON from Claude: {response_text[:200]}")
            return self._default_response()

        return {
            "player_name": data.get("player_name"),
            "team": data.get("team"),
            "injury_type": data.get("injury_type"),
            "severity": data.get("severity", "unknown"),
            "estimated_timeline": data.get("estimated_timeline"),
            "is_injury_related": data.get("is_injury_related", False),
            "confidence": float(data.get("confidence", 0.0)),
        }

    @staticmethod
    def _default_response():
        """Return a safe default when extraction fails."""
        return {
            "player_name": None,
            "team": None,
            "injury_type": None,
            "severity": "unknown",
            "estimated_timeline": None,
            "is_injury_related": False,
            "confidence": 0.0,
        }
