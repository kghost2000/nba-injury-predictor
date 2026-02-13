"""Parse semi-structured injury descriptions into model-ready features.

The NBA injury reports use a consistent format:
    "{Side} {BodyPart}; {InjuryType}; {Modifier}"

This module extracts structured categorical features from that text.
"""

import re

# ---------------------------------------------------------------------------
# Body-part keyword → canonical name mapping
# ---------------------------------------------------------------------------
# Order matters: longer/more-specific patterns must come first so that
# e.g. "Achilles Tendon" matches before "Achilles".

_BODY_PART_PATTERNS = [
    # Knee
    (r"knee|acl|mcl|lcl|pcl|meniscus|patellar|patella", "knee"),
    # Ankle / foot
    (r"ankle", "ankle"),
    (r"achilles\s*tendon|achilles", "achilles"),
    (r"foot|toe|plantar|turf toe|tib/fib|navicular", "foot"),
    (r"heel", "foot"),
    # Upper leg
    (r"hamstring", "hamstring"),
    (r"quadriceps|quad|thigh", "quadriceps"),
    (r"groin|adductor|hip flexor", "groin"),
    (r"hip|glute", "hip"),
    (r"calf|soleus", "calf"),
    # Lower leg
    (r"tibia|fibula|shin|lower leg", "lower_leg"),
    # Back / core
    (r"low back|lower back|back|lumbar|sacrum|sacral|spine", "back"),
    (r"core|abdom|oblique|rib|intercostal", "core"),
    # Shoulder / arm
    (r"shoulder|rotator|labrum", "shoulder"),
    (r"elbow|ucl", "elbow"),
    (r"wrist", "wrist"),
    (r"hand|finger|thumb|knuckle", "hand"),
    # Head / neck
    (r"head|concussion|facial|face|eye|jaw|nose|nasal|orbital", "head"),
    (r"neck|cervical|trapezius", "neck"),
    # Catch-all
    (r"chest|pectoral|sternum", "chest"),
]

# Map body_part → broader region
_BODY_REGION_MAP = {
    "knee": "lower_leg",
    "ankle": "lower_leg",
    "achilles": "lower_leg",
    "foot": "lower_leg",
    "calf": "lower_leg",
    "lower_leg": "lower_leg",
    "hamstring": "upper_leg",
    "quadriceps": "upper_leg",
    "groin": "upper_leg",
    "hip": "upper_leg",
    "back": "torso",
    "core": "torso",
    "chest": "torso",
    "shoulder": "arm",
    "elbow": "arm",
    "wrist": "arm",
    "hand": "arm",
    "head": "head_neck",
    "neck": "head_neck",
}

# ---------------------------------------------------------------------------
# Injury-type keyword → canonical name mapping
# ---------------------------------------------------------------------------

_INJURY_TYPE_PATTERNS = [
    # Structural damage (severe)
    (r"torn|tear|ligament tear|acl tear|mcl tear|rupture", "tear"),
    (r"fracture|stress fracture|broken", "fracture"),
    (r"surgery|arthroscopy|repair|surgical", "surgery"),
    (r"dislocation|dislocated|subluxation", "dislocation"),
    # Moderate
    (r"sprain|sprained|hyperextension", "sprain"),
    (r"strain|strained", "strain"),
    (r"contusion|bruise|bone bruise", "contusion"),
    (r"concussion", "concussion"),
    # Mild / chronic
    (r"soreness|sore|discomfort|irritation", "soreness"),
    (r"tightness|spasm|cramp", "tightness"),
    (r"tendinitis|tendinosis|tendinopathy", "tendinopathy"),
    (r"fasciitis|plantar fasciitis", "tendinopathy"),
    (r"inflammation|effusion|swelling|bursitis", "inflammation"),
    (r"impingement", "impingement"),
    (r"stress reaction", "inflammation"),
    (r"laceration|cut", "laceration"),
    (r"pain", "pain"),
    # Recovery (type unknown but indicates prior injury)
    (r"injury\s*recover|injury\s*management|injury\s*maintenance", "recovery"),
    # Illness
    (r"illness|flu|covid|gastro|stomach|virus|infection|respiratory", "illness"),
]

# Severity tiers (higher = more severe)
_SEVERITY_MAP = {
    "soreness": 1,
    "tightness": 1,
    "pain": 1,
    "laceration": 1,
    "illness": 1,
    "contusion": 2,
    "inflammation": 2,
    "impingement": 2,
    "tendinopathy": 2,
    "sprain": 3,
    "strain": 3,
    "concussion": 3,
    "dislocation": 3,
    "fracture": 4,
    "tear": 4,
    "surgery": 4,
    "recovery": 3,
}

# ---------------------------------------------------------------------------
# Non-injury categories
# ---------------------------------------------------------------------------

_NON_INJURY_PATTERNS = [
    r"g[\s-]*league",
    r"two[\s-]*way",
    r"on assignment",
    r"rest\b",
    r"personal\s*reasons",
    r"not with team",
    r"return to competition",
    r"reconditioning",
    r"load management",
    r"health and safety protocols",
    r"league suspension",
    r"trade pending",
    r"ineligible",
    r"coach.?s decision",
    r"^-$",
    r"^n/?a$",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_injury_description(desc):
    """Parse an injury description string into structured features.

    Args:
        desc: Raw injury description, e.g. "Left Knee; ACL Tear; Surgery"

    Returns:
        dict with keys:
            side           - "left", "right", "bilateral", or None
            body_part      - canonical body part (e.g. "knee", "ankle")
            body_region    - broader region (e.g. "lower_leg", "upper_leg")
            injury_type    - canonical injury type (e.g. "sprain", "tear")
            severity       - int 1-4 (1=mild, 4=severe) or None
            is_surgical    - bool
            is_recovery    - bool (injury management / recovery phase)
            is_non_injury  - bool (G League, rest, personal, etc.)
    """
    if not desc or not desc.strip():
        return _empty_result()

    text = desc.strip()
    lower = text.lower()

    # Check non-injury first
    is_non_injury = any(re.search(p, lower) for p in _NON_INJURY_PATTERNS)
    if is_non_injury:
        return {
            "side": None,
            "body_part": None,
            "body_region": None,
            "injury_type": None,
            "severity": None,
            "is_surgical": False,
            "is_recovery": False,
            "is_non_injury": True,
        }

    # Extract side
    side = None
    if re.search(r"\bleft\b", lower):
        side = "left"
    elif re.search(r"\bright\b", lower):
        side = "right"
    elif re.search(r"\bbilateral\b", lower):
        side = "bilateral"

    # Extract body part
    body_part = None
    for pattern, name in _BODY_PART_PATTERNS:
        if re.search(pattern, lower):
            body_part = name
            break

    body_region = _BODY_REGION_MAP.get(body_part)

    # Extract injury type
    injury_type = None
    for pattern, name in _INJURY_TYPE_PATTERNS:
        if re.search(pattern, lower):
            injury_type = name
            break

    severity = _SEVERITY_MAP.get(injury_type)

    # Flags
    is_surgical = bool(re.search(
        r"surgery|arthroscopy|repair|surgical|procedure", lower
    ))
    is_recovery = bool(re.search(
        r"injury\s*(?:recovery|management|maintenance)|reconditioning|recovery",
        lower
    ))

    return {
        "side": side,
        "body_part": body_part,
        "body_region": body_region,
        "injury_type": injury_type,
        "severity": severity,
        "is_surgical": is_surgical,
        "is_recovery": is_recovery,
        "is_non_injury": False,
    }


def featurize_descriptions(descriptions):
    """Parse a list of descriptions into a list of feature dicts.

    Useful for batch processing before converting to a DataFrame:
        >>> import pandas as pd
        >>> features_df = pd.DataFrame(featurize_descriptions(descriptions))
    """
    return [parse_injury_description(d) for d in descriptions]


def _empty_result():
    return {
        "side": None,
        "body_part": None,
        "body_region": None,
        "injury_type": None,
        "severity": None,
        "is_surgical": False,
        "is_recovery": False,
        "is_non_injury": False,
    }
