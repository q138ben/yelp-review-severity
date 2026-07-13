"""Aspect decomposition: route a review's clauses to aspects, then score each aspect with a caller-supplied
scorer.

The whole point is that the SAME routing and the SAME scorer are used everywhere. The scorer is the deployed
ordinal encoder: `sentences_by_aspect` splits a review into clauses and attributes each to the aspect(s) it
mentions, then the encoder scores each aspect's clauses — so the per-aspect breakdown is a faithful
decomposition of the model's own overall score, not a separate signal. Both the serving endpoint and the
aspect-model notebook call `aspect_scores(text, score_fn)` with that encoder scorer. Restaurant-scoped
taxonomy (per EDA: 73% of reviews are food-service).
"""
import re

ASPECTS = {
    "food": ["food", "dish", "dishes", "meal", "taste", "tasty", "flavor", "flavour", "delicious",
             "menu", "portion", "portions", "cooked", "fresh", "stale", "bland", "seasoned",
             "dessert", "appetizer", "entree", "cuisine", "yummy", "undercooked", "overcooked"],
    "service": ["service", "staff", "server", "servers", "waiter", "waitress", "employee", "employees",
                "host", "hostess", "manager", "rude", "friendly", "attentive", "polite", "helpful",
                "professional", "greeted", "unfriendly"],
    "wait": ["wait", "waited", "waiting", "slow", "quick", "fast", "prompt", "long", "line", "queue",
             "minutes", "hour", "hours", "forever", "delay", "delayed"],
    "price": ["price", "prices", "priced", "expensive", "cheap", "value", "worth", "overpriced",
              "affordable", "cost", "pricey", "deal", "bucks", "dollars"],
    "cleanliness": ["clean", "dirty", "filthy", "hygiene", "messy", "spotless", "sticky", "gross",
                    "sanitary", "unsanitary"],
    "ambiance": ["ambiance", "ambience", "atmosphere", "decor", "music", "cozy", "noisy", "loud",
                 "vibe", "seating", "comfortable", "romantic", "lighting", "crowded"],
    "drinks": ["drink", "drinks", "cocktail", "cocktails", "beer", "wine", "coffee", "bar", "latte",
               "margarita", "beverage"],
    "amenities": ["parking", "park", "wifi", "wi-fi", "bathroom", "restroom", "location", "patio"],
}
ASPECT_NAMES = list(ASPECTS)
_PAT = {a: re.compile(r"\b(" + "|".join(map(re.escape, kws)) + r")\b") for a, kws in ASPECTS.items()}
# clause splitter: sentence punctuation + commas + contrastive/coordinating conjunctions, so co-mentioned
# aspects ("great food BUT slow service and a long wait") land in separate clauses the encoder can score apart
_CLAUSE = re.compile(r"[.!?;,\n]+|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\band\b")


def sentences_by_aspect(text):
    """Route each CLAUSE of `text` to the aspect(s) it mentions, by keyword match. Returns {aspect: [clauses]}.
    Lower-cased because the encoder is uncased, so these are exactly the tokens it sees."""
    groups = {}
    for clause in (c.strip() for c in _CLAUSE.split(text.lower()) if c and c.strip()):
        for a, pat in _PAT.items():
            if pat.search(clause):
                groups.setdefault(a, []).append(clause)
    return groups


def aspect_scores(text, score_fn):
    """Attribute `text` to aspects (clause routing) and score each aspect's clauses with `score_fn`.

    `score_fn` maps a list of texts -> a sequence of scores; pass the deployed encoder's scorer so the
    per-aspect breakdown is the model's own read of each aspect's clauses. Returns {aspect: score} for the
    aspects mentioned (empty dict if none). This is the single routing+scoring path used by both the serving
    endpoint and the aspect-model notebook — same routing, same scorer.
    """
    groups = sentences_by_aspect(text)
    if not groups:
        return {}
    aspects = list(groups)
    scores = score_fn([" ".join(groups[a]) for a in aspects])
    return {a: float(s) for a, s in zip(aspects, scores)}
