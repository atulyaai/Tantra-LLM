"""NP-DNA topic classifier — keyword-based category tagger.

Maps text to one of 10 NP-DNA categories using weighted keyword matching.
Each category has primary keywords (high weight) and secondary (low weight).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── 10 NP-DNA categories with full topic/sub-topic structure ──────────────

TOPICS: dict[str, list[str]] = {
    "conversation": [
        "general chat", "small talk", "greetings", "introductions",
        "farewells", "thanks", "apologies", "opinions", "recommendations",
        "casual discussion", "storytelling", "personal anecdotes",
        "social dialogue", "questions and answers", "everyday talk",
        "friendly advice", "chit-chat", "banter", "jokes", "humor",
    ],
    "code": [
        "programming languages", "python", "javascript", "typescript", "rust",
        "c++", "go", "java", "swift", "kotlin", "ruby", "php",
        "algorithms", "data structures", "software design", "architecture",
        "debugging", "testing", "web development", "frontend", "backend",
        "api design", "rest", "graphql", "databases", "sql", "nosql",
        "devops", "ci/cd", "docker", "kubernetes", "cloud",
        "version control", "git", "code review", "refactoring",
        "functional programming", "oop", "design patterns",
        "systems programming", "compilers", "interpreters",
        "machine learning code", "data science coding",
        "scripting", "automation", "cli tools", "package management",
        "concurrency", "async", "parallel computing",
        "security", "authentication", "authorization",
        "performance optimization", "profiling", "memory management",
        "mobile development", "ios", "android", "cross-platform",
        "game development", "unity", "unreal engine",
        "embedded systems", "firmware", "arduino", "raspberry pi",
        "blockchain", "smart contracts", "web3",
        "regex", "parsing", "serialization",
        "open source", "licensing", "contribution",
    ],
    "math": [
        "arithmetic", "algebra", "linear algebra", "geometry",
        "calculus", "differential equations", "statistics",
        "probability", "number theory", "topology", "trigonometry",
        "discrete mathematics", "combinatorics", "graph theory",
        "real analysis", "complex analysis", "functional analysis",
        "set theory", "logic", "proofs", "mathematical reasoning",
        "optimization", "numerical methods", "linear programming",
        "fourier analysis", "signal processing", "wavelets",
        "game theory", "decision theory", "information theory",
        "category theory", "abstract algebra", "group theory",
        "ring theory", "field theory", "galois theory",
        "measure theory", "integration", "lebesgue integral",
        "differential geometry", "manifolds", "tensor calculus",
        "stochastic processes", "markov chains", "monte carlo",
        "bayesian statistics", "hypothesis testing", "regression",
        "mathematical modeling", "simulation",
        "p-adic numbers", "analytic number theory", "modular forms",
        "lie algebras", "representation theory",
    ],
    "science": [
        "physics", "mechanics", "thermodynamics", "quantum mechanics",
        "relativity", "electromagnetism", "optics", "acoustics",
        "chemistry", "organic chemistry", "inorganic chemistry",
        "biochemistry", "physical chemistry", "analytical chemistry",
        "biology", "molecular biology", "cell biology", "genetics",
        "evolution", "ecology", "microbiology", "zoology", "botany",
        "astronomy", "astrophysics", "cosmology", "planetary science",
        "earth science", "geology", "oceanography", "meteorology",
        "neuroscience", "cognitive science", "psychology (science)",
        "materials science", "nanotechnology", "crystallography",
        "particle physics", "nuclear physics", "plasma physics",
        "fluid dynamics", "aerodynamics", "hydrodynamics",
        "climate science", "environmental science", "sustainability",
        "scientific method", "experiment design", "peer review",
        "laboratory techniques", "spectroscopy", "chromatography",
        "bioinformatics", "computational biology", "systems biology",
        "epidemiology", "immunology", "pharmacology",
        "paleontology", "archaeology (scientific)", "anthropology (scientific)",
    ],
    "writing": [
        "creative writing", "fiction", "non-fiction", "poetry",
        "essays", "journalism", "copywriting", "technical writing",
        "storytelling", "narrative structure", "plot development",
        "character development", "dialogue writing", "worldbuilding",
        "grammar and style", "voice and tone", "editing", "proofreading",
        "screenwriting", "playwriting", "scriptwriting",
        "blogging", "content writing", "seo writing",
        "academic writing", "research papers", "theses", "dissertations",
        "business writing", "reports", "proposals", "memos",
        "persuasive writing", "argumentative writing", "descriptive writing",
        "expository writing", "narrative writing",
        "writing prompts", "writing exercises", "writing workshops",
        "genre fiction", "science fiction", "fantasy", "mystery", "romance",
        "literary analysis", "criticism", "book reviews",
        "novel writing", "short stories", "flash fiction",
        "autobiography", "memoir", "biographical writing",
        "travel writing", "food writing", "nature writing",
    ],
    "language": [
        "linguistics", "phonetics", "phonology", "morphology",
        "syntax", "semantics", "pragmatics", "discourse analysis",
        "grammar", "vocabulary", "etymology", "word origins",
        "translation", "interpretation", "localization",
        "language learning", "second language acquisition",
        "multilingualism", "bilingualism",
        "comparative linguistics", "historical linguistics",
        "sociolinguistics", "dialectology", "language variation",
        "psycholinguistics", "neurolinguistics",
        "computational linguistics", "nlp", "corpus linguistics",
        "sign languages", "deaf studies", "manual communication",
        "writing systems", "orthography", "scripts", "alphabets",
        "endangered languages", "language documentation",
        "language policy", "language planning", "linguistic rights",
        "pragmatics", "speech acts", "implicature",
        "cognitive linguistics", "construction grammar",
        "language typology", "universal grammar",
        "applied linguistics", "tesol", "efl", "esl",
        "lexicography", "dictionary making", "terminology",
        "forensic linguistics", "linguistic profiling",
    ],
    "history": [
        "world history", "ancient civilizations", "medieval history",
        "modern history", "contemporary history",
        "european history", "asian history", "african history",
        "american history", "middle eastern history",
        "wars and conflicts", "world war i", "world war ii",
        "cold war", "civil wars", "revolutions", "battles",
        "biographies", "historical figures", "leaders",
        "archaeology", "historical artifacts", "ruins",
        "cultural history", "social history", "economic history",
        "political history", "diplomatic history", "military history",
        "colonial history", "imperialism", "decolonization",
        "ancient rome", "ancient greece", "ancient egypt",
        "china history", "india history", "japan history",
        "renaissance", "enlightenment", "industrial revolution",
        "age of exploration", "age of discovery", "colonization",
        "historical methodology", "historiography",
        "primary sources", "archival research", "oral history",
        "genealogy", "family history", "ancestry",
        "timelines", "historical periods", "eras",
        "historical maps", "cartography history",
        "religious history", "history of ideas", "intellectual history",
        "gender history", "history of feminism",
        "environmental history", "climate history",
    ],
    "society": [
        "economics", "macroeconomics", "microeconomics", "finance",
        "politics", "political science", "government", "policy",
        "philosophy", "ethics", "morality", "metaphysics",
        "epistemology", "aesthetics", "logic (philosophical)",
        "psychology", "clinical psychology", "social psychology",
        "developmental psychology", "personality psychology",
        "sociology", "social theory", "social structures",
        "anthropology", "cultural anthropology", "social anthropology",
        "law", "legal systems", "constitutional law", "criminal law",
        "education", "pedagogy", "learning theory", "curriculum",
        "religion", "theology", "comparative religion",
        "political economy", "public policy", "governance",
        "international relations", "geopolitics", "diplomacy",
        "social justice", "human rights", "civil rights",
        "urban studies", "demography", "population studies",
        "media studies", "communication theory", "journalism (society)",
        "cultural studies", "postcolonial theory", "critical theory",
        "environmental policy", "sustainability (social)",
        "business ethics", "corporate governance",
        "inequality", "poverty", "social stratification",
        "gender studies", "race and ethnicity", "identity politics",
        "social movements", "activism", "protest",
        "consumer behavior", "marketing", "advertising (social)",
        "organizational behavior", "management theory",
    ],
    "health": [
        "medicine", "clinical medicine", "diagnosis", "treatment",
        "anatomy", "physiology", "pathology",
        "nutrition", "dietetics", "food science", "dietary guidelines",
        "fitness", "exercise", "physical training", "sports medicine",
        "mental health", "psychiatry", "therapy", "counseling",
        "public health", "epidemiology", "health policy",
        "diseases", "disorders", "conditions", "syndromes",
        "pharmacology", "medications", "drugs", "prescriptions",
        "surgery", "surgical procedures", "operative medicine",
        "preventive medicine", "vaccination", "screening",
        "alternative medicine", "holistic health", "herbal medicine",
        "sleep health", "stress management", "wellness",
        "pediatrics", "child health", "development",
        "geriatrics", "aging", "elderly care",
        "cardiology", "heart health", "circulatory system",
        "neurology", "brain health", "nervous system",
        "oncology", "cancer", "tumors",
        "immunology", "immune system", "allergies", "autoimmune",
        "endocrinology", "hormones", "diabetes", "thyroid",
        "dermatology", "skin health", "hair", "nails",
        "orthopedics", "bones", "joints", "muscles",
        "ophthalmology", "vision", "eye health",
        "dentistry", "oral health", "dental care",
        "gynecology", "obstetrics", "women's health",
        "urology", "kidney health", "urinary system",
        "gastroenterology", "digestive health", "gut",
        "pulmonology", "respiratory health", "lungs",
        "emergency medicine", "first aid", "trauma care",
        "addiction", "substance abuse", "recovery",
    ],
    "art": [
        "music", "music theory", "composition", "performance",
        "visual art", "painting", "drawing", "sculpture", "printmaking",
        "design", "graphic design", "industrial design", "interior design",
        "film", "cinema", "movie making", "film theory", "cinematography",
        "photography", "photo editing", "camera techniques",
        "architecture", "architectural design", "building design",
        "fashion", "fashion design", "textiles", "clothing",
        "performing arts", "theatre", "dance", "opera", "ballet",
        "art history", "art criticism", "art theory", "aesthetics (art)",
        "digital art", "computer graphics", "3d modeling", "animation",
        "illustration", "cartooning", "comics", "manga",
        "ceramics", "pottery", "glass art", "jewelry making",
        "calligraphy", "typography", "lettering",
        "mixed media", "collage", "assemblage", "installation art",
        "conceptual art", "contemporary art", "modern art",
        "street art", "graffiti", "public art",
        "video games as art", "game design", "interactive media",
        "art education", "art techniques", "color theory",
        "composition (visual)", "perspective", "lighting",
        "art restoration", "conservation", "curation",
    ],
}


@dataclass
class TopicClassification:
    category: str
    sub_topic: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


class NpDnaTopicClassifier:
    """Keyword-based classifier for NP-DNA's 10 topic categories.

    Uses weighted keyword matching with primary and secondary keywords
    per category. Returns the best-match category and sub-topic.
    """

    def __init__(self):
        # Build keyword patterns for each category
        self._patterns: dict[str, list[tuple[str, float]]] = {}
        for category, sub_topics in TOPICS.items():
            patterns: list[tuple[str, float]] = []
            # Add category name itself as high-weight pattern
            patterns.append((rf"\b{category}\b", 5.0))
            for sub in sub_topics:
                # Primary keywords from the sub-topic name
                for word in sub.split():
                    if len(word) > 3:
                        patterns.append((rf"\b{re.escape(word)}\b", 2.0))
                # Multi-word phrases
                if " " in sub:
                    patterns.append((rf"{re.escape(sub)}", 4.0))
            self._patterns[category] = _dedupe_patterns(patterns)

        # Pre-compile all regex patterns for performance (compile once)
        self._compiled: dict[str, list[tuple[re.Pattern, float]]] = {}
        for cat, pats in self._patterns.items():
            self._compiled[cat] = [(re.compile(p, re.IGNORECASE), w) for p, w in pats]

        # Category-exclusive keywords (very high weight)
        self._exclusive: dict[str, list[tuple[str, float]]] = {
            "code": [
                (r"\bdef\s+\w+\s*\(", 8.0), (r"\bimport\s+\w+", 8.0),
                (r"\bclass\s+\w+", 8.0), (r"\bif\s+__name__\s*==\s*['\"]__main__['\"]", 10.0),
                (r"\bprint\(.*\)", 5.0), (r"\breturn\s+\w+", 4.0),
                (r"\bfor\s+\w+\s+in\s+range", 6.0), (r"\bwhile\s+True", 6.0),
                (r"\btry\s*:", 5.0), (r"\bexcept\s+\w+", 5.0),
                (r"\bconst\s+\w+\s*=", 5.0), (r"\bfunction\s+\w+\s*\(", 5.0),
                (r"\blet\s+\w+\s*=", 4.0), (r"\bvar\s+\w+\s*=", 4.0),
                (r"\bdef\s+main\b", 7.0), (r"\basync\s+def\b", 6.0),
                (r"\bfrom\s+\w+\s+import\b", 7.0), (r"\b#include\b", 8.0),
                (r"\bpublic\s+(static\s+)?void\b", 6.0),
                (r"\bint\s+main\s*\(", 6.0), (r"\bfn\s+\w+\s*\(", 6.0),
                (r"\b```\w*\n", 5.0),  # code block markers
            ],
            "math": [
                (r"\\frac\{", 6.0), (r"\\int", 6.0), (r"\\sum", 6.0),
                (r"\\lim", 5.0), (r"\\partial", 5.0), (r"\\sqrt", 5.0),
                (r"\be\^\{", 4.0), (r"\bπ\b", 6.0), (r"\bΣ\b", 5.0),
                (r"\b∫\b", 5.0), (r"\b∀\b", 4.0), (r"\b∃\b", 4.0),
                (r"\bx\^2\b", 4.0), (r"\bdx\b", 4.0),
                (r"\=\s*\{", 3.0),  # set notation
                (r"\\mathcal", 5.0), (r"\\mathbb", 5.0),
            ],
            "conversation": [
                (r"\b(hi|hello|hey)\b", 4.0), (r"\bhow are you\b", 6.0),
                (r"\b(thanks|thank you)\b", 4.0), (r"\byou're welcome\b", 4.0),
                (r"\bwhat's up\b", 5.0), (r"\bhow's it going\b", 5.0),
                (r"\bnice to meet\b", 5.0), (r"\bgood (morning|afternoon|evening)\b", 4.0),
                (r"\bhave a (great|nice|good) day\b", 5.0),
                (r"\bcan you (help|assist)\b", 4.0),
                (r"\bwhat do you think\b", 4.0),
                (r"\bI (think|feel|believe)\b", 3.0),
                (r"\bjust saying\b", 4.0), (r"\bby the way\b", 3.0),
            ],
            "science": [
                (r"F\s*=\s*ma", 7.0), (r"E\s*=\s*mc\^2", 8.0),
                (r"\bH₂O\b", 6.0), (r"\bCO₂\b", 5.0),
                (r"\bπr²\b", 5.0), (r"\bDNA\b", 4.0),
                (r"\bRNA\b", 4.0), (r"\bATP\b", 4.0),
                (r"\bpH\s*=\b", 5.0), (r"\bPV\s*=\s*nRT\b", 8.0),
                (r"\bλ\b", 4.0), (r"\bν\b", 4.0),
                (r"\bΔG\b", 5.0), (r"\bΔH\b", 5.0),
            ],
            "writing": [
                (r"\bonce upon a time\b", 6.0), (r"\bchapter \d+\b", 4.0),
                (r"\bthe end\b", 3.0), (r"\bdear reader\b", 4.0),
                (r"\bdear diary\b", 5.0), (r"\bin conclusion\b", 3.0),
                (r"\bfirst draft\b", 4.0), (r"\bwriter's block\b", 5.0),
                (r"\bnovel\b", 3.0), (r"\bshort story\b", 4.0),
            ],
            "health": [
                (r"\bmg/kg\b", 4.0), (r"\bbpm\b", 4.0), (r"\bmm Hg\b", 5.0),
                (r"\bBMI\b", 4.0), (r"\bECG\b", 4.0), (r"\bMRI\b", 4.0),
                (r"\bCT scan\b", 5.0), (r"\bblood pressure\b", 4.0),
                (r"\bheart rate\b", 3.0), (r"\btake\s+\d+\s*mg\b", 5.0),
            ],
            "art": [
                (r"\bCMYK\b", 5.0), (r"\bRGB\b", 4.0),
                (r"\bcolor palette\b", 4.0), (r"\bcomposition\b", 3.0),
                (r"\bfocal point\b", 4.0), (r"\bcontrast\b", 3.0),
                (r"\bperspective\b", 3.0), (r"\bbrush\s+stroke\b", 4.0),
                (r"\boil on canvas\b", 5.0), (r"\bwatercolor\b", 5.0),
                (r"\bacoustic\b", 4.0), (r"\bchord\s+progression\b", 5.0),
                (r"\btempo\b", 3.0), (r"\bcolor theory\b", 5.0),
            ],
            "language": [
                (r"\bverb\b", 4.0), (r"\bnoun\b", 4.0), (r"\badjective\b", 4.0),
                (r"\badverb\b", 4.0), (r"\bconjugation\b", 5.0),
                (r"\bdeclension\b", 5.0), (r"\bsyntax\b", 4.0),
                (r"\bmorphology\b", 5.0), (r"\bphoneme\b", 5.0),
                (r"\bIPA\b", 4.0), (r"\bglottal stop\b", 6.0),
                (r"\bgrammatical (gender|case)\b", 5.0),
                (r"\bsite:lang\b", 3.0),
            ],
            "history": [
                (r"\b\d{3,4}\s*(BC|BCE|AD|CE)\b", 5.0),
                (r"\b(B.C.|A.D.)\b", 5.0),
                (r"\bin the \d+th century\b", 5.0),
                (r"\bcentury\b", 3.0), (r"\bdynasty\b", 5.0),
                (r"\bempire\b", 4.0), (r"\bkingdom\b", 3.0),
                (r"\btreaty of\b", 5.0), (r"\bbattle of\b", 5.0),
                (r"\binvasion\b", 4.0), (r"\bwar\b", 3.0),
            ],
            "society": [
                (r"\bGDP\b", 4.0), (r"\binflation\b", 4.0),
                (r"\binvisible hand\b", 6.0), (r"\bsupply and demand\b", 5.0),
                (r"\bcategorical imperative\b", 7.0),
                (r"\bsocial contract\b", 6.0),
                (r"\bcorrelation does not imply\b", 5.0),
                (r"\bnull hypothesis\b", 5.0), (r"\bp-value\b", 5.0),
                (r"\bconstitutional\b", 4.0), (r"\bamendment\b", 4.0),
                (r"\bplaintiff\b", 4.0), (r"\bdefendant\b", 4.0),
            ],
        }

        # Pre-compile exclusive patterns too
        self._exclusive_compiled: dict[str, list[tuple[re.Pattern, float]]] = {}
        for cat, pats in self._exclusive.items():
            self._exclusive_compiled[cat] = [(re.compile(p, re.IGNORECASE), w) for p, w in pats]

    def classify(self, text: str) -> TopicClassification:
        """Classify text into NP-DNA topic category.

        Returns TopicClassification with category, sub-topic, and confidence.
        """
        text_lower = text.lower()[:50000]  # cap text length for performance
        scores: dict[str, float] = {}

        # Score each category using pre-compiled patterns
        for category, patterns in self._compiled.items():
            score = 0.0
            for pattern, weight in patterns:
                if pattern.search(text_lower):
                    score += weight
            scores[category] = score

        # Add exclusive keyword bonuses
        for category, patterns in self._exclusive_compiled.items():
            bonus = 0.0
            for pattern, weight in patterns:
                if pattern.search(text_lower):
                    bonus += weight
            if bonus > 0:
                scores[category] = scores.get(category, 0) + bonus

        # Find best category — fallback to conversation if nothing matches
        if not scores:
            return TopicClassification(
                category="conversation",
                sub_topic="general chat",
                confidence=0.5,
                scores={"conversation": 1.0},
            )
        best_cat = max(scores, key=scores.get)
        total = sum(scores.values()) or 1.0
        top_score = max(scores.values())
        if top_score > 0:
            confidence = scores[best_cat] / min(total, top_score * 3)
        else:
            confidence = 0.5

        # Find best sub-topic within the best category
        best_sub = self._find_best_sub_topic(text_lower, best_cat)

        return TopicClassification(
            category=best_cat,
            sub_topic=best_sub,
            confidence=confidence,
            scores=scores,
        )

    def classify_batch(self, texts: list[str]) -> list[TopicClassification]:
        """Classify a batch of texts."""
        return [self.classify(t) for t in texts]

    def _find_best_sub_topic(self, text_lower: str, category: str) -> str:
        """Find best-matching sub-topic within a category."""
        if category not in TOPICS:
            return "general"
        best_sub = "general"
        best_score = 0
        for sub in TOPICS[category]:
            score = 0
            for word in sub.split():
                if len(word) > 3 and word in text_lower:
                    score += 1
            if " " in sub and sub in text_lower:
                score += 3
            if score > best_score:
                best_score = score
                best_sub = sub
        return best_sub


def _dedupe_patterns(
    patterns: list[tuple[str, float]]
) -> list[tuple[str, float]]:
    """Remove duplicate patterns, keeping the highest weight."""
    seen: dict[str, float] = {}
    for pat, weight in patterns:
        if pat not in seen or weight > seen[pat]:
            seen[pat] = weight
    return list(seen.items())


# Convenience instance
classifier = NpDnaTopicClassifier()


def tag_text(text: str) -> TopicClassification:
    """Quick-tag a text with topic category."""
    return classifier.classify(text)


# Backward-compat alias for scripts importing `CATEGORIES`
CATEGORIES: list[str] = list(TOPICS.keys())
