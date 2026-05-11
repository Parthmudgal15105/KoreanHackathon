"""Core resume matching logic built with only Python standard libraries."""

from typing import Dict, Iterable, List, Sequence, Tuple
import math
import re

from data import SKILL_ALIASES


_BOUNDARY_CHARS = r"A-Za-z0-9_+/#.-"


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _alias_sort_key(alias: str) -> Tuple[int, int, str]:
    # Prefer phrases before individual tokens, then longer aliases before shorter ones.
    return (-len(alias.split()), -len(alias), alias)


def normalize_skills(raw_skill_string: str) -> List[str]:
    """Convert noisy skill text into canonical skill names.

    Unknown tokens are ignored. Aliases with spaces are attempted before
    single-token aliases so phrases such as "rest api" win over "rest".
    """

    if not raw_skill_string:
        return []

    aliases = sorted(SKILL_ALIASES, key=_alias_sort_key)
    normalized: List[str] = []

    for raw_part in raw_skill_string.split(","):
        part = _clean_text(raw_part)
        if not part:
            continue

        if part in SKILL_ALIASES:
            normalized.append(SKILL_ALIASES[part])
            continue

        masked = part
        for alias in aliases:
            pattern = re.compile(
                rf"(?<![{_BOUNDARY_CHARS}]){re.escape(alias)}(?![{_BOUNDARY_CHARS}])"
            )
            match = pattern.search(masked)
            if match:
                normalized.append(SKILL_ALIASES[alias])
                start, end = match.span()
                masked = masked[:start] + (" " * (end - start)) + masked[end:]

    return normalized


def deduplicate_skills(skills: Iterable[str]) -> List[str]:
    """Remove duplicate skills while preserving first-seen order."""

    seen = set()
    unique_skills = []
    for skill in skills:
        if skill not in seen:
            seen.add(skill)
            unique_skills.append(skill)
    return unique_skills


def build_vocabulary(normalized_resumes: Sequence[Sequence[str]]) -> List[str]:
    """Build an alphabetically sorted vocabulary from resume skills only."""

    vocabulary = set()
    for skills in normalized_resumes:
        vocabulary.update(skills)
    return sorted(vocabulary)


def compute_document_frequency(normalized_resumes: Sequence[Sequence[str]]) -> Dict[str, int]:
    """Count how many resumes contain each skill."""

    df: Dict[str, int] = {}
    for skills in normalized_resumes:
        for skill in set(skills):
            df[skill] = df.get(skill, 0) + 1
    return df


def compute_idf(df: Dict[str, int], total_docs: int) -> Dict[str, float]:
    """Compute unsmoothed IDF using the natural logarithm."""

    if total_docs <= 0:
        return {}
    return {skill: math.log(total_docs / doc_count) for skill, doc_count in df.items()}


def compute_tf(skills: Sequence[str]) -> Dict[str, float]:
    """Compute TF for each skill in a resume: TF = 1 / N after deduplication."""

    n = len(skills)
    if n == 0:
        return {}
    return {skill: 1.0 / n for skill in skills}


def build_resume_tfidf_vector(
    skills: Sequence[str],
    vocabulary: Sequence[str],
    idf: Dict[str, float],
) -> List[float]:
    """Build a TF-IDF vector for one resume using TF = 1/N."""

    tf = compute_tf(skills)
    return [tf.get(skill, 0.0) * idf.get(skill, 0.0) for skill in vocabulary]


def build_jd_binary_vector(jd_skills: Sequence[str], vocabulary: Sequence[str]) -> List[float]:
    """Build a binary JD vector over the resume vocabulary."""

    jd_skill_set = set(jd_skills)
    return [1.0 if skill in jd_skill_set else 0.0 for skill in vocabulary]


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Return cosine similarity, or 0.0 when either vector is empty/zero."""

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _prepare_resume_skills(resumes: Sequence[Dict[str, str]]) -> List[List[str]]:
    return [
        deduplicate_skills(normalize_skills(resume["skills"]))
        for resume in resumes
    ]


def _prepare_job_skills(job: Dict[str, str]) -> List[str]:
    raw_skills = f"{job.get('required', '')}, {job.get('preferred', '')}"
    return deduplicate_skills(normalize_skills(raw_skills))


def rank_candidates(
    resumes: Sequence[Dict[str, str]],
    jobs: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    """Rank candidates for each job and return the top 3 matches."""

    normalized_resumes = _prepare_resume_skills(resumes)
    vocabulary = build_vocabulary(normalized_resumes)
    df = compute_document_frequency(normalized_resumes)
    idf = compute_idf(df, len(resumes))

    resume_vectors = [
        build_resume_tfidf_vector(skills, vocabulary, idf)
        for skills in normalized_resumes
    ]

    ranked_jobs = []
    for job in jobs:
        jd_skills = _prepare_job_skills(job)
        jd_vector = build_jd_binary_vector(jd_skills, vocabulary)

        scored_candidates = []
        for resume, resume_vector in zip(resumes, resume_vectors):
            score = cosine_similarity(resume_vector, jd_vector)
            scored_candidates.append((resume["name"], score))

        scored_candidates.sort(key=lambda item: (-item[1], item[0]))
        ranked_jobs.append(
            {
                "job": job,
                "matches": scored_candidates[:3],
            }
        )

    return ranked_jobs
