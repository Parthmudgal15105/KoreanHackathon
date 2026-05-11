from typing import Dict, Iterable, List, Sequence, Tuple
import math
import re

from data import SKILL_ALIASES

_CHARS = r"A-Za-z0-9_+/#.-"

def _clean(val: str) -> str:
    return re.sub(r"\s+", " ", val.strip().lower())

def _sort_key(alias: str) -> Tuple[int, int, str]:
    return (-len(alias.split()), -len(alias), alias)

def normalize_skills(raw_skill_string: str) -> List[str]:
    """Split, clean, and map skills to canonical names."""
    if not raw_skill_string:
        return []

    aliases = sorted(SKILL_ALIASES, key=_sort_key)
    normalized: List[str] = []

    for raw_part in raw_skill_string.split(","):
        part = _clean(raw_part)
        if not part or part in SKILL_ALIASES:
            if part in SKILL_ALIASES:
                normalized.append(SKILL_ALIASES[part])
            continue

        masked = part
        for alias in aliases:
            pattern = re.compile(rf"(?<![{_CHARS}]){re.escape(alias)}(?![{_CHARS}])")
            match = pattern.search(masked)
            if match:
                normalized.append(SKILL_ALIASES[alias])
                start, end = match.span()
                masked = masked[:start] + (" " * (end - start)) + masked[end:]

    return normalized

def deduplicate_skills(skills: Iterable[str]) -> List[str]:
    seen = set()
    unique = []
    for s in skills:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique

def build_vocabulary(resumes: Sequence[Sequence[str]]) -> List[str]:
    vocab = set()
    for skills in resumes:
        vocab.update(skills)
    return sorted(vocab)

def compute_df(resumes: Sequence[Sequence[str]]) -> Dict[str, int]:
    df = {}
    for skills in resumes:
        for s in set(skills):
            df[s] = df.get(s, 0) + 1
    return df

def compute_idf(df: Dict[str, int], n: int) -> Dict[str, float]:
    return {s: math.log(n / count) for s, count in df.items()} if n > 0 else {}

def compute_tf(skills: Sequence[str]) -> Dict[str, float]:
    n = len(skills)
    return {s: 1.0 / n for s in skills} if n > 0 else {}

def build_resume_vec(skills: Sequence[str], vocab: Sequence[str], idf: Dict[str, float]) -> List[float]:
    tf = compute_tf(skills)
    return [tf.get(s, 0.0) * idf.get(s, 0.0) for s in vocab]

def build_jd_vec(jd_skills: Sequence[str], vocab: Sequence[str]) -> List[float]:
    jd_set = set(jd_skills)
    return [1.0 if s in jd_set else 0.0 for s in vocab]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

def rank_candidates(resumes: Sequence[Dict[str, str]], jobs: Sequence[Dict[str, str]]) -> List[Dict]:
    norm_resumes = [deduplicate_skills(normalize_skills(r["skills"])) for r in resumes]
    vocab = build_vocabulary(norm_resumes)
    df = compute_df(norm_resumes)
    idf = compute_idf(df, len(resumes))
    
    res_vecs = [build_resume_vec(s, vocab, idf) for s in norm_resumes]
    
    results = []
    for job in jobs:
        raw = f"{job.get('required', '')}, {job.get('preferred', '')}"
        jskills = deduplicate_skills(normalize_skills(raw))
        jvec = build_jd_vec(jskills, vocab)
        
        scores = []
        for i, res in enumerate(resumes):
            score = cosine_similarity(res_vecs[i], jvec)
            scores.append((res["name"], score))
            
        scores.sort(key=lambda x: (-x[1], x[0]))
        results.append({"job": job, "matches": scores[:3]})
    return results
