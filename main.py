"""Run the Redrob resume matching engine.

Usage:
    python main.py            # Normal output
    python main.py --verbose  # Show intermediate pipeline values
"""

import sys

from data import JOBS, RESUMES
from matching_engine import (
    build_jd_binary_vector,
    build_resume_tfidf_vector,
    build_vocabulary,
    compute_document_frequency,
    compute_idf,
    compute_tf,
    cosine_similarity,
    deduplicate_skills,
    normalize_skills,
    rank_candidates,
)


def show_verbose_pipeline() -> None:
    """Print every intermediate value for manual verification."""

    # ── Step 1 + 2: Normalize & Deduplicate ──
    print("=" * 70)
    print("STEP 1–2: Skill Normalization & Deduplication")
    print("=" * 70)
    normalized_resumes = []
    for resume in RESUMES:
        raw = normalize_skills(resume["skills"])
        deduped = deduplicate_skills(raw)
        normalized_resumes.append(deduped)
        print(f"  [{resume['id']}] {resume['name']}")
        print(f"       Raw input : {resume['skills']}")
        print(f"       Normalized: {raw}")
        print(f"       Deduped   : {deduped}  (N={len(deduped)})")

    # ── Step 3: Vocabulary ──
    print(f"\n{'=' * 70}")
    print("STEP 3: Vocabulary Construction")
    print("=" * 70)
    vocab = build_vocabulary(normalized_resumes)
    print(f"  Size: {len(vocab)} skills")
    for i, skill in enumerate(vocab):
        print(f"    [{i:2d}] {skill}")

    # ── Step 4: TF-IDF ──
    print(f"\n{'=' * 70}")
    print("STEP 4: TF-IDF Computation")
    print("=" * 70)
    df = compute_document_frequency(normalized_resumes)
    idf = compute_idf(df, len(RESUMES))

    print("\n  Document Frequencies:")
    for skill in vocab:
        print(f"    {skill:30s} df={df.get(skill, 0)}  IDF={idf.get(skill, 0.0):.4f}")

    print("\n  TF-IDF Vectors (non-zero entries only):")
    resume_vectors = []
    for resume, skills in zip(RESUMES, normalized_resumes):
        vec = build_resume_tfidf_vector(skills, vocab, idf)
        resume_vectors.append(vec)
        tf = compute_tf(skills)
        non_zero = [(vocab[i], v, tf.get(vocab[i], 0.0), idf.get(vocab[i], 0.0))
                     for i, v in enumerate(vec) if v > 0]
        print(f"\n  [{resume['id']}] {resume['name']} (N={len(skills)}, TF=1/{len(skills)}):")
        for skill, tfidf, tf_val, idf_val in non_zero:
            print(f"      {skill:30s} TF={tf_val:.4f} x IDF={idf_val:.4f} = {tfidf:.4f}")

    # ── Step 5: JD Binary Vectors ──
    print(f"\n{'=' * 70}")
    print("STEP 5: JD Binary Vectors")
    print("=" * 70)
    jd_vectors = []
    for job in JOBS:
        raw_skills = f"{job.get('required', '')}, {job.get('preferred', '')}"
        jd_skills = deduplicate_skills(normalize_skills(raw_skills))
        jd_vec = build_jd_binary_vector(jd_skills, vocab)
        jd_vectors.append(jd_vec)
        matched = [vocab[i] for i, v in enumerate(jd_vec) if v == 1.0]
        print(f"\n  {job['id']} — {job['company']} ({job['role']})")
        print(f"    Normalized JD skills: {jd_skills}")
        print(f"    Matched in vocab    : {matched}")

    # ── Step 6: Cosine Similarity & Ranking ──
    print(f"\n{'=' * 70}")
    print("STEP 6: Cosine Similarity & Ranking")
    print("=" * 70)
    for job, jd_vec in zip(JOBS, jd_vectors):
        print(f"\n  {job['id']} — {job['company']} ({job['role']})")
        scores = []
        for resume, rv in zip(RESUMES, resume_vectors):
            score = cosine_similarity(rv, jd_vec)
            scores.append((resume["name"], score))
            print(f"    {resume['name']:25s} -> {score:.4f}")
        scores.sort(key=lambda x: (-x[1], x[0]))
        print(f"    {'-' * 40}")
        print(f"    Top 3: {', '.join(f'{n}({s:.2f})' for n, s in scores[:3])}")


def main() -> None:
    verbose = "--verbose" in sys.argv

    if verbose:
        show_verbose_pipeline()
        print(f"\n{'=' * 70}")
        print("FINAL OUTPUT")
        print("=" * 70)

    ranked_jobs = rank_candidates(RESUMES, JOBS)

    for index, ranked_job in enumerate(ranked_jobs):
        job = ranked_job["job"]
        matches = ranked_job["matches"]

        print(f"{job['id']} — {job['company']} ({job['role']})")
        print(", ".join(f"{name}({score:.2f})" for name, score in matches))

        if index != len(ranked_jobs) - 1:
            print()


if __name__ == "__main__":
    main()
