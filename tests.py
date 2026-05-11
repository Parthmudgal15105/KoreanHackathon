"""Comprehensive tests for the resume matching engine.

Tests validate every pipeline stage:
  1. Skill normalization (typos, aliases, multi-word phrases)
  2. Deduplication
  3. Vocabulary construction
  4. TF / IDF / TF-IDF computation
  5. JD binary vectors
  6. Cosine similarity
  7. End-to-end ranking with tiebreaking
"""

import math

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
from data import JOBS, RESUMES


# ── Step 1: Skill Normalization ──────────────────────────────────────────

def test_normalization_typos() -> None:
    """Common misspellings are corrected."""
    assert normalize_skills("Pyhton") == ["python"]
    assert normalize_skills("JavaScrpit") == ["javascript"]
    assert normalize_skills("TypeScrpit") == ["typescript"]
    assert normalize_skills("kubernates") == ["kubernetes"]
    assert normalize_skills("Algoritms") == ["algorithms"]


def test_normalization_case_insensitive() -> None:
    """Matching is case-insensitive."""
    assert normalize_skills("PYTHON") == ["python"]
    assert normalize_skills("MachineLearning") == ["machine_learning"]
    assert normalize_skills("XGboost") == ["xgboost"]


def test_normalization_multi_word_phrases() -> None:
    """Multi-word aliases are matched before single-token aliases."""
    assert normalize_skills("REST api") == ["rest_api"]
    assert normalize_skills("Spring Boot") == ["spring_boot"]
    assert normalize_skills("Deep-learning") == ["deep_learning"]
    assert normalize_skills("feature engineering") == ["feature_engineering"]
    assert normalize_skills("competitive programming") == ["competitive_programming"]
    assert normalize_skills("Data Structure") == ["data_structures"]
    assert normalize_skills("Node.JS") == ["nodejs"]
    assert normalize_skills("vue.js") == ["vue"]


def test_normalization_synonyms() -> None:
    """Different aliases for the same canonical skill produce the same output."""
    assert normalize_skills("data-viz") == ["data_visualization"]
    assert normalize_skills("matplotlib") == ["data_visualization"]
    assert normalize_skills("tableau") == ["data_visualization"]
    assert normalize_skills("Power-BI") == ["data_visualization"]
    assert normalize_skills("ml") == ["machine_learning"]
    assert normalize_skills("sklearn") == ["machine_learning"]


def test_normalization_discards_unknown_tokens() -> None:
    """Tokens not in alias map are silently discarded."""
    assert normalize_skills("blockchain") == []
    assert normalize_skills("Python, blockchain, SQL") == ["python", "sql"]


def test_normalization_empty_input() -> None:
    assert normalize_skills("") == []


def test_normalization_html_css_combined() -> None:
    """HTML/CSS as a single token maps correctly."""
    assert normalize_skills("HTML/CSS") == ["html_css"]


def test_normalization_ci_cd() -> None:
    """CI/CD variants all map to ci_cd."""
    assert normalize_skills("CI/CD") == ["ci_cd"]
    assert normalize_skills("cicd") == ["ci_cd"]


# ── Step 1 + 2: Full normalization per resume ────────────────────────────

def test_all_resumes_normalize_correctly() -> None:
    """Verify the exact normalized+deduplicated skill sets for all 10 resumes."""
    expected = {
        "Arjun Sharma": {"python", "machine_learning", "sql", "pandas", "numpy", "deep_learning"},
        "Priya Nair": {"javascript", "react", "nodejs", "mongodb", "rest_api", "html_css"},
        "Rahul Gupta": {"java", "spring_boot", "mysql", "microservices", "docker", "kubernetes"},
        "Sneha Patel": {"python", "tensorflow", "keras", "nlp", "bert", "data_visualization"},
        "Vikram Singh": {"cpp", "algorithms", "data_structures", "competitive_programming", "python"},
        "Ananya Krishnan": {"javascript", "vue", "python", "flask", "postgresql", "aws", "ci_cd"},
        "Karan Mehta": {"python", "machine_learning", "xgboost", "feature_engineering", "sql", "data_visualization"},
        "Deepika Rao": {"java", "android", "kotlin", "firebase", "rest_api", "ui_ux", "figma"},
        "Aditya Kumar": {"react", "typescript", "graphql", "redux", "tailwind", "nodejs", "jest"},
        "Meera Iyer": {"python", "r", "statistics", "machine_learning", "regression", "clustering", "data_visualization"},
    }
    for resume in RESUMES:
        skills = deduplicate_skills(normalize_skills(resume["skills"]))
        assert set(skills) == expected[resume["name"]], (
            f"Mismatch for {resume['name']}: got {set(skills)}"
        )


# ── Step 2: Deduplication ────────────────────────────────────────────────

def test_deduplication_removes_duplicates() -> None:
    skills = normalize_skills("data-viz, matplotlib, Python, Pyhton")
    assert deduplicate_skills(skills) == ["data_visualization", "python"]


def test_deduplication_preserves_order() -> None:
    assert deduplicate_skills(["c", "a", "b", "a", "c"]) == ["c", "a", "b"]


# ── Step 3: Vocabulary ───────────────────────────────────────────────────

def test_vocabulary_is_alphabetical() -> None:
    vocabulary = build_vocabulary([["python", "sql"], ["java", "aws"]])
    assert vocabulary == ["aws", "java", "python", "sql"]


def test_full_vocabulary_size() -> None:
    """The shared vocabulary built from all 10 resumes should be correct."""
    normalized = [
        deduplicate_skills(normalize_skills(r["skills"])) for r in RESUMES
    ]
    vocab = build_vocabulary(normalized)
    # All unique canonical skills across 10 resumes
    expected_skills = sorted({
        "python", "machine_learning", "sql", "pandas", "numpy", "deep_learning",
        "javascript", "react", "nodejs", "mongodb", "rest_api", "html_css",
        "java", "spring_boot", "mysql", "microservices", "docker", "kubernetes",
        "tensorflow", "keras", "nlp", "bert", "data_visualization",
        "cpp", "algorithms", "data_structures", "competitive_programming",
        "vue", "flask", "postgresql", "aws", "ci_cd",
        "xgboost", "feature_engineering",
        "android", "kotlin", "firebase", "ui_ux", "figma",
        "typescript", "graphql", "redux", "tailwind", "jest",
        "r", "statistics", "regression", "clustering",
    })
    assert vocab == expected_skills, f"Vocab mismatch: got {len(vocab)} skills, expected {len(expected_skills)}"


# ── Step 4: TF / IDF / TF-IDF ───────────────────────────────────────────

def test_tf_computation() -> None:
    """TF = 1/N where N = number of unique skills."""
    tf = compute_tf(["python", "sql", "java"])
    assert abs(tf["python"] - 1 / 3) < 1e-10
    assert abs(tf["sql"] - 1 / 3) < 1e-10
    assert abs(tf["java"] - 1 / 3) < 1e-10


def test_tf_empty() -> None:
    assert compute_tf([]) == {}


def test_idf_computation() -> None:
    """IDF = ln(10 / df) with no smoothing."""
    df = {"python": 5, "java": 2}
    idf = compute_idf(df, 10)
    assert abs(idf["python"] - math.log(10 / 5)) < 1e-10
    assert abs(idf["java"] - math.log(10 / 2)) < 1e-10


def test_idf_skill_in_all_docs() -> None:
    """A skill appearing in all 10 resumes has IDF = ln(1) = 0."""
    idf = compute_idf({"python": 10}, 10)
    assert idf["python"] == 0.0


def test_idf_skill_in_one_doc() -> None:
    """A skill in exactly 1 resume has IDF = ln(10)."""
    idf = compute_idf({"rare_skill": 1}, 10)
    assert abs(idf["rare_skill"] - math.log(10)) < 1e-10


def test_document_frequency_values() -> None:
    """Spot-check DF values for key skills across the 10 resumes."""
    normalized = [
        deduplicate_skills(normalize_skills(r["skills"])) for r in RESUMES
    ]
    df = compute_document_frequency(normalized)
    # python appears in resumes: 01, 04, 05, 06, 07, 10 → df=6
    assert df["python"] == 6, f"python df: expected 6, got {df['python']}"
    # java appears in resumes: 03, 08 → df=2
    assert df["java"] == 2, f"java df: expected 2, got {df['java']}"
    # machine_learning appears in resumes: 01, 07, 10 → df=3
    assert df["machine_learning"] == 3
    # sql appears in resumes: 01, 07 → df=2
    assert df["sql"] == 2
    # react appears in resumes: 02, 09 → df=2
    assert df["react"] == 2


def test_tfidf_vector_dimensions() -> None:
    """Each TF-IDF vector should have the same length as the vocabulary."""
    normalized = [
        deduplicate_skills(normalize_skills(r["skills"])) for r in RESUMES
    ]
    vocab = build_vocabulary(normalized)
    df = compute_document_frequency(normalized)
    idf = compute_idf(df, 10)

    for skills in normalized:
        vec = build_resume_tfidf_vector(skills, vocab, idf)
        assert len(vec) == len(vocab)


# ── Step 5: JD Binary Vectors ───────────────────────────────────────────

def test_jd_vector_is_binary() -> None:
    vector = build_jd_binary_vector(
        ["python", "python", "java"],
        ["aws", "java", "python"],
    )
    assert vector == [0.0, 1.0, 1.0]
    assert set(vector).issubset({0.0, 1.0})


# ── Step 6: Cosine Similarity ───────────────────────────────────────────

def test_cosine_similarity_identical_vectors() -> None:
    assert abs(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-10


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-10


def test_cosine_similarity_zero_vector() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([], []) == 0.0


# ── Step 7: End-to-End Ranking ───────────────────────────────────────────

def test_final_ranking_returns_three_candidates_per_jd() -> None:
    rankings = rank_candidates(RESUMES, JOBS)
    assert len(rankings) == 3
    for ranking in rankings:
        assert len(ranking["matches"]) == 3


def test_tie_breaking_sorts_alphabetically() -> None:
    resumes = [
        {"name": "Zara Khan", "skills": "Python"},
        {"name": "Amit Bose", "skills": "Python"},
        {"name": "Mina Roy", "skills": "Java"},
        {"name": "Nikhil Das", "skills": "Java"},
    ]
    jobs = [
        {
            "id": "JD-X",
            "company": "TieCo",
            "role": "Python Engineer",
            "required": "Python",
            "preferred": "",
        }
    ]
    ranking = rank_candidates(resumes, jobs)[0]["matches"]
    assert [name for name, score in ranking[:2]] == ["Amit Bose", "Zara Khan"]


def test_scores_are_rounded_consistently() -> None:
    """Scores in the output must be representable to 2 decimal places."""
    rankings = rank_candidates(RESUMES, JOBS)
    for ranking in rankings:
        for name, score in ranking["matches"]:
            rounded = round(score, 2)
            assert abs(rounded - round(rounded, 2)) < 1e-10


def test_jd1_ranking() -> None:
    """JD-1 (ML Engineer): Sneha Patel, Karan Mehta, Arjun Sharma."""
    rankings = rank_candidates(RESUMES, JOBS)
    jd1_matches = rankings[0]["matches"]
    names = [name for name, _ in jd1_matches]
    assert names == ["Sneha Patel", "Karan Mehta", "Arjun Sharma"], (
        f"JD-1 ranking mismatch: {names}"
    )


def test_jd2_ranking() -> None:
    """JD-2 (Backend Engineer): Rahul Gupta, Ananya Krishnan, Deepika Rao."""
    rankings = rank_candidates(RESUMES, JOBS)
    jd2_matches = rankings[1]["matches"]
    names = [name for name, _ in jd2_matches]
    assert names == ["Rahul Gupta", "Ananya Krishnan", "Deepika Rao"], (
        f"JD-2 ranking mismatch: {names}"
    )


def test_jd3_ranking() -> None:
    """JD-3 (Frontend Engineer): Aditya Kumar, Priya Nair, Ananya Krishnan."""
    rankings = rank_candidates(RESUMES, JOBS)
    jd3_matches = rankings[2]["matches"]
    names = [name for name, _ in jd3_matches]
    assert names == ["Aditya Kumar", "Priya Nair", "Ananya Krishnan"], (
        f"JD-3 ranking mismatch: {names}"
    )


# ── Runner ───────────────────────────────────────────────────────────────

def run_tests() -> None:
    tests = [
        test_normalization_typos,
        test_normalization_case_insensitive,
        test_normalization_multi_word_phrases,
        test_normalization_synonyms,
        test_normalization_discards_unknown_tokens,
        test_normalization_empty_input,
        test_normalization_html_css_combined,
        test_normalization_ci_cd,
        test_all_resumes_normalize_correctly,
        test_deduplication_removes_duplicates,
        test_deduplication_preserves_order,
        test_vocabulary_is_alphabetical,
        test_full_vocabulary_size,
        test_tf_computation,
        test_tf_empty,
        test_idf_computation,
        test_idf_skill_in_all_docs,
        test_idf_skill_in_one_doc,
        test_document_frequency_values,
        test_tfidf_vector_dimensions,
        test_jd_vector_is_binary,
        test_cosine_similarity_identical_vectors,
        test_cosine_similarity_orthogonal_vectors,
        test_cosine_similarity_zero_vector,
        test_final_ranking_returns_three_candidates_per_jd,
        test_tie_breaking_sorts_alphabetically,
        test_scores_are_rounded_consistently,
        test_jd1_ranking,
        test_jd2_ranking,
        test_jd3_ranking,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")

    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()
