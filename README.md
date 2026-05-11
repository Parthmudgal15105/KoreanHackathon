# Resume Matching Engine

**Redrob AI Campus Hackathon** — Korean Technology Companies Resume Matcher

A Python program that matches Indian university student resumes against Korean tech
company job descriptions using **TF-IDF** and **cosine similarity** — built with
only Python standard libraries.

## Architecture

```
data.py              ← Resumes, JDs, and SKILL_ALIASES (unmodified)
matching_engine.py   ← Core pipeline: normalize → dedup → TF-IDF → cosine
main.py              ← CLI runner with optional --verbose mode
tests.py             ← 30 tests covering every pipeline stage
```

## Pipeline Stages

| Stage | Function | Description |
|-------|----------|-------------|
| 1 | `normalize_skills()` | Split on commas, lowercase, match multi-word phrases first, apply alias map, discard unknown tokens |
| 2 | `deduplicate_skills()` | Remove duplicate canonical skills (order-preserving) |
| 3 | `build_vocabulary()` | Alphabetically sorted union of all resume skills |
| 4a | `compute_tf()` | TF = 1/N where N = unique skills in resume |
| 4b | `compute_idf()` | IDF = ln(10/df), no smoothing |
| 4c | `build_resume_tfidf_vector()` | TF × IDF per skill per resume |
| 5 | `build_jd_binary_vector()` | 1.0 if JD requires/prefers the skill, else 0.0 |
| 6 | `cosine_similarity()` | A·B / (‖A‖ × ‖B‖), 0.0 for zero vectors |
| 7 | `rank_candidates()` | Sort by score desc, ties broken alphabetically |

## Usage

```bash
# Standard output (submission format)
python main.py

# Verbose mode — shows all intermediate values for verification
python main.py --verbose

# Run all 30 tests
python tests.py
```

## Results

```
JD-1 — Kakao (ML Engineer)
Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)

JD-2 — Naver (Backend Engineer)
Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

JD-3 — Line (Frontend Engineer)
Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
```

## Constraints Met

- **No external libraries** — only `math`, `re`, `sys`, and `typing`
- **SKILL_ALIASES unmodified** — exact mapping from problem sheet
- **Exact formulas** — TF = 1/N, IDF = ln(10/df), cosine similarity
- **Alphabetical tiebreaking** on equal scores
- **Scores rounded to 2 decimal places**
