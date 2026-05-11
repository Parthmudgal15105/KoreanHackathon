# Resume Matching Engine - Hackathon Solution

Matches resumes to job descriptions using TF-IDF and Cosine Similarity.

## How to run
1. `python main.py` for results
2. `python tests.py` to check logic

## Features
- Skill normalization (typo matching and alias mapping)
- TF-IDF calculation (TF=1/N, IDF=ln(10/df))
- Cosine similarity ranking with alphabetical tie-breaking
- Uses standard libraries only (re, math)
