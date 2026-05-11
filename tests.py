from matching_engine import normalize_skills, deduplicate_skills, rank_candidates
from data import JOBS, RESUMES

def test_basic():
    # test some basics
    assert normalize_skills("Pyhton") == ["python"]
    assert normalize_skills("JavaScrpit") == ["javascript"]
    
    skills = normalize_skills("python, python, java")
    assert deduplicate_skills(skills) == ["python", "java"]

def test_results():
    results = rank_candidates(RESUMES, JOBS)
    
    # JD1 check
    jd1_top = [n for n, s in results[0]["matches"]]
    assert jd1_top == ["Sneha Patel", "Karan Mehta", "Arjun Sharma"]
    
    # JD2 check
    jd2_top = [n for n, s in results[1]["matches"]]
    assert jd2_top == ["Rahul Gupta", "Ananya Krishnan", "Deepika Rao"]

if __name__ == "__main__":
    test_basic()
    test_results()
    print("Tests passed!")
