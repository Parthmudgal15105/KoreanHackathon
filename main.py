import sys
from data import JOBS, RESUMES
from matching_engine import rank_candidates

def main():
    # check for debug mode
    debug = "--debug" in sys.argv or "-v" in sys.argv
    
    results = rank_candidates(RESUMES, JOBS)

    for i, res in enumerate(results):
        job = res["job"]
        matches = res["matches"]

        print(f"{job['id']} — {job['company']} ({job['role']})")
        print(", ".join(f"{name}({score:.2f})" for name, score in matches))
        
        if i < len(results) - 1:
            print()

if __name__ == "__main__":
    main()
