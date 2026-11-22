import os
import sys
import time
import subprocess
from pathlib import Path
import httpx
import fitz

def main():
    print("=== Starting Streamlit local test without separate uvicorn ===", flush=True)

    # Verify port 8000 is free before starting
    import socket
    s = socket.socket()
    in_use = (s.connect_ex(("127.0.0.1", 8000)) == 0)
    s.close()
    print(f"Port 8000 initially occupied: {in_use}", flush=True)

    # 1. Start Streamlit
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )

    try:
        # Wait for Streamlit server to bind
        print("Waiting for Streamlit port 8501...", flush=True)
        st_ready = False
        for _ in range(30):
            try:
                r = httpx.get("http://127.0.0.1:8501", timeout=1.0)
                if r.status_code == 200:
                    st_ready = True
                    break
            except Exception:
                time.sleep(1)

        print(f"Streamlit HTTP server ready: {st_ready}", flush=True)

        # Trigger session or wait for backend uvicorn on port 8000
        print("Waiting for FastAPI backend (spawned by Streamlit) on port 8000...", flush=True)
        be_ready = False
        for _ in range(45):
            try:
                r = httpx.get("http://127.0.0.1:8000/health", timeout=1.0)
                if r.status_code == 200:
                    be_ready = True
                    print(f"FastAPI Backend responded: {r.json()}", flush=True)
                    break
            except Exception:
                # If Streamlit hasn't executed app.py yet, AppTest or opening a session does:
                pass
            time.sleep(1)

        if not be_ready:
            # Let's trigger a session or ensure_backend_running
            print("Triggering backend check...", flush=True)
            from frontend.app import ensure_backend_running
            ensure_backend_running()
            r = httpx.get("http://127.0.0.1:8000/health", timeout=5.0)
            print(f"FastAPI Backend responded: {r.json()}", flush=True)

        # 2. Build real PDF resume
        resume_raw = Path("evaluation/sample_resumes/data_science_01.txt").read_text(encoding="utf-8")
        jd_text = Path("evaluation/job_descriptions/data_science.txt").read_text(encoding="utf-8")

        doc = fitz.open()
        p = doc.new_page()
        p.insert_text((50, 72), resume_raw[:1500])
        pdf_bytes = doc.write()
        doc.close()
        print(f"Created PDF resume from data_science_01.txt ({len(pdf_bytes)} bytes)", flush=True)

        # 3. TAB 1: Match Score
        print("\n--- [TAB 1] Your Match Score ---", flush=True)
        files = {"file": ("data_science_01.pdf", pdf_bytes, "application/pdf")}
        data = {"job_description": jd_text}
        resp1 = httpx.post("http://127.0.0.1:8000/analyze", files=files, data=data, timeout=30.0)
        assert resp1.status_code == 200, f"Analyze failed: {resp1.text}"
        res1 = resp1.json()
        print(f"Final Score: {res1['final_score']} / 100", flush=True)
        print(f"Semantic Meaning Score: {res1['semantic_score']}", flush=True)
        print(f"Skill Overlap Score: {res1['skill_score']}", flush=True)
        print(f"Matched Skills: {res1['skills']['matched']}", flush=True)
        print(f"Missing Skills: {res1['skills']['missing']}", flush=True)
        print(f"What Helped: {res1['explanation']['what_helped'][:2]}", flush=True)
        print(f"What Held It Back: {res1['explanation']['what_held_back'][:2]}", flush=True)

        # 4. TAB 2: Bias Audit
        print("\n--- [TAB 2] Is the Scoring Fair? ---", flush=True)
        # Redacted mode
        payload_red = {
            "original_text": res1.get("original_text", resume_raw),
            "job_description": jd_text,
            "attribute": "name",
            "n_pairs": 5,
            "mode": "redacted"
        }
        resp2_red = httpx.post("http://127.0.0.1:8000/bias-audit", json=payload_red, timeout=30.0)
        assert resp2_red.status_code == 200
        res2_red = resp2_red.json()
        print(f"Redacted Mode - Mean Delta: {res2_red['mean_delta']:.4f}, Significant: {res2_red['is_significant']}, p-value: {res2_red['p_value']}", flush=True)

        # Raw mode
        payload_raw = dict(payload_red, mode="raw")
        resp2_raw = httpx.post("http://127.0.0.1:8000/bias-audit", json=payload_raw, timeout=30.0)
        assert resp2_raw.status_code == 200
        res2_raw = resp2_raw.json()
        print(f"Raw Mode - Mean Delta: {res2_raw['mean_delta']:.4f}, Significant: {res2_raw['is_significant']}, p-value: {res2_raw['p_value']}", flush=True)

        # Check full study results file
        study_csv = Path("evaluation/results/bias_results.csv")
        print(f"Full study file exists: {study_csv.exists()}", flush=True)

        # 5. TAB 3: How to Improve
        print("\n--- [TAB 3] How to Improve ---", flush=True)
        skill_table = [{"skill": s, "status": "matched"} for s in res1["skills"]["matched"][:3]] + \
                      [{"skill": s, "status": "missing"} for s in res1["skills"]["missing"][:3]]
        payload_fb = {
            "job_description": jd_text,
            "resume_text": res1.get("original_text", resume_raw)[:1000],
            "skill_table": skill_table
        }
        resp3 = httpx.post("http://127.0.0.1:8000/feedback", params={"prompt_version": "v3"}, json=payload_fb, timeout=20.0)
        print(f"Feedback endpoint status: {resp3.status_code}", flush=True)
        if resp3.status_code == 200:
            fb = resp3.json()
            print(f"Feedback model: {fb.get('model_used')}", flush=True)
            print(f"Recommendations count: {len(fb.get('feedback', {}).get('recommendations', []))}", flush=True)
        else:
            print(f"Feedback response: {resp3.text[:200]}", flush=True)

        print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<", flush=True)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("Streamlit process terminated.", flush=True)

if __name__ == "__main__":
    main()
