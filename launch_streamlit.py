"""
Utility script: launches the Streamlit app on localhost:8085 and verifies it.
Writes progress/status to streamlit_status.txt for reliable verification.
"""
import subprocess
import sys
import os
import time
import urllib.request

PROJECT_DIR = r'E:\CDAC BENGALORE\Project\project_NSE_500_News_Filter_real_data\project_NSE_500_News_Filter'
os.chdir(PROJECT_DIR)

STATUS_FILE = os.path.join(PROJECT_DIR, 'streamlit_status.txt')
OUT_LOG = os.path.join(PROJECT_DIR, 'streamlit_out.log')
ERR_LOG = os.path.join(PROJECT_DIR, 'streamlit_err.log')


def log(msg):
    with open(STATUS_FILE, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')


# Reset status file
with open(STATUS_FILE, 'w', encoding='utf-8') as f:
    f.write('LAUNCHING...\n')

try:
    proc = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', 'app.py',
         '--server.port', '8085', '--server.headless', 'true',
         '--browser.gatherUsageStats', 'false'],
        cwd=PROJECT_DIR,
        stdout=open(OUT_LOG, 'w', encoding='utf-8'),
        stderr=open(ERR_LOG, 'w', encoding='utf-8'),
        creationflags=subprocess.CREATE_NO_WINDOW
    )
    log(f'STARTED PID={proc.pid}')
except Exception as e:
    log(f'START_ERROR={e}')
    sys.exit(1)

# Health check loop (up to ~60s)
healthy = False
for i in range(30):
    time.sleep(2)
    try:
        r = urllib.request.urlopen('http://localhost:8085/_stcore/health', timeout=3)
        log(f'HEALTH_OK status={r.status}')
        healthy = True
        break
    except Exception as e:
        log(f'TRY{i}={str(e)[:80]}')

if healthy:
    log('APP_IS_RUNNING')
else:
    log('APP_NOT_REACHABLE - check streamlit_err.log')

