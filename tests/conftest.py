import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import HealthCheck, settings

# Property tests build queues of up to 40 jobs; on a slow machine Hypothesis's
# input-speed health check can fail a run that would otherwise pass.
settings.register_profile("triage", suppress_health_check=[HealthCheck.too_slow], deadline=None)
settings.load_profile("triage")
