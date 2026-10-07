"""Housing maintenance triage for remote NT communities.

Pipeline: intake -> extraction -> verification -> evaluation -> ranking -> explain.
A model reads each report for facts with quoted spans; code applies all policy.
"""
