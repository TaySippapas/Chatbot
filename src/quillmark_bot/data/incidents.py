# Root causes are only written down in agent notes, in many template variants per incident,
# so they are summarized here by hand from those notes (see notebooks/01_eda.ipynb §8).
ROOT_CAUSES = {
    "INC-2025-06": (
        "Elm Park escrow re-analyses run in June 2025 used prior-year assessment data from the "
        "tax data vendor, overstating property taxes and creating false escrow shortages. "
        "Corrected analyses were mailed 07/14/2025 and overpayments refunded by check."
    ),
    "INC-2025-11": (
        "November-cycle Horizon paper statements for Riverton and Cedar Falls were mailed 9 days "
        "late by print vendor Ashgrove Print Services, causing late fees. Fees were reversed under "
        "POL-DSP-04 without counting toward the courtesy waiver (POL-CRD-06)."
    ),
    "INC-2026-02": (
        "Third-party agency Brightway Recovery Partners, assigned charged-off Spark Card accounts in "
        "September 2025, broke collection contact rules (POL-COL-02): too many call attempts, calls "
        "outside allowed hours, workplace calls and added fees. Accounts were recalled to Quillmark "
        "Recovery Services and Brightway-added fees removed."
    ),
    "INC-2026-03": (
        "Card processing outage from 03/14/2026 9:40 p.m. to 03/15/2026 3:15 a.m. posted duplicate "
        "card charges. Duplicates were reversed automatically, resulting overdraft fees reversed "
        "under POL-DSP-04, and a $25 goodwill credit issued."
    ),
    "INC-2026-07": (
        "App version 6.2.0 (released 07/08/2026) locked accounts after a single failed sign-in and "
        "rejected valid one-time codes. Fixed in 6.2.1 (07/10/2026); late fees caused by the "
        "lockout were reversed under POL-DSP-04."
    ),
}

# Fills every other incidents column from the cases that carry the incident_id
REFRESH_INCIDENT_STATS = """
UPDATE incidents i
SET issue          = s.issue,
    branches       = s.branches,
    first_opened   = s.first_opened,
    last_opened    = s.last_opened,
    case_count     = s.case_count,
    total_refunded = s.total_refunded
FROM (
    SELECT incident_id,
           string_agg(DISTINCT issue, ', ')        AS issue,
           array_agg(DISTINCT branch ORDER BY branch) AS branches,
           min(opened_date)                        AS first_opened,
           max(opened_date)                        AS last_opened,
           count(*)                                AS case_count,
           sum(amount_refunded)                    AS total_refunded
    FROM cases
    WHERE incident_id IS NOT NULL
    GROUP BY incident_id
) s
WHERE i.incident_id = s.incident_id
"""
