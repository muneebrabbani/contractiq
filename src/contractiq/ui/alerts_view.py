from __future__ import annotations

import streamlit as st

from contractiq.alerts import compute_expiry_alerts, generate_digest

_BUCKET_TITLES = {
    "expired": "Expired",
    "30": "Expiring within 30 days",
    "60": "Expiring within 60 days",
    "90": "Expiring within 90 days",
}


def render() -> None:
    st.title("Contract Expiry Alerts")
    st.caption("Computed directly from the contract metadata table -- no LLM involved.")

    alerts, unparseable = compute_expiry_alerts()

    if not alerts:
        st.success("No contracts expiring within 90 days.")
    else:
        for bucket in ("expired", "30", "60", "90"):
            bucket_alerts = [a for a in alerts if a.bucket == bucket]
            if not bucket_alerts:
                continue
            st.subheader(_BUCKET_TITLES[bucket])
            st.table(
                [
                    {
                        "Document": a.source_file,
                        # No document-versioning system exists yet -- every
                        # contract is a single upload, so this is always 1.
                        "Version": 1,
                        "Business Unit": a.business_unit or "-",
                        "Vendor": a.vendor or "-",
                        "Department": a.department or "-",
                        "Expiry date": a.expiry_date,
                        "Days": a.days_until_expiry,
                    }
                    for a in bucket_alerts
                ]
            )

    if unparseable:
        with st.expander(
            f"{len(unparseable)} contract(s) need manual review (unparseable expiry date)"
        ):
            st.table(
                [
                    {
                        "Document": u.source_file,
                        "Department": u.department or "-",
                        "Stated expiry": u.expiry_date_raw,
                    }
                    for u in unparseable
                ]
            )

    st.divider()
    st.subheader("Per-owner digest preview")
    st.text(generate_digest())
