from __future__ import annotations

import datetime

from contractiq.alerts.models import ExpiryAlert, UnparseableExpiry
from contractiq.extraction.db import ContractRecord, get_session, get_superseded_doc_ids

# Mutually exclusive thresholds, checked tightest-first.
_THRESHOLDS = [
    (30, "30"),
    (60, "60"),
    (90, "90"),
]


def _bucket_for(days: int) -> str | None:
    if days < 0:
        return "expired"
    for threshold, label in _THRESHOLDS:
        if days <= threshold:
            return label
    return None  # beyond 90 days -- not alert-worthy


def compute_expiry_alerts(
    as_of: datetime.date | None = None,
) -> tuple[list[ExpiryAlert], list[UnparseableExpiry]]:
    """Pure computation over the metadata table -- no LLM, no retrieval.
    Buckets are mutually exclusive (a contract expiring in 12 days appears
    only in the 30-day bucket, not also 60/90), assigned by tightest
    threshold. Dates that don't parse as ISO 8601 go into a separate
    "needs manual review" list rather than being silently dropped."""
    as_of = as_of or datetime.date.today()

    session = get_session()
    superseded = get_superseded_doc_ids(session)
    records = session.query(ContractRecord).filter(ContractRecord.expiry_date.isnot(None)).all()
    session.close()

    alerts: list[ExpiryAlert] = []
    unparseable: list[UnparseableExpiry] = []

    for record in records:
        if record.doc_id in superseded:
            continue  # a later amendment/renewal already supersedes this one
        raw = record.expiry_date
        try:
            expiry = datetime.date.fromisoformat(raw)
        except (ValueError, TypeError):
            unparseable.append(
                UnparseableExpiry(
                    doc_id=record.doc_id,
                    source_file=record.source_file,
                    department=record.department,
                    expiry_date_raw=raw,
                )
            )
            continue

        days = (expiry - as_of).days
        bucket = _bucket_for(days)
        if bucket is None:
            continue

        alerts.append(
            ExpiryAlert(
                doc_id=record.doc_id,
                source_file=record.source_file,
                vendor=record.vendor,
                business_unit=record.business_unit,
                department=record.department,
                expiry_date=raw,
                days_until_expiry=days,
                bucket=bucket,
            )
        )

    alerts.sort(key=lambda a: a.days_until_expiry)
    return alerts, unparseable
