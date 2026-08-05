from __future__ import annotations

import datetime

from contractiq.alerts.compute import compute_expiry_alerts
from contractiq.alerts.models import ExpiryAlert


def generate_digest(as_of: datetime.date | None = None) -> str:
    """Groups expiry alerts by department -- zero Streamlit dependency, callable
    from an external scheduler (cron, docker-compose sidecar) independent
    of the running UI process. Run directly: python -m contractiq.alerts.digest"""
    as_of = as_of or datetime.date.today()
    alerts, unparseable = compute_expiry_alerts(as_of)

    if not alerts and not unparseable:
        return f"Contract expiry digest ({as_of.isoformat()}): no upcoming expiries.\n"

    by_department: dict[str, list[ExpiryAlert]] = {}
    for alert in alerts:
        department = alert.department or "(no department assigned)"
        by_department.setdefault(department, []).append(alert)

    lines = [f"Contract expiry digest ({as_of.isoformat()})", ""]

    for department in sorted(by_department):
        department_alerts = by_department[department]
        lines.append(f"{department}: {len(department_alerts)} contract(s)")
        for a in department_alerts:
            label = "EXPIRED" if a.bucket == "expired" else f"{a.days_until_expiry} days"
            lines.append(
                f"  - {a.source_file} ({a.vendor or 'unknown vendor'}): {label} ({a.expiry_date})"
            )
        lines.append("")

    if unparseable:
        lines.append(f"{len(unparseable)} contract(s) have an expiry date that needs manual review:")
        for u in unparseable:
            lines.append(
                f"  - {u.source_file} (department: {u.department or 'unassigned'}): "
                f"{u.expiry_date_raw!r}"
            )

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    print(generate_digest())
