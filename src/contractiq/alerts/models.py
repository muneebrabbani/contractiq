from __future__ import annotations

from pydantic import BaseModel


class ExpiryAlert(BaseModel):
    doc_id: str
    source_file: str
    vendor: str | None
    business_unit: str | None
    department: str | None
    expiry_date: str
    days_until_expiry: int
    bucket: str  # "expired" | "30" | "60" | "90"


class UnparseableExpiry(BaseModel):
    doc_id: str
    source_file: str
    department: str | None
    expiry_date_raw: str
