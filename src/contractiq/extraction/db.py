from __future__ import annotations

import datetime
from pathlib import Path

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from contractiq.extraction.models import ContractMetadata

PROCESSED_DIR = Path("data/processed")
DB_PATH = PROCESSED_DIR / "contractiq.sqlite3"

Base = declarative_base()


class ContractRecord(Base):
    __tablename__ = "contracts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, unique=True, nullable=False)
    source_file = Column(String, nullable=False)
    contract_number = Column(String)
    agreement_title = Column(String)
    contract_type = Column(String)
    agreement_category = Column(String)
    vendor = Column(String)
    project_name = Column(String)
    business_unit = Column(String)
    department = Column(String)
    segment = Column(String)
    contract_description = Column(String)
    status = Column(String)
    execution_date = Column(String)
    effective_date = Column(String)
    expiry_date = Column(String)
    renewal_date = Column(String)
    contract_duration = Column(String)
    notice_period = Column(String)
    auto_renewal = Column(Boolean)
    value = Column(Float)
    currency = Column(String)
    payment_terms = Column(String)
    payment_milestones = Column(String)
    signatory_names = Column(JSON)
    extracted_at = Column(DateTime, default=datetime.datetime.utcnow)
    # doc_id of the contract this document amends/renews, if any. Set only via
    # the upload UI -- never LLM-extracted (see extraction/pipeline.py).
    related_doc_id = Column(String)


def _migrate_schema(engine) -> None:
    """create_all() only creates missing tables, not missing columns on an
    existing one -- this repo's contracts table already has real extracted
    data, so new columns need an explicit, idempotent ALTER TABLE."""
    with engine.begin() as conn:
        existing_cols = {
            row[1] for row in conn.exec_driver_sql("PRAGMA table_info(contracts)").fetchall()
        }
        if "related_doc_id" not in existing_cols:
            conn.exec_driver_sql("ALTER TABLE contracts ADD COLUMN related_doc_id VARCHAR")


def get_engine(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    _migrate_schema(engine)
    return engine


def get_session(db_path: Path = DB_PATH) -> Session:
    return sessionmaker(bind=get_engine(db_path))()


def save_record(
    session: Session,
    doc_id: str,
    source_file: str,
    metadata: ContractMetadata,
    related_doc_id: str | None = None,
) -> ContractRecord:
    fields = metadata.model_dump()
    record = session.query(ContractRecord).filter_by(doc_id=doc_id).one_or_none()
    if record is None:
        record = ContractRecord(doc_id=doc_id, source_file=source_file)
        session.add(record)
    for key, value in fields.items():
        setattr(record, key, value)
    # Only overwrite when explicitly passed, so re-running extraction never
    # silently clears a link a user set via the upload UI.
    if related_doc_id is not None:
        record.related_doc_id = related_doc_id
    record.extracted_at = datetime.datetime.utcnow()
    session.commit()
    return record


def get_superseded_doc_ids(session: Session) -> set[str]:
    """doc_ids that some other record's related_doc_id points at -- i.e.
    contracts that have since been amended/renewed by a later document."""
    rows = session.query(ContractRecord).filter(ContractRecord.related_doc_id.isnot(None)).all()
    return {r.related_doc_id for r in rows}
