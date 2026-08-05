from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClauseType(str, Enum):
    TERMINATION = "termination"
    GOVERNING_LAW = "governing_law"
    PAYMENT_TERMS = "payment_terms"
    CONFIDENTIALITY = "confidentiality"
    INDEMNIFICATION = "indemnification"
    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    NOTICES = "notices"
    WARRANTIES = "warranties"
    ASSIGNMENT = "assignment"
    FORCE_MAJEURE = "force_majeure"
    DISPUTE_RESOLUTION = "dispute_resolution"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    INSURANCE = "insurance"
    DATA_PROTECTION = "data_protection"
    DEFINITIONS = "definitions"
    ENTIRE_AGREEMENT = "entire_agreement"
    AMENDMENT = "amendment"
    SEVERABILITY = "severability"
    OTHER = "other"


class ReconResult(BaseModel):
    file: str
    file_format: str
    page_count: int | None
    char_count: int
    classification: str  # "native" | "scanned" | "mixed"
    pages_native: int
    pages_scanned: int
    likely_tables: bool


class RedactionRecord(BaseModel):
    file: str
    category: str  # "PHONE" | "EMAIL" | "IP" | "ADDRESS"
    original_snippet: str
    replacement: str
    page_number: int


class RedactedPage(BaseModel):
    page_number: int
    text: str


class RedactedDocument(BaseModel):
    doc_id: str
    file_path: str
    pages: list[RedactedPage]


class ClauseChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_file: str
    chunk_index: int
    clause_number: str | None
    section_title: str | None
    page: int
    text: str
    token_count: int


class ContractMetadata(BaseModel):
    """Core Metadata fields per the procurement team's metadata requirements
    doc (business/user-maintained tier) -- fields realistically extractable
    from contract text. Deliberately excludes that doc's "Contract Owner"
    (an internal person) -- `department` (the responsible department) is
    tracked instead. AI-analysis fields (risk scoring, clause-availability
    flags, etc.) and system/repository fields (file version, upload date,
    etc.) from that doc's other two tiers are out of scope here."""

    contract_number: str | None = Field(
        default=None, description="Official contract reference number, if stated"
    )
    agreement_title: str | None = Field(
        default=None,
        description="The agreement's own title as it appears on the document, if distinct "
        "from the general contract type",
    )
    contract_type: str | None = Field(
        default=None,
        description="Contract document type, e.g. MSA, SOW, NDA, Lease, Purchase Agreement",
    )
    agreement_category: str | None = Field(
        default=None,
        description="Broad business-function category the agreement falls under, e.g. "
        "Procurement, IT, HR, Finance, Legal -- if stated or clearly implied",
    )
    vendor: str | None = Field(
        default=None, description="Name of the vendor/counterparty organization"
    )
    project_name: str | None = Field(
        default=None, description="Name of the project this contract is associated with, if stated"
    )
    business_unit: str | None = Field(
        default=None,
        description="The internal TES / Trans World Enterprise Services / Transworld "
        "Associates (TWA) entity that is the contracting party on this agreement (e.g. "
        "'TES', 'TWA'), as stated in the document",
    )
    department: str | None = Field(
        default=None,
        description="The internal department responsible for administering, approving, or "
        "overseeing this contract -- e.g. a department named in an approval or authorization "
        "block, such as 'Procurement & Contracts', 'IT', 'HR', 'Engineering'. Do NOT return "
        "TES, Trans World Enterprise Services, or Transworld Associates itself -- that is the "
        "contracting party, not a department. Do NOT return a signatory's name or title from "
        "the execution/signature block -- that identifies who signed, not which department "
        "owns the contract. Leave null if no such department is stated.",
    )
    segment: str | None = Field(
        default=None,
        description="Procurement/business segment or category, e.g. Network Equipment, "
        "Software Licensing, Professional Services, Cloud Services -- if stated or clearly "
        "implied by the subject matter",
    )
    contract_description: str | None = Field(
        default=None, description="A brief one- or two-sentence description of the agreement"
    )
    status: str | None = Field(
        default=None,
        description="Contract lifecycle status, only if explicitly stated in the text (e.g. "
        "a DRAFT watermark, 'fully executed', termination/renewal language) -- do not infer "
        "from dates alone; leave null if not clearly stated",
    )
    execution_date: str | None = Field(
        default=None,
        description="The date the agreement was signed/executed, if distinct from its "
        "effective date. Use ISO 8601 (YYYY-MM-DD) if a specific date is stated, otherwise "
        "the contract's own wording.",
    )
    effective_date: str | None = Field(
        default=None,
        description="Contract effective/start date. Use ISO 8601 (YYYY-MM-DD) if a specific "
        "date is stated; otherwise use the contract's own wording.",
    )
    expiry_date: str | None = Field(
        default=None,
        description="Contract expiry/termination date. Use ISO 8601 (YYYY-MM-DD) if a specific "
        "date is stated; otherwise use the contract's own wording.",
    )
    renewal_date: str | None = Field(
        default=None, description="Next renewal due date, if explicitly stated"
    )
    contract_duration: str | None = Field(
        default=None,
        description="The agreement's stated validity period, in its own wording, e.g. '1 year'",
    )
    notice_period: str | None = Field(
        default=None, description="Notice period required for termination, e.g. '30 days'"
    )
    auto_renewal: bool | None = Field(
        default=None,
        description="Whether the contract renews automatically, only if explicitly stated -- "
        "leave null if not addressed in the text",
    )
    value: float | None = Field(
        default=None,
        description="Total contract value as a plain number, no currency symbol or commas",
    )
    currency: str | None = Field(default=None, description="ISO 4217 currency code, e.g. USD")
    payment_terms: str | None = Field(
        default=None, description="Payment terms as stated, e.g. 'Net 45', 'quarterly installments'"
    )
    payment_milestones: str | None = Field(
        default=None,
        description="Payment schedule/milestones as stated, if distinct from the general "
        "payment terms, e.g. '30% on mobilization, 70% on completion'",
    )
    signatory_names: list[str] = Field(
        default_factory=list,
        description="Full names of all signatories across both parties, as they appear in "
        "signature blocks or execution clauses",
    )
