"""
rag_engine.py - Vector Database (ChromaDB), Semantic Retrieval & Corporate Breakthrough Extractor
=================================================================================================
Stores SEBI filings, earnings transcripts, and corporate disclosures in ChromaDB.
Provides semantic retrieval with document chunk attribution and source citations.
Includes a Corporate Breakthrough Extractor for capacity expansions, tech innovations, and plant additions.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_FN,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ISIN_MAP,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample SEBI-compliant corporate disclosure documents (seed data)
# ---------------------------------------------------------------------------
SEED_DOCUMENTS: List[Dict[str, str]] = [
    {
        "id": "rel_earnings_q3_2025",
        "ticker": "RELIANCE.NS",
        "isin": "INE002A01018",
        "source": "Q3 FY2025 Earnings Transcript",
        "category": "earnings_transcript",
        "text": (
            "Reliance Industries reported consolidated revenue of Rs 2,35,481 crore for Q3 FY2025, "
            "a growth of 7.3% YoY. Net profit stood at Rs 18,540 crore. Jio Platforms saw 8.2 crore "
            "subscriber additions. Retail segment revenue grew 18% driven by store expansion. "
            "New Energy division committed Rs 75,000 crore capex for giga-factories. "
            "Company announced expansion of Jamnagar refinery complex with a $10 billion green energy hub."
        ),
    },
    {
        "id": "hdfc_bank_annual_2025",
        "ticker": "HDFCBANK.NS",
        "isin": "INE040A01012",
        "source": "Annual Report FY2025",
        "category": "annual_report",
        "text": (
            "HDFC Bank's total deposits grew 15.6% YoY to Rs 24.12 lakh crore. Net Interest Margin "
            "improved to 3.46% from 3.31%. Gross NPAs reduced to 1.24% from 1.33%. The bank added "
            "1,200 new branches during the year, bringing total branches to 9,180. Digital transactions "
            "now constitute 73% of total retail transactions. Credit growth stood at 16.2%."
        ),
    },
    {
        "id": "infy_earnings_q3_2025",
        "ticker": "INFY.NS",
        "isin": "INE009A01021",
        "source": "Q3 FY2025 Earnings Call Transcript",
        "category": "earnings_transcript",
        "text": (
            "Infosys reported Q3 revenue of Rs 40,794 crore, up 6.1% YoY in constant currency. "
            "Operating margin expanded to 21.3%. Large deal TCV was $8.8 billion for the quarter. "
            "Company raised FY25 guidance to 4.5-5% revenue growth. AI-first strategy with Topaz "
            "platform seeing 200+ client engagements. Attrition reduced to 12.3%. New delivery center "
            "inaugurated in Hyderabad with capacity for 15,000 employees."
        ),
    },
    {
        "id": "tcs_annual_2025",
        "ticker": "TCS.NS",
        "isin": "INE467B01014",
        "source": "Annual Report FY2025",
        "category": "annual_report",
        "text": (
            "TCS delivered revenue of Rs 2,49,647 crore in FY25, growth of 5.8%. Operating margin "
            "at 26.8%. Company announced Rs 5,000 crore investment in AI and cloud infrastructure. "
            "New semiconductor design facility established in Noida. Workforce expanded to 6,14,000 "
            "employees globally. Bagged 12 deals each exceeding $100 million. Board recommended "
            "final dividend of Rs 30 per share."
        ),
    },
    {
        "id": "lt_capex_expansion",
        "ticker": "LT.NS",
        "isin": "INE154A01026",
        "source": "Investor Presentation Q3 FY2025",
        "category": "investor_presentation",
        "text": (
            "Larsen & Toubro's order book surged to Rs 5,06,300 crore, a 19% increase. "
            "Infrastructure segment secured major orders including the Mumbai-Ahmedabad high-speed rail "
            "package. Company inaugurated a new heavy fabrication facility in Odisha with 50,000 TPA "
            "capacity. International order inflow grew 24%. Water & effluent treatment vertical "
            "won three large smart city projects."
        ),
    },
    {
        "id": "sebi_circular_market_integrity",
        "ticker": "MARKET",
        "isin": "N/A",
        "source": "SEBI Circular SEBI/HO/ISD/IMD/ISC/P/CIR/2025/018",
        "category": "regulatory",
        "text": (
            "SEBI has tightened surveillance measures for stocks showing abnormal price movement. "
            "Brokers must report bulk/block deals within 15 minutes. Enhanced monitoring of "
            "shell companies and penny stocks. IPO proceeds must be utilized within 12 months. "
            "Stricter norms for related party transactions. All investment advisers must register "
            "under SEBI (Investment Advisers) Regulations, 2013."
        ),
    },
    {
        "id": "sebi_fraud_prevention",
        "ticker": "MARKET",
        "isin": "N/A",
        "source": "SEBI Prohibition of Fraudulent and Unfair Trade Practices Regulations 2024 Amendment",
        "category": "regulatory",
        "text": (
            "SEBI prohibited front-running, layering, and spoofing in securities markets. "
            "Social media influencers giving stock tips must register as investment advisers. "
            "Pump-and-dump schemes attract penalties up to Rs 1 crore or three times profits. "
            "Whistleblower mechanism strengthened with 10% of disgorgement amount as reward."
        ),
    },
    {
        "id": "itc_consolidation",
        "ticker": "ITC.NS",
        "isin": "INE023A01015",
        "source": "Corporate Announcement - Board Meeting Outcome",
        "category": "board_meeting",
        "text": (
            "ITC Board approved demerger of Hotels business into ITC Hotels Ltd. "
            "FMCG segment revenue crossed Rs 20,000 crore milestone. Agri business expanded "
            "leaf tobacco procurement capacity by 15%. New paperboard mill commissioned in "
            "Bhadrachalam with 2,00,000 TPA capacity. Company declared interim dividend of Rs 6.25 per share."
        ),
    },
    {
        "id": "sbi_q3_results",
        "ticker": "SBIN.NS",
        "isin": "INE528G01035",
        "source": "Q3 FY2025 Press Release",
        "category": "earnings_transcript",
        "text": (
            "State Bank of India reported net profit of Rs 18,331 crore for Q3 FY25, up 24% YoY. "
            "Net Interest Margin at 3.45%. Gross NPA ratio improved to 2.07% from 2.43%. "
            "Deposit growth of 11.8% with CASA ratio at 41%. Digital platform YONO crossed "
            "60 million registered users. Global credit card business expanded with 5 lakh new cards."
        ),
    },
    {
        "id": "maruti_capacity_addition",
        "ticker": "MARUTI.NS",
        "isin": "INE192A01025",
        "source": "Annual Report FY2025",
        "category": "annual_report",
        "text": (
            "Maruti Suzuki's total installed capacity reached 22.5 lakh units annually with the "
            "completion of Phase 2 expansion at Hansalpur, Gujarat. Company invested Rs 4,500 crore "
            "in R&D for electric and hybrid vehicles. First EV model eVX launched with 500 km range. "
            "Export volume grew 12% to 2.8 lakh units. Operating profit margin expanded to 13.2%."
        ),
    },
]


# ---------------------------------------------------------------------------
# Document Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks for vector storage."""
    words = text.split()
    if len(words) <= chunk_size // 5:  # roughly char-based
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size // 5, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - (overlap // 5)
        if start <= 0 and len(chunks) > 0:
            break
    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# RAG Engine Class
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Single retrieval result with chunk attribution."""
    chunk_text: str
    source: str
    category: str
    ticker: str
    isin: str
    document_id: str
    relevance_score: float
    chunk_index: int


class RAGEngine:
    """
    Vector-database backed retrieval engine for SEBI filings and corporate disclosures.
    Uses ChromaDB for persistent vector storage and cosine similarity retrieval.
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        """Initialize ChromaDB client and collection; seed if empty."""
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": CHROMA_DISTANCE_FN},
        )
        self._seeded = False
        if self.collection.count() == 0:
            self._seed_documents()

    def _seed_documents(self) -> None:
        """Populate the collection with seed SEBI filings and corporate disclosures."""
        all_ids: List[str] = []
        all_texts: List[str] = []
        all_metas: List[Dict[str, Any]] = []

        for doc in SEED_DOCUMENTS:
            chunks = _chunk_text(doc["text"])
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_chunk_{idx}"
                all_ids.append(chunk_id)
                all_texts.append(chunk)
                all_metas.append({
                    "source": doc["source"],
                    "category": doc["category"],
                    "ticker": doc["ticker"],
                    "isin": doc["isin"],
                    "document_id": doc["id"],
                    "chunk_index": idx,
                })

        if all_ids:
            batch_size = 100
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i + batch_size]
                batch_texts = all_texts[i:i + batch_size]
                batch_metas = all_metas[i:i + batch_size]
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metas,
                )
            logger.info("Seeded %d chunks into ChromaDB collection '%s'",
                        len(all_ids), CHROMA_COLLECTION_NAME)
            self._seeded = True

    def add_document(
        self,
        doc_id: str,
        text: str,
        source: str,
        category: str,
        ticker: str = "MARKET",
        isin: str = "N/A",
    ) -> int:
        """Add a new document to the vector store. Returns number of chunks added."""
        chunks = _chunk_text(text)
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metas = [
            {
                "source": source,
                "category": category,
                "ticker": ticker,
                "isin": isin,
                "document_id": doc_id,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        self.collection.add(ids=ids, documents=chunks, metadatas=metas)
        return len(chunks)

    def retrieve(
        self,
        query: str,
        ticker_filter: Optional[str] = None,
        n_results: int = 5,
        category_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Semantic retrieval with optional ticker and category filters.
        Returns ranked list of RetrievalResult with full source attribution.
        """
        where_clause: Optional[Dict[str, Any]] = None
        conditions = []
        if ticker_filter:
            conditions.append({"ticker": ticker_filter})
        if category_filter:
            conditions.append({"category": category_filter})

        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        try:
            query_params: Dict[str, Any] = {
                "query_texts": [query],
                "n_results": min(n_results, self.collection.count() or 1),
            }
            if where_clause:
                query_params["where"] = where_clause

            results = self.collection.query(**query_params)
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        retrieval_results: List[RetrievalResult] = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc_text, meta, dist in zip(docs, metas, distances):
                score = max(0, min(1, 1 - dist)) if dist <= 1 else 0.0
                retrieval_results.append(RetrievalResult(
                    chunk_text=doc_text,
                    source=meta.get("source", "Unknown"),
                    category=meta.get("category", "Unknown"),
                    ticker=meta.get("ticker", "N/A"),
                    isin=meta.get("isin", "N/A"),
                    document_id=meta.get("document_id", "Unknown"),
                    relevance_score=round(score, 4),
                    chunk_index=meta.get("chunk_index", 0),
                ))
        return retrieval_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return collection statistics for monitoring."""
        count = self.collection.count()
        all_meta = self.collection.get()["metadatas"] if count > 0 else []
        categories = {}
        tickers = set()
        for m in all_meta:
            cat = m.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            tickers.add(m.get("ticker", "N/A"))
        return {
            "total_chunks": count,
            "categories": categories,
            "unique_tickers": list(tickers),
        }


# ---------------------------------------------------------------------------
# Corporate Breakthrough Extractor
# ---------------------------------------------------------------------------

BREAKTHROUGH_KEYWORDS: List[str] = [
    "capacity expansion", "new plant", "greenfield", "brownfield",
    "commissioned", "inaugurated", "expanded", "expansion",
    "technology", "innovation", "patent", "breakthrough",
    "giga-factory", "gigafactory", "new facility",
    "installed capacity", "capex", "investment",
    "r&d", "research", "new product launch",
    "joint venture", "strategic partnership", "acquisition",
]

BREAKTHROUGH_PATTERNS: Dict[str, List[str]] = {
    "capacity_expansion": [
        "capacity expansion", "expanded capacity", "new capacity",
        "installed capacity", "production capacity", "TPA",
    ],
    "technology_innovation": [
        "technology", "innovation", "patent", "AI", "digital",
        "R&D", "research", "breakthrough",
    ],
    "plant_addition": [
        "new plant", "new facility", "greenfield", "commissioned",
        "inaugurated", "facility", "giga-factory",
    ],
    "strategic_investment": [
        "capex", "investment", "joint venture", "partnership",
        "acquisition", "strategic",
    ],
}


@dataclass
class BreakthroughEvent:
    """A single corporate breakthrough event."""
    event_type: str
    description: str
    source: str
    ticker: str
    relevance_score: float


class CorporateBreakthroughExtractor:
    """
    Extracts recent corporate breakthroughs (capacity expansions, tech innovations,
    plant additions) from the RAG knowledge base using keyword pattern matching.
    """

    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine

    def extract(
        self,
        ticker: str,
        n_results: int = 10,
    ) -> List[BreakthroughEvent]:
        """
        Retrieve and extract corporate breakthrough events for a given ticker.
        Searches across all document categories.
        """
        all_events: List[BreakthroughEvent] = []

        # Query with breakthrough-specific terms
        queries = [
            "capacity expansion technology innovation plant addition",
            "new facility commissioned investment capex",
            "acquisition partnership strategic expansion",
        ]

        seen_chunks: set = set()

        for query in queries:
            results = self.rag.retrieve(
                query=query,
                ticker_filter=ticker,
                n_results=n_results,
            )
            for r in results:
                if r.document_id in seen_chunks:
                    continue
                seen_chunks.add(r.document_id)
                text_lower = r.chunk_text.lower()
                for event_type, patterns in BREAKTHROUGH_PATTERNS.items():
                    for pattern in patterns:
                        if pattern.lower() in text_lower:
                            events = self._extract_event_sentences(
                                r.chunk_text, event_type, r.source, r.ticker
                            )
                            all_events.extend(events)
                            break

        # Deduplicate by description
        unique: List[BreakthroughEvent] = []
        seen_descs: set = set()
        for ev in all_events:
            short_desc = ev.description[:100]
            if short_desc not in seen_descs:
                seen_descs.add(short_desc)
                unique.append(ev)

        unique.sort(key=lambda x: x.relevance_score, reverse=True)
        return unique[:20]

    def _extract_event_sentences(
        self, text: str, event_type: str, source: str, ticker: str,
    ) -> List[BreakthroughEvent]:
        """Extract sentences from a text chunk that mention breakthroughs."""
        sentences = text.split(". ")
        events: List[BreakthroughEvent] = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for pattern in BREAKTHROUGH_PATTERNS.get(event_type, []):
                if pattern.lower() in sentence_lower:
                    # Score based on keyword density
                    keyword_count = sum(
                        1 for kw in BREAKTHROUGH_KEYWORDS
                        if kw.lower() in sentence_lower
                    )
                    score = min(0.5 + keyword_count * 0.15, 0.95)
                    events.append(BreakthroughEvent(
                        event_type=event_type,
                        description=sentence.strip(),
                        source=source,
                        ticker=ticker,
                        relevance_score=round(score, 2),
                    ))
                    break
        return events


# ---------------------------------------------------------------------------
# Convenience Singleton
# ---------------------------------------------------------------------------

_rag_instance: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get or create the singleton RAG engine instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGEngine()
    return _rag_instance


def get_breakthrough_extractor() -> CorporateBreakthroughExtractor:
    """Get or create the Corporate Breakthrough Extractor."""
    return CorporateBreakthroughExtractor(get_rag_engine())
