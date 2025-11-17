"""
super7_resolver.py

Pipeline:
- Input: Super7Input (company name + optional hints)
- Web search (Tavily) to get candidate URLs
- Scrape each URL (HTML) with polite rules
- Extract focused snippets (name, address, phone, zip)
- Use LLM (OpenAI via LangChain) to:
  - Extract entities (with per-entity source_urls + confidence)
  - Compute page-level match scores
- Score & select best candidate per Super7 field
- Output JSON with:
  - primary_url, primary_confidence
  - candidates (URLs + scores)
  - super7_summary (value, source, confidence, all_sources per field)
"""

import os
import time
import json
import re
import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants / Config
# -----------------------------------------------------------------------------

# Domains we NEVER want to scrape at all (for HTML scraping).
SCRAPER_DOMAIN_BLACKLIST = {
    "www.dnb.com",
    "dnb.com",
}

# Domains we don't want to use as primary sources in Super7 summary
SUMMARY_DOMAIN_EXCLUDE = {
    # Data vendors / noisy aggregators
    "www.dnb.com",
    "dnb.com",
    "www.b2bhint.com",
    "b2bhint.com",
    # Social / user-generated
    "www.facebook.com",
    "facebook.com",
    "www.instagram.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "www.tiktok.com",
    "tiktok.com",
    # News / media
    "www.thetimes-tribune.com",
    "thetimes-tribune.com",
}

# File extensions we skip (we don't want PDFs/Office for this stage)
SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar",
}

# Maximum HTML text length per page (post-clean) we keep
MAX_HTML_CHARS = 50000

# Throttling boundaries (to be polite)
REQUEST_DELAY_MIN = 0.5
REQUEST_DELAY_MAX = 1.5

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

CORP_SUFFIXES = [
    "llc", "inc", "corp", "corporation", "ltd", "limited",
    "oy", "oyj", "sa", "gmbh", "plc", "lp", "llp", "bv",
    "srl", "sro", "pte", "sdn", "bhd", "ag", "nv"
]


def normalize_company_name(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    tokens = [t for t in s.split() if t]
    filtered = [t for t in tokens if t not in CORP_SUFFIXES]
    return " ".join(filtered)


def jaccard_name_similarity(a: str, b: str) -> float:
    na = set(normalize_company_name(a).split())
    nb = set(normalize_company_name(b).split())
    if not na or not nb:
        return 0.0
    inter = len(na & nb)
    union = len(na | nb)
    return inter / union


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def url_has_skip_extension(url: str) -> bool:
    path = urlparse(url).path.lower()
    for ext in SKIP_EXTENSIONS:
        if path.endswith(ext):
            return True
    return False


def should_consider_search_result(company_name: str, title: str, snippet: str) -> bool:
    """
    Decide if a Tavily search result looks relevant enough to scrape.
    """
    if not (title or snippet):
        return False

    # basic domain heuristics can be added here if you like

    sim = jaccard_name_similarity(company_name, title or "")
    if sim >= 0.2:
        return True

    # check company name (normalized) appears in snippet
    norm_name = normalize_company_name(company_name)
    if norm_name and snippet:
        if norm_name in snippet.lower():
            return True

    # fallback: raw name substring match
    if company_name and snippet and company_name.lower() in snippet.lower():
        return True

    return False


def doc_mentions_company(company_name: str, text: str, min_token_hits: int = 2) -> bool:
    """
    Quick filter: does the doc text look like it's about this company?
    """
    if not text:
        return False

    # raw name match
    if company_name.lower() in text.lower():
        return True

    norm = normalize_company_name(company_name)
    tokens = [t for t in norm.split() if t]
    if not tokens:
        return False

    # pick the longest token as main
    main = max(tokens, key=len)

    hits = text.lower().count(main.lower())
    return hits >= min_token_hits


def extract_snippets_for_company(
    text: str,
    company_name: str,
    max_snippets: int = 25,
    window_chars: int = 300,
) -> List[Dict[str, Any]]:
    """
    Extract small text windows where the company is mentioned + likely address/phone/zip lines.
    Returns a list of dicts:
      {
        "id": int,
        "type": "name_context" | "phone_context" | "zip_context" | "address_context" | "generic",
        "text": str
      }
    """
    snippets: List[Dict[str, Any]] = []
    if not text:
        return snippets

    lower_text = text.lower()
    norm_name = normalize_company_name(company_name)
    raw_name = company_name.lower()

    variants = set()
    if norm_name:
        variants.add(norm_name)
    if raw_name:
        variants.add(raw_name)
    # & vs and
    variants |= {v.replace("&", " and ") for v in variants}
    variants |= {v.replace(" and ", " & ") for v in variants}

    # 1) name-based windows
    used_ranges = []
    for v in variants:
        if not v.strip():
            continue
        start = 0
        while True:
            idx = lower_text.find(v, start)
            if idx == -1:
                break
            left = max(0, idx - window_chars)
            right = min(len(text), idx + len(v) + window_chars)
            candidate = text[left:right].strip()
            if candidate:
                snippets.append(
                    {
                        "id": len(snippets),
                        "type": "name_context",
                        "text": candidate,
                    }
                )
                used_ranges.append((left, right))
            start = idx + len(v)
            if len(snippets) >= max_snippets:
                break
        if len(snippets) >= max_snippets:
            break

    if len(snippets) >= max_snippets:
        return snippets[:max_snippets]

    # 2) regex-based patterns (phone, zip, address-like lines)
    phone_pattern = re.compile(
        r"(\+?\d[\d\-\(\)\s]{6,}\d)",
        re.MULTILINE,
    )
    zip_pattern = re.compile(r"\b\d{5}(?:-\d{4})?\b")
    address_keywords = [
        "street", "st.", "st ", "road", "rd.", "rd ",
        "avenue", "ave", "blvd", "lane", "ln", "drive", "dr", "way",
    ]

    # We'll work line-wise
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower_line = stripped.lower()

        # Phone context
        if phone_pattern.search(stripped):
            snippets.append(
                {
                    "id": len(snippets),
                    "type": "phone_context",
                    "text": stripped,
                }
            )

        # Zip context
        if zip_pattern.search(stripped):
            snippets.append(
                {
                    "id": len(snippets),
                    "type": "zip_context",
                    "text": stripped,
                }
            )

        # Address-like context
        if any(k in lower_line for k in address_keywords) and re.search(r"\d", stripped):
            snippets.append(
                {
                    "id": len(snippets),
                    "type": "address_context",
                    "text": stripped,
                }
            )

        if len(snippets) >= max_snippets:
            break

    if not snippets:
        # fallback: generic snippet
        snippet = text[:800].strip()
        if snippet:
            snippets.append(
                {
                    "id": 0,
                    "type": "generic",
                    "text": snippet,
                }
            )

    # deduplicate by text
    seen_text = set()
    unique_snippets = []
    for sn in snippets:
        if sn["text"] not in seen_text:
            seen_text.add(sn["text"])
            unique_snippets.append(sn)

    return unique_snippets[:max_snippets]


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

class Super7Input(BaseModel):
    company_name: str
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    street_address: Optional[str] = None
    zip: Optional[str] = None
    phone: Optional[str] = None


class ExtractedEntity(BaseModel):
    entity_type: str
    value: str
    source_urls: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None


class PageExtractionResult(BaseModel):
    url: str
    entities: List[ExtractedEntity] = Field(default_factory=list)
    match_score_name: float = 0.0
    match_score_address: float = 0.0
    match_score_phone: float = 0.0
    looks_like_official_site: bool = False
    overall_score: float = 0.0
    reason: str = ""


@dataclass
class CandidateRecord:
    url: str
    source_type: str
    extraction: PageExtractionResult
    first_seen_at: float = field(default_factory=time.time)
    last_checked_at: float = field(default_factory=time.time)


# -----------------------------------------------------------------------------
# Web search tool (Tavily)
# -----------------------------------------------------------------------------

class WebSearchTool:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self._tool = TavilySearch(max_results=max_results)

    def search_candidates(self, queries: List[str]) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            res = self._tool.invoke({"query": q})
            results = res.get("results", [])

            for r in results:
                url = r.get("url")
                title = r.get("title", "")
                snippet = r.get("content", "") or r.get("snippet", "")
                if not url:
                    continue
                if url not in seen:
                    seen[url] = {
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "source_type": "web_search",
                    }
        return list(seen.values())


# -----------------------------------------------------------------------------
# Scraper tool (HTML only, no PDFs/Office)
# -----------------------------------------------------------------------------

class ScraperTool:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Connection": "keep-alive",
            }
        )

    def _polite_delay(self):
        time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))

    def fetch_html(self, url: str) -> str:
        """
        Fetch HTML page and return cleaned text.
        No PDFs/Office – they are skipped.
        """
        domain = get_domain(url)
        if domain in SCRAPER_DOMAIN_BLACKLIST:
            logger.info(f"[SCRAPER] Domain blacklisted: {domain}, skipping {url}")
            return ""

        if url_has_skip_extension(url):
            logger.info(f"[SCRAPER] Skipping non-HTML extension: {url}")
            return ""

        self._polite_delay()

        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 403:
                logger.info(f"[SCRAPER] HTTP 403 for {url}, skipping.")
            else:
                logger.info(f"[SCRAPER] HTTP error {status} for {url}: {e}")
            return ""
        except Exception as e:
            logger.info(f"[SCRAPER] Failed {url}: {e}")
            return ""

        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            logger.info(f"[SCRAPER] Non-HTML Content-Type ({content_type}) for {url}, skipping.")
            return ""

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        if len(text) > MAX_HTML_CHARS:
            text = text[:MAX_HTML_CHARS]

        return text


# -----------------------------------------------------------------------------
# LLM extractor
# -----------------------------------------------------------------------------

class LLMExtractor:
    """
    Uses OpenAI Chat model via LangChain to:
    - Extract entities from snippets
    - Compute page-level match scores
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """Convert value to float, handling None and bad types gracefully."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def build_extraction_prompt(
        self,
        super7: Super7Input,
        url: str,
        snippets: List[Dict[str, Any]],
    ) -> str:
        s7_dict = super7.model_dump()
        snippets_text = "\n\n".join(
            f"[SNIPPET {sn['id']} - {sn['type']}]\n{sn['text']}"
            for sn in snippets
        )

        instructions = f"""
You are a precise data extraction assistant.

We are trying to extract the **Super7** identity fields for a company from web page snippets:

Super7 fields:
- company_name
- street_address
- city
- state
- country
- zip
- phone

You are given:
1) A target Super7Input (company_name is mandatory; others are optional hints).
2) The URL of a web page.
3) A set of focused text snippets from that page.

Tasks:
1. Decide if this page is about the SAME company as the target.
2. Extract entities related to the company's identity:
   - Use entity_type values exactly from this set when relevant:
     ["company_name", "street_address", "city", "state", "country", "zip", "phone",
      "email", "website", "social_link", "other_id", "other"]
   - For each entity, include:
     - value (string)
     - source_urls: list of URLs where this value is supported (at least include the page URL)
     - confidence: number between 0 and 1

3. Compute page-level scores (0.0 to 1.0):
   - match_score_name
   - match_score_address
   - match_score_phone
   - overall_score
   - looks_like_official_site: boolean
   - reason: short explanation

Be conservative:
- If the page is unrelated, set scores near 0 and return few/no entities.
- If unsure about a value, use a lower confidence.

Return **STRICT JSON** only, no extra commentary, with this shape:

{{
  "url": "<page URL>",
  "entities": [
    {{
      "entity_type": "company_name" | "street_address" | "city" | "state" | "country" | "zip" | "phone" |
                     "email" | "website" | "social_link" | "other_id" | "other",
      "value": "<string>",
      "source_urls": ["<url1>", "<url2>", ...],
      "confidence": <number between 0 and 1 or null>
    }}
  ],
  "match_score_name": <number between 0 and 1>,
  "match_score_address": <number between 0 and 1>,
  "match_score_phone": <number between 0 and 1>,
  "looks_like_official_site": <true or false>,
  "overall_score": <number between 0 and 1>,
  "reason": "<short string>"
}}

Super7Input (hints):

{json.dumps(s7_dict, indent=2)}

Page URL: {url}

Snippets:
{snippets_text}
"""
        return instructions

    def extract_from_snippets(
        self,
        super7: Super7Input,
        url: str,
        snippets: List[Dict[str, Any]],
    ) -> PageExtractionResult:
        """
        Run the LLM on snippets for a single page.
        """
        if not snippets:
            return PageExtractionResult(
                url=url,
                entities=[],
                match_score_name=0.0,
                match_score_address=0.0,
                match_score_phone=0.0,
                looks_like_official_site=False,
                overall_score=0.0,
                reason="No snippets extracted.",
            )

        prompt = self.build_extraction_prompt(super7, url, snippets)
        response = self.llm.invoke(prompt)
        text = response.content

        # Try to parse JSON
        try:
            data = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                except Exception:
                    data = {}
            else:
                data = {}

        if not isinstance(data, dict):
            data = {}

        url_out = data.get("url", url)

        entities_raw = data.get("entities", [])
        if not isinstance(entities_raw, list):
            entities_raw = []

        entities: List[ExtractedEntity] = []
        for e in entities_raw:
            if not isinstance(e, dict):
                continue
            srcs = e.get("source_urls") or [url_out]
            if not isinstance(srcs, list):
                srcs = [url_out]
            if url_out not in srcs:
                srcs.append(url_out)

            raw_conf = e.get("confidence")
            conf = None
            if raw_conf is not None:
                try:
                    conf = float(raw_conf)
                    # clamp to [0, 1]
                    conf = max(0.0, min(1.0, conf))
                except (TypeError, ValueError):
                    conf = None

            entities.append(
                ExtractedEntity(
                    entity_type=str(e.get("entity_type", "unknown")),
                    value=str(e.get("value") or ""),
                    source_urls=srcs,
                    confidence=conf,
                )
            )

        pe = PageExtractionResult(
            url=url_out,
            entities=entities,
            match_score_name=self._safe_float(data.get("match_score_name"), 0.0),
            match_score_address=self._safe_float(data.get("match_score_address"), 0.0),
            match_score_phone=self._safe_float(data.get("match_score_phone"), 0.0),
            looks_like_official_site=bool(data.get("looks_like_official_site", False)),
            overall_score=self._safe_float(data.get("overall_score"), 0.0),
            reason=str(data.get("reason", "")),
        )

        # clamp page scores as well
        pe.match_score_name = max(0.0, min(1.0, pe.match_score_name))
        pe.match_score_address = max(0.0, min(1.0, pe.match_score_address))
        pe.match_score_phone = max(0.0, min(1.0, pe.match_score_phone))
        pe.overall_score = max(0.0, min(1.0, pe.overall_score))

        return pe


# -----------------------------------------------------------------------------
# Same-company guard & scoring
# -----------------------------------------------------------------------------

def is_page_same_company(
    target_company_name: str,
    page_entities: List[ExtractedEntity],
    threshold: float = 0.6,
) -> bool:
    """
    Decide if this page is about the same company based on extracted company_name entities.
    """
    best_sim = 0.0
    for ent in page_entities:
        if ent.entity_type != "company_name":
            continue
        sim = jaccard_name_similarity(target_company_name, ent.value)
        if sim > best_sim:
            best_sim = sim
    return best_sim >= threshold


def score_field_candidate(
    s7: Super7Input,
    field: str,
    ent: ExtractedEntity,
    page: PageExtractionResult,
) -> float:
    """
    Compute a raw score for one candidate entity for one Super7 field.
    """
    base_conf = ent.confidence if ent.confidence is not None else 0.0
    score = base_conf

    # page relevance
    score += 0.5 * page.overall_score

    # official site bonus
    if page.looks_like_official_site:
        score += 0.2

    # hint-based bonus
    hint_value = getattr(s7, field, None)
    if hint_value and ent.value:
        hv = str(hint_value).lower()
        ev = ent.value.lower()
        if hv == ev:
            score += 0.3
        elif hv in ev or ev in hv:
            score += 0.15

    return score


# -----------------------------------------------------------------------------
# Super7 summarization
# -----------------------------------------------------------------------------

def summarize_super7_simple(
    s7: Super7Input,
    candidates: List[CandidateRecord],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    For each Super7 field, pick the best entity across all candidate pages.
    """
    fields = [
        "company_name",
        "street_address",
        "city",
        "state",
        "country",
        "zip",
        "phone",
    ]

    summary: Dict[str, Optional[Dict[str, Any]]] = {f: None for f in fields}

    for field in fields:
        best_score = -1.0
        best_ent: Optional[ExtractedEntity] = None
        best_sources: List[str] = []
        best_page: Optional[PageExtractionResult] = None

        for cand in candidates:
            page = cand.extraction
            page_domain = get_domain(page.url)

            # For non-name fields, enforce same-company guard
            if field != "company_name":
                if not is_page_same_company(s7.company_name, page.entities):
                    continue

            for ent in page.entities:
                if ent.entity_type != field:
                    continue
                if not ent.value:
                    continue

                # filter out entities where ALL sources are excluded domains
                all_srcs = ent.source_urls or [page.url]
                if all(
                    get_domain(src) in SUMMARY_DOMAIN_EXCLUDE
                    for src in all_srcs
                ):
                    continue

                raw_score = score_field_candidate(s7, field, ent, page)
                if raw_score > best_score:
                    best_score = raw_score
                    best_ent = ent
                    best_sources = list(set(all_srcs))
                    best_page = page

        if best_ent is not None and best_score >= 0.3:
            # Normalize raw_score ~ [0,2] → [0,1]
            conf = min(max(best_score / 2.0, 0.0), 1.0)
            # pick a primary source not excluded if possible
            primary_source = None
            for src in best_sources:
                if get_domain(src) not in SUMMARY_DOMAIN_EXCLUDE:
                    primary_source = src
                    break
            if primary_source is None and best_sources:
                primary_source = best_sources[0]

            summary[field] = {
                "value": best_ent.value,
                "source": primary_source,
                "confidence": conf,
                "all_sources": best_sources,
            }
        else:
            summary[field] = None

    return summary


# -----------------------------------------------------------------------------
# Resolver Orchestrator
# -----------------------------------------------------------------------------

class Super7Resolver:
    def __init__(
        self,
        search_tool: WebSearchTool,
        scraper: ScraperTool,
        extractor: LLMExtractor,
    ):
        self.search_tool = search_tool
        self.scraper = scraper
        self.extractor = extractor

    def build_queries(self, s7: Super7Input) -> List[str]:
        name = s7.company_name.strip()
        parts = [name]
        if s7.city:
            parts.append(s7.city)
        if s7.state:
            parts.append(s7.state)
        if s7.country:
            parts.append(s7.country)

        base = " ".join(parts)

        queries = [
            f"{base} official website",
            f"{base} company",
            f"\"{name}\"",
        ]

        if s7.phone:
            queries.append(f"\"{name}\" \"{s7.phone}\"")

        return queries

    def process_company(self, s7: Super7Input) -> Dict[str, Any]:
        """
        Full pipeline for one company.
        """
        queries = self.build_queries(s7)
        search_results = self.search_tool.search_candidates(queries)

        candidate_records: List[CandidateRecord] = []

        primary_url: Optional[str] = None
        primary_conf: float = 0.0

        for sr in search_results:
            url = sr["url"]
            title = sr.get("title", "")
            snippet = sr.get("snippet", "")

            if not should_consider_search_result(s7.company_name, title, snippet):
                continue

            # fetch HTML
            text = self.scraper.fetch_html(url)
            if not text:
                # fall back to Tavily snippet if present
                if snippet:
                    text = snippet
                else:
                    continue

            if not doc_mentions_company(s7.company_name, text):
                continue

            snippets = extract_snippets_for_company(text, s7.company_name)
            if not snippets:
                continue

            extraction = self.extractor.extract_from_snippets(s7, url, snippets)

            candidate_records.append(
                CandidateRecord(
                    url=url,
                    source_type=sr.get("source_type", "web_search"),
                    extraction=extraction,
                )
            )

            if extraction.overall_score > primary_conf:
                primary_conf = extraction.overall_score
                primary_url = url

        # Summarize Super7 fields
        super7_summary = summarize_super7_simple(s7, candidate_records)

        # compress candidate info for output
        candidates_out = [
            {
                "url": c.url,
                "overall_score": c.extraction.overall_score,
                "reason": c.extraction.reason,
            }
            for c in sorted(
                candidate_records,
                key=lambda x: x.extraction.overall_score,
                reverse=True,
            )
        ]

        return {
            "company_id": normalize_company_name(s7.company_name),
            "input": s7.model_dump(),
            "primary_url": primary_url,
            "primary_confidence": primary_conf,
            "candidates": candidates_out,
            "super7_summary": super7_summary,
        }


# -----------------------------------------------------------------------------
# Batch interface
# -----------------------------------------------------------------------------

def resolve_super7_batch(super7_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    High-level function you call from your notebook / app.

    Example:
        batch_input = [
            {"company_name": "Home Fit Solutions LLC", "country": "United States", "city": "Honesdale"},
            {"company_name": "r&k firesupport llc"},
        ]
        res = resolve_super7_batch(batch_input)
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY in your environment.")
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not set; TavilySearch will fail.")

    search = WebSearchTool(max_results=5)
    scraper = ScraperTool(timeout=10)
    extractor = LLMExtractor(model_name="gpt-4o-mini", temperature=0.0)
    resolver = Super7Resolver(search, scraper, extractor)

    results = []
    for payload in super7_payloads:
        s7 = Super7Input(**payload)
        out = resolver.process_company(s7)
        results.append(out)

    return {"results": results}

