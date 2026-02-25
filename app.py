import re
import json
import time
import math
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

import requests
import feedparser
from dateutil import parser as dateparser

import streamlit as st
import pandas as pd

# ----------------------------
# Optional embeddings
# ----------------------------
_EMBEDDINGS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBEDDINGS_AVAILABLE = True
except Exception:
    _EMBEDDINGS_AVAILABLE = False


# ----------------------------
# Models
# ----------------------------
@dataclass
class Lesson:
    title: str
    learning_objectives: List[str]
    include_terms: List[str]
    exclude_terms: List[str]
    max_age_days: int
    max_items_total: int
    max_items_per_lo: int
    sources: Dict[str, str]
    use_embeddings: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    source_weights: Optional[Dict[str, float]] = None


@dataclass
class NormalizedItem:
    source: str
    title: str
    link: str
    summary: str
    published_dt: Optional[datetime]


@dataclass
class ScoredItem:
    source: str
    title: str
    link: str
    summary: str
    published: Optional[str]
    score: float
    matched_los: List[int]


# ----------------------------
# Text utils
# ----------------------------
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']{1,}")
STOPWORDS = {
    "the","and","or","of","to","in","for","on","with","as","by","an","a","is","are","be",
    "this","that","from","at","it","its","into","within","their","they","we","you","your",
    "will","should","can","may","must","lesson","learning","objective","objectives"
}

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").replace("\n", " ").strip()

def tokenize(text: str) -> List[str]:
    tokens = [m.group(0).lower() for m in WORD_RE.finditer(text or "")]
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def parse_published(entry) -> Optional[datetime]:
    if getattr(entry, "published_parsed", None):
        try:
            return datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
        except Exception:
            pass
    for field in ("published", "updated", "created"):
        v = getattr(entry, field, None)
        if v:
            try:
                dt = dateparser.parse(v)
                if not dt.tzinfo:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                continue
    return None

def safe_summary(entry) -> str:
    if getattr(entry, "summary", None):
        return str(entry.summary)
    if getattr(entry, "description", None):
        return str(entry.description)
    if getattr(entry, "content", None) and isinstance(entry.content, list) and entry.content:
        if isinstance(entry.content[0], dict) and "value" in entry.content[0]:
            return str(entry.content[0]["value"])
    return ""

def recency_component(published_dt: Optional[datetime], max_age_days: int) -> float:
    if not published_dt:
        return 0.0
    age_days = (now_utc() - published_dt).total_seconds() / 86400.0
    if age_days < 0:
        age_days = 0
    if age_days > max_age_days:
        return -2.0
    return max(0.0, 1.0 - (age_days / max_age_days))

def contains_excluded(text: str, exclude_terms: List[str]) -> bool:
    t = (text or "").lower()
    return any(term.lower() in t for term in exclude_terms)

def must_include_ok(text: str, include_terms: List[str]) -> bool:
    if not include_terms:
        return True
    t = (text or "").lower()
    return any(term.lower() in t for term in include_terms)

def canonical_dedup_key(title: str, link: str) -> str:
    link2 = (link or "").strip()
    link2 = re.sub(r"(\?|&)utm_[^=&]+=[^&]+", "", link2)
    link2 = re.sub(r"[?&]$", "", link2)
    return sha1(link2 or (title or "").strip().lower())


# ----------------------------
# RSS collection + normalisation
# ----------------------------
class RssCollector:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def fetch(self, url: str) -> feedparser.FeedParserDict:
        headers = {"User-Agent": "rss-lo-curator/1.0"}
        r = requests.get(url, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return feedparser.parse(r.content)

    def collect(self, sources: Dict[str, str]) -> List[Tuple[str, object]]:
        entries = []
        for name, url in sources.items():
            try:
                feed = self.fetch(url)
                for entry in feed.entries:
                    entries.append((name, entry))
            except Exception as e:
                # UI-friendly: keep going
                st.warning(f"Feed failed: {name} ({e})")
        return entries


class Normalizer:
    def normalize(self, source: str, entry: object) -> NormalizedItem:
        title = str(getattr(entry, "title", "")).strip()
        link = str(getattr(entry, "link", "")).strip()
        summary = strip_html(safe_summary(entry))
        published_dt = parse_published(entry)
        return NormalizedItem(
            source=source,
            title=title,
            link=link,
            summary=summary,
            published_dt=published_dt
        )


# ----------------------------
# Scoring
# ----------------------------
def lexical_overlap_score(text: str, lo_text: str, include_terms: List[str]) -> float:
    text_tokens = set(tokenize(text))
    q_tokens = tokenize(lo_text) + [t.lower() for t in include_terms]
    q_tokens = list(dict.fromkeys(q_tokens))
    if not q_tokens:
        return 0.0
    hits = sum(1 for qt in q_tokens if qt in text_tokens)
    return math.log1p(hits) / math.log1p(len(set(q_tokens)) + 1e-9)


class SemanticRanker:
    def __init__(self, model_name: str):
        if not _EMBEDDINGS_AVAILABLE:
            raise RuntimeError("Embeddings unavailable.")
        self.model = SentenceTransformer(model_name)

    def score_items(self, items: List[NormalizedItem], learning_objectives: List[str]):
        lo_vecs = self.model.encode(learning_objectives, normalize_embeddings=True)
        texts = [f"{it.title}. {it.summary}" for it in items]
        item_vecs = self.model.encode(texts, normalize_embeddings=True)
        sims = (item_vecs @ lo_vecs.T)
        best = sims.argmax(axis=1).tolist()
        return sims.tolist(), best


# ----------------------------
# Pipeline
# ----------------------------
class CuratorPipeline:
    def __init__(self, lesson: Lesson):
        self.lesson = lesson
        self.collector = RssCollector()
        self.normalizer = Normalizer()
        self.semantic_ranker = None
        if lesson.use_embeddings and _EMBEDDINGS_AVAILABLE:
            try:
                self.semantic_ranker = SemanticRanker(lesson.embedding_model)
            except Exception:
                self.semantic_ranker = None

    def filter_and_dedupe(self, items: List[NormalizedItem]) -> List[NormalizedItem]:
        cutoff = now_utc() - timedelta(days=self.lesson.max_age_days)
        seen = set()
        kept: List[NormalizedItem] = []
        for it in items:
            blob = f"{it.title}\n{it.summary}\n{it.link}"
            if contains_excluded(blob, self.lesson.exclude_terms):
                continue
            if not must_include_ok(blob, self.lesson.include_terms):
                continue
            if it.published_dt and it.published_dt < cutoff:
                continue

            key = canonical_dedup_key(it.title, it.link)
            if key in seen:
                continue
            seen.add(key)
            kept.append(it)
        return kept

    def rank(self, items: List[NormalizedItem]) -> List[ScoredItem]:
        weights = self.lesson.source_weights or {}

        scored: List[ScoredItem] = []

        if self.semantic_ranker:
            sims, _best = self.semantic_ranker.score_items(items, self.lesson.learning_objectives)
            for it, sim_row in zip(items, sims):
                matched = [i for i, s in enumerate(sim_row) if s >= 0.25]
                if not matched:
                    continue
                rel = max(sim_row[i] for i in matched)
                rec = recency_component(it.published_dt, self.lesson.max_age_days)
                src_w = weights.get(it.source, 1.0)
                score = (1.9 * rel) + (0.6 * rec) + (0.15 * (src_w - 1.0))

                scored.append(ScoredItem(
                    source=it.source,
                    title=it.title,
                    link=it.link,
                    summary=it.summary[:700],
                    published=(it.published_dt.isoformat() if it.published_dt else None),
                    score=round(score, 4),
                    matched_los=matched
                ))
        else:
            for it in items:
                text = f"{it.title}. {it.summary}"
                per_lo = [lexical_overlap_score(text, lo, self.lesson.include_terms)
                          for lo in self.lesson.learning_objectives]
                matched = [i for i, s in enumerate(per_lo) if s > 0.0]
                if not matched:
                    continue
                rel = max(per_lo[i] for i in matched)
                rec = recency_component(it.published_dt, self.lesson.max_age_days)
                src_w = weights.get(it.source, 1.0)
                score = (1.7 * rel) + (0.6 * rec) + (0.15 * (src_w - 1.0))

                scored.append(ScoredItem(
                    source=it.source,
                    title=it.title,
                    link=it.link,
                    summary=it.summary[:700],
                    published=(it.published_dt.isoformat() if it.published_dt else None),
                    score=round(score, 4),
                    matched_los=matched
                ))

        def sort_key(si: ScoredItem):
            ts = 0
            if si.published:
                try:
                    ts = dateparser.parse(si.published).timestamp()
                except Exception:
                    ts = 0
            return (si.score, ts)

        scored.sort(key=sort_key, reverse=True)
        return scored

    def enforce_limits(self, scored: List[ScoredItem]) -> List[ScoredItem]:
        per_lo_counts = [0] * len(self.lesson.learning_objectives)
        chosen: List[ScoredItem] = []

        for it in scored:
            best_lo = None
            best_need = -1
            for lo_idx in it.matched_los:
                need = self.lesson.max_items_per_lo - per_lo_counts[lo_idx]
                if need > best_need:
                    best_need = need
                    best_lo = lo_idx
            if best_lo is None or best_need <= 0:
                continue

            chosen.append(it)
            per_lo_counts[best_lo] += 1
            if len(chosen) >= self.lesson.max_items_total:
                break

        return chosen

    def run(self) -> List[ScoredItem]:
        raw_entries = self.collector.collect(self.lesson.sources)
        normalized = [self.normalizer.normalize(src, ent) for src, ent in raw_entries]
        filtered = self.filter_and_dedupe(normalized)
        ranked = self.rank(filtered)
        final = self.enforce_limits(ranked)
        return final


# ----------------------------
# Streamlit UI
# ----------------------------
DEFAULT_SOURCES = {
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Reuters Top": "https://feeds.reuters.com/reuters/topNews",
    "UK Parliament": "https://www.parliament.uk/business/news/rss/",
    "RAND": "https://www.rand.org/rss.xml",
    "NATO Review": "https://www.nato.int/cps/en/natohq/rss/news_371.xml",
}

DEFAULT_LOS = [
    "Explain how disinformation campaigns influence public perception during urban operations.",
    "Analyse indicators and warnings of coordinated online influence activity.",
    "Identify practical countermeasures and communications approaches for unit-level leaders.",
]

def parse_los(multiline: str) -> List[str]:
    los = []
    for line in multiline.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("-•").strip()
        los.append(line)
    return los

st.set_page_config(page_title="LO → RSS Curator", layout="wide")
st.title("LO → RSS Teaching Pack (Browser-only)")

left, right = st.columns([1, 1], gap="large")

with left:
    title = st.text_input("Lesson title", value="Urban Ops: Information Environment")
    los_text = st.text_area("Learning objectives (one per line)", value="\n".join(DEFAULT_LOS), height=160)

    include_terms = st.text_input("Include terms (comma-separated, optional)",
                                 value="urban, information, influence, disinformation")
    exclude_terms = st.text_input("Exclude terms (comma-separated, optional)",
                                 value="sports, celebrity, crypto, coupon, horoscope")

    max_age_days = st.slider("Max age (days)", 1, 90, 21)
    max_items_total = st.slider("Max items (total)", 5, 60, 18)
    max_items_per_lo = st.slider("Max items per LO", 1, 20, 6)

    use_embeddings = st.toggle("Use embeddings (semantic matching)", value=True)
    embedding_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2",
                                    disabled=not use_embeddings)

with right:
    st.subheader("RSS sources")

    if "sources_df" not in st.session_state:
        st.session_state.sources_df = pd.DataFrame(
            [{"name": k, "url": v} for k, v in DEFAULT_SOURCES.items()]
        )

    edited = st.data_editor(
        st.session_state.sources_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Source name", required=True),
            "url": st.column_config.TextColumn("RSS URL", required=True),
        },
    )
    st.session_state.sources_df = edited

    sources = {}
    for _, row in edited.iterrows():
        name = str(row.get("name", "")).strip()
        url = str(row.get("url", "")).strip()
        if name and url:
            sources[name] = url

    run = st.button("Generate teaching pack", type="primary", use_container_width=True)

    if run:
        los = parse_los(los_text)
        if not los:
            st.error("Please enter at least one learning objective.")
            st.stop()
        if not sources:
            st.error("Please provide at least one RSS source.")
            st.stop()

        lesson = Lesson(
            title=title.strip() or "Untitled lesson",
            learning_objectives=los,
            include_terms=[t.strip() for t in include_terms.split(",") if t.strip()],
            exclude_terms=[t.strip() for t in exclude_terms.split(",") if t.strip()],
            max_age_days=int(max_age_days),
            max_items_total=int(max_items_total),
            max_items_per_lo=int(max_items_per_lo),
            sources=sources,
            use_embeddings=bool(use_embeddings),
            embedding_model=(embedding_model.strip() if use_embeddings else "sentence-transformers/all-MiniLM-L6-v2"),
        )

        with st.spinner("Fetching RSS feeds and ranking items..."):
            pipeline = CuratorPipeline(lesson)
            items = pipeline.run()

        st.success(f"Generated {len(items)} curated items.")
        df = pd.DataFrame([{
            "score": it.score,
            "published": it.published,
            "source": it.source,
            "title": it.title,
            "link": it.link,
            "matched_los": ", ".join(str(i+1) for i in it.matched_los),
        } for it in items])

        st.dataframe(df, use_container_width=True, hide_index=True)

        json_bytes = json.dumps([asdict(i) for i in items], ensure_ascii=False, indent=2).encode("utf-8")

        md_lines = []
        md_lines.append(f"# Teaching content pack: {lesson.title}")
        md_lines.append("")
        md_lines.append(f"Generated: {now_utc().isoformat()}")
        md_lines.append(f"Embeddings: {'ON' if lesson.use_embeddings and _EMBEDDINGS_AVAILABLE else 'OFF (lexical fallback)'}")
        md_lines.append("")
        md_lines.append("## Learning objectives")
        for i, lo in enumerate(lesson.learning_objectives, start=1):
            md_lines.append(f"{i}. {lo}")
        md_lines.append("")
        md_lines.append("## Curated items")
        for it in items:
            los_str = ", ".join(str(i+1) for i in it.matched_los)
            md_lines.append(
                f"- **{it.title}** ({it.source})  \n"
                f"  LO(s): {los_str} | Score: {it.score} | Published: {it.published or 'N/A'}  \n"
                f"  {it.link}"
            )
            if it.summary:
                md_lines.append(f"  \n  > {it.summary}")
            md_lines.append("")
        md_bytes = ("\n".join(md_lines)).encode("utf-8")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download JSON", data=json_bytes, file_name="content_pack.json",
                               mime="application/json", use_container_width=True)
        with c2:
            st.download_button("Download Markdown", data=md_bytes, file_name="content_pack.md",
                               mime="text/markdown", use_container_width=True)