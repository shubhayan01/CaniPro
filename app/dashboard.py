"""
Keyword Cannibalization Detector — Streamlit dashboard.

Two ways to find cannibalization:
  1. Crawl a website (sitemap.xml + internal-link BFS), extract each page's
     targeting signals (title / H1 / meta / H2s), and use Groq to cluster
     pages competing for the same keyword intent.
  2. Upload a Google Search Console query x page CSV to find keywords that
     rank with multiple URLs, with clicks / impressions / CTR / position,
     severity scoring, and exportable reports.

Run:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import html
import io
import json
import os
import re
import time
from collections import defaultdict
from datetime import date, timedelta
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree as ET

import altair as alt
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

try:
    from groq import Groq
except Exception:  # groq is optional at import time
    Groq = None

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Keyword Cannibalization Detector", page_icon="🔍", layout="wide")

DEFAULT_MODEL = "llama-3.3-70b-versatile"
USER_AGENT = "Mozilla/5.0 (compatible; CanniBot/1.0; +keyword-cannibalization-detector)"
REQUEST_TIMEOUT = 12
SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


# --------------------------------------------------------------------------- #
# Groq helpers
# --------------------------------------------------------------------------- #
def get_groq_client(api_key: str):
    if not api_key or Groq is None:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def groq_json(client, model: str, system: str, user: str) -> dict | list | None:
    """Call Groq and parse a JSON object/array from the response. Returns None on failure."""
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:  # fall back to heuristics upstream
        st.session_state["last_groq_error"] = str(e)
        return None


# --------------------------------------------------------------------------- #
# Crawler
# --------------------------------------------------------------------------- #
def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    # drop fragments, keep path/query
    return parsed._replace(fragment="").geturl()


def same_registrable_host(a: str, b: str) -> bool:
    ha = urlparse(a).netloc.lower().replace("www.", "")
    hb = urlparse(b).netloc.lower().replace("www.", "")
    return ha == hb


def fetch(url: str) -> requests.Response | None:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", "") or url.endswith(".xml"):
            return r
        if r.status_code == 200:
            return r
    except requests.RequestException:
        return None
    return None


def get_sitemap_urls(base: str, cap: int) -> list[str]:
    """Collect URLs from /sitemap.xml (and nested sitemaps). Best-effort."""
    root = urlparse(base)
    candidates = [
        urljoin(f"{root.scheme}://{root.netloc}", "/sitemap.xml"),
        urljoin(f"{root.scheme}://{root.netloc}", "/sitemap_index.xml"),
    ]
    found: list[str] = []
    seen_maps: set[str] = set()
    queue = list(candidates)

    while queue and len(found) < cap:
        sm = queue.pop(0)
        if sm in seen_maps:
            continue
        seen_maps.add(sm)
        r = fetch(sm)
        if not r:
            continue
        try:
            tree = ET.fromstring(r.content)
        except ET.ParseError:
            continue
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        # nested sitemap index
        for loc in tree.findall(".//sm:sitemap/sm:loc", ns):
            if loc.text:
                queue.append(loc.text.strip())
        # actual page urls
        for loc in tree.findall(".//sm:url/sm:loc", ns):
            if loc.text and same_registrable_host(base, loc.text):
                found.append(loc.text.strip())
                if len(found) >= cap:
                    break
    return list(dict.fromkeys(found))


def _meta_content(soup, **attrs) -> str:
    el = soup.find("meta", attrs=attrs)
    return el["content"].strip() if el and el.get("content") else ""


def extract_signals(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    h1_list = [h.get_text(" ", strip=True) for h in soup.find_all("h1")[:3]]
    h1 = " | ".join(h1_list)
    h2 = " | ".join(h.get_text(" ", strip=True) for h in soup.find_all("h2")[:6])
    meta = _meta_content(soup, name="description")
    og_title = _meta_content(soup, property="og:title")
    tw_title = _meta_content(soup, attrs={"name": "twitter:title"}) or _meta_content(soup, property="twitter:title")

    # canonical
    can_el = soup.find("link", attrs={"rel": lambda v: v and "canonical" in (v if isinstance(v, list) else [v])})
    canonical = can_el["href"].strip() if can_el and can_el.get("href") else ""

    # unique-ish body length (proxy for content depth)
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    body_text = soup.get_text(" ", strip=True)
    word_count = len(body_text.split())

    # image alt coverage
    imgs = soup.find_all("img")
    n_imgs = len(imgs)
    n_missing_alt = sum(1 for i in imgs if not (i.get("alt") or "").strip())

    # JSON-LD schema types present
    schema_types: list[str] = []
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(s.string or "{}")
        except Exception:
            continue
        for node in (payload if isinstance(payload, list) else [payload]):
            if isinstance(node, dict):
                t = node.get("@type")
                if isinstance(t, list):
                    schema_types.extend(str(x) for x in t)
                elif t:
                    schema_types.append(str(t))

    links = [a["href"] for a in soup.find_all("a", href=True)]
    return {
        "url": url,
        "title": title,
        "h1": h1,
        "h1_list": h1_list,
        "h2": h2,
        "meta": meta,
        "og_title": og_title,
        "tw_title": tw_title,
        "canonical": canonical,
        "word_count": word_count,
        "n_imgs": n_imgs,
        "n_missing_alt": n_missing_alt,
        "schema_types": sorted(set(schema_types)),
        "_links": links,
    }


def crawl_site(base: str, max_pages: int, use_sitemap: bool, progress=None) -> list[dict]:
    base = normalize_url(base)
    pages: list[dict] = []
    visited: set[str] = set()

    seed_urls: list[str] = []
    if use_sitemap:
        seed_urls = get_sitemap_urls(base, cap=max_pages)
    queue = seed_urls[:] if seed_urls else [base]
    if base not in queue:
        queue.insert(0, base)

    while queue and len(pages) < max_pages:
        url = normalize_url(queue.pop(0))
        if url in visited or not same_registrable_host(base, url):
            continue
        visited.add(url)
        r = fetch(url)
        if progress:
            progress(len(pages), max_pages, url)
        if not r or "text/html" not in r.headers.get("Content-Type", ""):
            continue
        sig = extract_signals(url, r.text)
        # queue internal links (BFS fallback / discovery)
        for href in sig.pop("_links"):
            absu = normalize_url(urljoin(url, href))
            if absu not in visited and same_registrable_host(base, absu) and len(queue) < max_pages * 4:
                queue.append(absu)
        if sig["title"] or sig["h1"]:
            pages.append(sig)
        time.sleep(0.05)  # be polite
    return pages


# --------------------------------------------------------------------------- #
# Cannibalization from crawled content (Groq + heuristic fallback)
# --------------------------------------------------------------------------- #
STOPWORDS = set(
    "a an the and or for of to in on with your you our we is are be this that how what "
    "best top guide review vs your it its by from at as we’re com www https http".split()
)


def keyword_signature(page: dict) -> set[str]:
    text = f"{page.get('title','')} {page.get('h1','')}".lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def heuristic_clusters(pages: list[dict]) -> list[dict]:
    """Group pages whose title/H1 keyword sets overlap strongly."""
    groups: list[dict] = []
    sigs = [(p, keyword_signature(p)) for p in pages]
    used = set()
    for i, (p, s) in enumerate(sigs):
        if i in used or not s:
            continue
        cluster = [p]
        cluster_idx = [i]
        for j in range(i + 1, len(sigs)):
            if j in used:
                continue
            p2, s2 = sigs[j]
            if not s2:
                continue
            overlap = len(s & s2) / max(1, len(s | s2))
            if overlap >= 0.45:
                cluster.append(p2)
                cluster_idx.append(j)
        if len(cluster) > 1:
            used.update(cluster_idx)
            shared = set.intersection(*[keyword_signature(c) for c in cluster])
            keyword = " ".join(sorted(shared)[:4]) or (p.get("title") or "")[:40]
            sev = "High" if len(cluster) >= 3 else "Medium"
            groups.append(
                {
                    "keyword": keyword,
                    "severity": sev,
                    "urls": [c["url"] for c in cluster],
                    "titles": [c.get("title", "") for c in cluster],
                    "recommendation": "Pages share overlapping title/H1 keywords — consolidate or differentiate intent.",
                }
            )
    return groups


def analyze_crawl_with_groq(client, model: str, pages: list[dict]) -> list[dict]:
    if client is None:
        return heuristic_clusters(pages)

    # keep payload bounded
    slim = [
        {"url": p["url"], "title": p.get("title", "")[:120], "h1": p.get("h1", "")[:120]}
        for p in pages[:80]
    ]
    system = (
        "You are an SEO analyst. Identify keyword cannibalization: groups of URLs on the SAME site "
        "that target the same primary keyword / search intent and therefore compete with each other. "
        "Only group URLs that genuinely overlap in intent. Respond ONLY as JSON: "
        '{"groups":[{"keyword":"...","severity":"Critical|High|Medium|Low",'
        '"urls":["..."],"recommendation":"one concrete action"}]}. '
        "Severity by how directly the pages compete and how many URLs are involved. "
        "Omit keywords that map to a single URL."
    )
    user = "Pages:\n" + json.dumps(slim, ensure_ascii=False)
    data = groq_json(client, model, system, user)
    if not data or "groups" not in data:
        return heuristic_clusters(pages)

    title_map = {p["url"]: p.get("title", "") for p in pages}
    out = []
    for g in data["groups"]:
        urls = [u for u in g.get("urls", []) if u]
        if len(urls) < 2:
            continue
        out.append(
            {
                "keyword": g.get("keyword", ""),
                "severity": g.get("severity", "Medium").capitalize(),
                "urls": urls,
                "titles": [title_map.get(u, "") for u in urls],
                "recommendation": g.get("recommendation", ""),
            }
        )
    return out or heuristic_clusters(pages)


# --------------------------------------------------------------------------- #
# GSC CSV analysis
# --------------------------------------------------------------------------- #
def _find_col(cols: list[str], *aliases: str) -> str | None:
    low = {c.lower().strip(): c for c in cols}
    for a in aliases:
        if a in low:
            return low[a]
    # fuzzy contains
    for a in aliases:
        for lc, orig in low.items():
            if a in lc:
                return orig
    return None


def _to_number(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": "0", "nan": "0"})
        .astype(float)
    )


def load_gsc_csv(file) -> pd.DataFrame:
    # GSC exports sometimes have a couple of header lines; try robust parse
    raw = file.read()
    for kwargs in ({}, {"skiprows": 1}, {"sep": ";"}):
        try:
            df = pd.read_csv(io.BytesIO(raw), **kwargs)
            if df.shape[1] >= 2:
                break
        except Exception:
            continue
    else:
        raise ValueError("Could not parse CSV.")

    cols = list(df.columns)
    q = _find_col(cols, "query", "keyword", "search query", "queries")
    p = _find_col(cols, "page", "landing page", "url", "address", "pages")
    if not q or not p:
        raise ValueError(
            "CSV must contain a query column and a page/URL column. "
            f"Found columns: {cols}"
        )
    clicks = _find_col(cols, "clicks", "click")
    impr = _find_col(cols, "impressions", "impr")
    ctr = _find_col(cols, "ctr", "click through")
    pos = _find_col(cols, "position", "avg position", "average position", "rank")

    out = pd.DataFrame()
    out["query"] = df[q].astype(str).str.strip()
    out["page"] = df[p].astype(str).str.strip()
    out["clicks"] = _to_number(df[clicks]) if clicks else 0.0
    out["impressions"] = _to_number(df[impr]) if impr else 0.0
    out["ctr"] = _to_number(df[ctr]) if ctr else 0.0
    out["position"] = _to_number(df[pos]) if pos else 0.0
    out = out[(out["query"] != "") & (out["page"] != "")]
    return out


def fetch_gsc_api(sa_json_bytes: bytes, site_url: str, start: str, end: str, max_rows: int = 25000) -> pd.DataFrame:
    """Pull query x page rows straight from the Search Console API via a service account.

    Set-up (once): create a service account + JSON key in Google Cloud, enable the
    'Search Console API', then in Search Console > Settings > Users and permissions
    add the service account's client_email as a Full/Restricted user.
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Google API libraries missing. Run: pip install google-api-python-client google-auth"
        ) from e

    info = json.loads(sa_json_bytes.decode("utf-8"))
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/webmasters.readonly"]
    )
    service = build("searchconsole", "v1", credentials=creds, cache_discovery=False)

    rows: list[dict] = []
    start_row = 0
    page_size = 25000
    while len(rows) < max_rows:
        body = {
            "startDate": start,
            "endDate": end,
            "dimensions": ["query", "page"],
            "rowLimit": min(page_size, max_rows - len(rows)),
            "startRow": start_row,
        }
        resp = service.searchanalytics().query(siteUrl=site_url, body=body).execute()
        batch = resp.get("rows", [])
        if not batch:
            break
        for r in batch:
            keys = r.get("keys", ["", ""])
            rows.append(
                {
                    "query": keys[0],
                    "page": keys[1],
                    "clicks": r.get("clicks", 0.0),
                    "impressions": r.get("impressions", 0.0),
                    "ctr": r.get("ctr", 0.0) * 100.0,  # API returns a fraction
                    "position": r.get("position", 0.0),
                }
            )
        start_row += len(batch)
        if len(batch) < body["rowLimit"]:
            break

    if not rows:
        return pd.DataFrame(columns=["query", "page", "clicks", "impressions", "ctr", "position"])
    df = pd.DataFrame(rows)
    df["query"] = df["query"].astype(str).str.strip()
    df["page"] = df["page"].astype(str).str.strip()
    return df[(df["query"] != "") & (df["page"] != "")]


def gsc_severity(n_urls: int, impressions: float, pos_spread: float) -> str:
    score = 0
    score += {1: 0, 2: 1, 3: 2}.get(n_urls, 3) if n_urls >= 2 else 0
    if impressions >= 1000:
        score += 2
    elif impressions >= 100:
        score += 1
    if pos_spread >= 10:
        score += 2
    elif pos_spread >= 3:
        score += 1
    if score >= 5:
        return "Critical"
    if score >= 3:
        return "High"
    if score >= 1:
        return "Medium"
    return "Low"


def analyze_gsc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query, grp in df.groupby("query"):
        pages = grp.groupby("page").agg(
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            position=("position", "mean"),
        )
        if len(pages) < 2:
            continue  # not cannibalized
        n = len(pages)
        total_clicks = float(pages["clicks"].sum())
        total_impr = float(pages["impressions"].sum())
        avg_pos = float((pages["position"] * pages["impressions"]).sum() / max(1.0, total_impr))
        pos_spread = float(pages["position"].max() - pages["position"].min())
        ctr = (total_clicks / total_impr * 100.0) if total_impr else 0.0
        sev = gsc_severity(n, total_impr, pos_spread)
        rows.append(
            {
                "keyword": query,
                "competing_urls": n,
                "clicks": round(total_clicks),
                "impressions": round(total_impr),
                "ctr_%": round(ctr, 2),
                "avg_position": round(avg_pos, 1),
                "position_spread": round(pos_spread, 1),
                "severity": sev,
                "urls": list(pages.index),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["severity", "impressions"],
        key=lambda s: s.map(SEVERITY_ORDER) if s.name == "severity" else s,
        ascending=[True, False],
    ).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def df_download_buttons(df: pd.DataFrame, basename: str,
                        pdf_sections: list[dict] | None = None,
                        pdf_title: str | None = None,
                        pdf_summary: list[str] | None = None):
    exp = df.copy()
    for c in exp.columns:
        if exp[c].apply(lambda x: isinstance(x, list)).any():
            exp[c] = exp[c].apply(lambda x: " ; ".join(x) if isinstance(x, list) else x)
    cols = st.columns(3 if pdf_sections is not None else 2)
    cols[0].download_button("⬇️ Download CSV", exp.to_csv(index=False).encode("utf-8"),
                            f"{basename}.csv", "text/csv", use_container_width=True)
    cols[1].download_button("⬇️ Download JSON", df.to_json(orient="records", indent=2).encode("utf-8"),
                            f"{basename}.json", "application/json", use_container_width=True)
    if pdf_sections is not None:
        try:
            pdf_bytes = build_pdf_report(pdf_title or "Cannibalization report",
                                         pdf_summary or [], pdf_sections)
            cols[2].download_button("⬇️ Download PDF report", pdf_bytes,
                                    f"{basename}.pdf", "application/pdf",
                                    use_container_width=True)
        except Exception as e:  # fpdf2 missing or render failure — never crash the page
            cols[2].caption(f"PDF export needs `fpdf2` — pip install fpdf2 ({type(e).__name__})")


# --------------------------------------------------------------------------- #
# PDF report
# --------------------------------------------------------------------------- #
def _pdf_text(s: str) -> str:
    """Make text safe for FPDF's Latin-1 core fonts (map smart punctuation, drop emoji)."""
    if not s:
        return ""
    repl = {
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...", "→": "->",
        "↪": "->", "↗": "^", "↑": "", "↓": "",
        "≥": ">=", "≤": "<=", "×": "x", "·": "-",
        "✓": "-", "•": "-", " ": " ",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "ignore").decode("latin-1")


def _pdf_plain(s: str) -> str:
    """Strip light markdown (**bold**, `code`) then make the text PDF-safe."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s or "")
    s = s.replace("`", "")
    return _pdf_text(s)


def build_pdf_report(title: str, summary: list[str], sections: list[dict]) -> bytes:
    """Render a cannibalization report to PDF bytes. Raises if fpdf2 isn't installed."""
    from fpdf import FPDF

    sev_rgb = {"Critical": (185, 28, 28), "High": (154, 52, 18),
               "Medium": (133, 77, 14), "Low": (55, 65, 81)}

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def cell(h: float, txt: str, wrap: str = "WORD"):
        # multi_cell in fpdf2 defaults to leaving the cursor at the right edge, so
        # the next call gets zero width — reset to the left margin every time.
        # wrap="CHAR" lets long unbroken strings (URLs) break mid-word without raising.
        pdf.multi_cell(0, h, txt, new_x="LMARGIN", new_y="NEXT", wrapmode=wrap)

    pdf.set_font("Helvetica", "B", 18)
    cell(9, _pdf_plain(title))
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(110, 110, 110)
    cell(6, _pdf_text(f"Generated {date.today().isoformat()}  -  "
                      "Keyword Cannibalization Detector"))
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    if summary:
        pdf.set_font("Helvetica", "B", 12)
        cell(7, "Summary")
        pdf.set_font("Helvetica", "", 10)
        for line in summary:
            cell(6, _pdf_plain(f"- {line}"))
        pdf.ln(3)

    for sec in sections:
        pdf.set_font("Helvetica", "B", 13)
        r, g, b = sev_rgb.get(sec.get("severity", ""), (17, 24, 39))
        pdf.set_text_color(r, g, b)
        cell(7, _pdf_plain(sec.get("heading", "")))
        pdf.set_text_color(0, 0, 0)

        for u in sec.get("urls", []):
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(80, 80, 80)
            cell(5, _pdf_text(f"   - {u}"), wrap="CHAR")
            pdf.set_text_color(0, 0, 0)

        if sec.get("recommendation"):
            pdf.set_font("Helvetica", "", 10)
            cell(6, _pdf_plain(f"Recommendation: {sec['recommendation']}"))

        diag = [d for d in sec.get("diagnosis", []) if d]
        if diag:
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 10)
            cell(6, "Why this is holding the page back")
            pdf.set_font("Helvetica", "", 10)
            for i, d in enumerate(diag, 1):
                cell(6, _pdf_plain(f"{i}. {d}"), wrap="CHAR")

        checklist = sec.get("checklist", {})
        if any(checklist.get(k) for k, _ in PLAN_BUCKETS):
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 10)
            cell(6, "Actionable checklist")
            for slug, heading in PLAN_BUCKETS:
                items = [it for it in checklist.get(slug, []) if it]
                if not items:
                    continue
                pdf.set_font("Helvetica", "B", 9)
                cell(6, _pdf_plain(heading))
                pdf.set_font("Helvetica", "", 10)
                for it in items:
                    cell(6, _pdf_plain(f"[ ] {it}"), wrap="CHAR")
        pdf.ln(4)

    return bytes(pdf.output())


def severity_style(df: pd.DataFrame):
    colors = {"Critical": "#7f1d1d", "High": "#9a3412", "Medium": "#854d0e", "Low": "#374151"}
    return df.style.apply(
        lambda row: [f"background-color: {colors.get(row['severity'], '')}; color: white"
                     if col == "severity" else "" for col in df.columns],
        axis=1,
    )


# --------------------------------------------------------------------------- #
# Visual overview (Altair charts + KPI cards)
# --------------------------------------------------------------------------- #
# Bright, dark-mode-friendly palette for the charts (distinct from the muted
# table-cell shades in severity_style above).
SEV_LIST = ["Critical", "High", "Medium", "Low"]
SEVERITY_COLORS = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#eab308", "Low": "#9ca3af"}


def _sev_scale() -> alt.Scale:
    return alt.Scale(domain=SEV_LIST, range=[SEVERITY_COLORS[s] for s in SEV_LIST])


def _clean(chart: alt.Chart) -> alt.Chart:
    """Strip chart chrome so it sits cleanly on the app background (light or dark)."""
    return chart.configure_view(stroke=None).configure_axis(grid=False)


def severity_donut(counts: dict, title: str = "Severity mix") -> alt.Chart:
    data = pd.DataFrame({"severity": SEV_LIST, "count": [counts.get(s, 0) for s in SEV_LIST]})
    data = data[data["count"] > 0]
    total = int(data["count"].sum())
    arc = (
        alt.Chart(data)
        .mark_arc(innerRadius=58, outerRadius=92, stroke="#0e1117", strokeWidth=2)
        .encode(
            theta=alt.Theta("count:Q", stack=True),
            color=alt.Color("severity:N", scale=_sev_scale(), sort=SEV_LIST,
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[alt.Tooltip("severity:N", title="Severity"),
                     alt.Tooltip("count:Q", title="Keywords")],
        )
    )
    center = (
        alt.Chart(pd.DataFrame({"n": [total]}))
        .mark_text(size=34, fontWeight="bold", color="#e5e7eb")
        .encode(text="n:Q")
    )
    return _clean((arc + center).properties(height=260, title=title))


def top_keywords_bar(report: pd.DataFrame, metric: str, label: str, n: int = 10) -> alt.Chart:
    d = report.nlargest(n, metric).copy()
    tips = [alt.Tooltip("keyword:N", title="Keyword"),
            alt.Tooltip("severity:N", title="Severity"),
            alt.Tooltip(f"{metric}:Q", title=label, format=","),
            alt.Tooltip("competing_urls:Q", title="Competing URLs")]
    chart = (
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=4, height=alt.RelativeBandSize(0.72))
        .encode(
            x=alt.X(f"{metric}:Q", title=label, axis=alt.Axis(format="~s")),
            y=alt.Y("keyword:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=260)),
            color=alt.Color("severity:N", scale=_sev_scale(), sort=SEV_LIST, legend=None),
            tooltip=tips,
        )
        .properties(height=max(220, 30 * len(d)), title=f"Top {len(d)} keywords by {label.lower()}")
    )
    return _clean(chart)


def impact_scatter(report: pd.DataFrame) -> alt.Chart:
    """Impact map: worse rank (right) + more impressions (up) + bigger bubble = higher priority."""
    d = report.copy()
    chart = (
        alt.Chart(d)
        .mark_circle(opacity=0.8, stroke="#0e1117", strokeWidth=0.6)
        .encode(
            x=alt.X("avg_position:Q", title="Avg position  →  worse",
                    scale=alt.Scale(zero=False, nice=True)),
            y=alt.Y("impressions:Q", title="Impressions affected", axis=alt.Axis(format="~s")),
            size=alt.Size("competing_urls:Q", title="Competing URLs",
                          scale=alt.Scale(range=[80, 900])),
            color=alt.Color("severity:N", scale=_sev_scale(), sort=SEV_LIST,
                            legend=alt.Legend(title="Severity", orient="bottom")),
            tooltip=[alt.Tooltip("keyword:N", title="Keyword"),
                     alt.Tooltip("severity:N", title="Severity"),
                     alt.Tooltip("competing_urls:Q", title="Competing URLs"),
                     alt.Tooltip("impressions:Q", title="Impressions", format=","),
                     alt.Tooltip("clicks:Q", title="Clicks", format=","),
                     alt.Tooltip("avg_position:Q", title="Avg position"),
                     alt.Tooltip("position_spread:Q", title="Position spread")],
        )
        .properties(height=340, title="Impact map — bigger & higher = fix first")
    )
    return _clean(chart)


def _sev_counts(df: pd.DataFrame) -> dict:
    return {s: int((df["severity"] == s).sum()) for s in SEV_LIST}


def render_gsc_overview(report: pd.DataFrame):
    """Graphical, at-a-glance overview for the Search Console report."""
    counts = _sev_counts(report)
    total_impr = int(report["impressions"].sum())
    total_clicks = int(report["clicks"].sum())

    st.markdown("### 📊 Overview")
    k = st.columns(6)
    k[0].metric("Cannibalized keywords", f"{len(report):,}")
    k[1].metric("🔴 Critical", counts["Critical"])
    k[2].metric("🟠 High", counts["High"])
    k[3].metric("🟡 Medium", counts["Medium"])
    k[4].metric("Impressions affected", f"{total_impr:,}")
    k[5].metric("Clicks affected", f"{total_clicks:,}")

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.altair_chart(severity_donut(counts), use_container_width=True)
    with right:
        st.altair_chart(top_keywords_bar(report, "impressions", "Impressions"),
                        use_container_width=True)
    st.altair_chart(impact_scatter(report), use_container_width=True)


def render_crawl_overview(rep: pd.DataFrame):
    """Graphical overview for the crawl report (keyword / severity / competing_urls)."""
    counts = _sev_counts(rep)

    st.markdown("### 📊 Overview")
    k = st.columns(5)
    k[0].metric("Clusters found", f"{len(rep):,}")
    k[1].metric("🔴 Critical", counts["Critical"])
    k[2].metric("🟠 High", counts["High"])
    k[3].metric("🟡 Medium", counts["Medium"])
    k[4].metric("Competing pages", int(rep["competing_urls"].sum()))

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.altair_chart(severity_donut(counts), use_container_width=True)
    with right:
        st.altair_chart(top_keywords_bar(rep, "competing_urls", "Competing URLs"),
                        use_container_width=True)


# --------------------------------------------------------------------------- #
# Cannibalization explainers (why + how-to-fix)
# --------------------------------------------------------------------------- #
# An experienced-SEO resolution playbook, ordered the way you'd actually work a
# cannibalization case: confirm → pick a winner → consolidate/differentiate →
# clean up internal links → re-index → monitor.
FIX_CHECKLIST = [
    "**Confirm the overlap** — open each URL and check they genuinely answer the same query. "
    "Different intent → keep both but differentiate; same intent → consolidate.",
    "**Pick the primary URL** — keep the one with the strongest signals (best average position, "
    "most clicks, most backlinks, deepest/most-recent content). That's your winner.",
    "**Consolidate _or_ canonicalize** — merge the weaker pages' unique value into the winner, then "
    "301-redirect them to it. If every page must stay live, add `rel=\"canonical\"` from each "
    "secondary page to the primary instead.",
    "**Differentiate if you keep them** — re-target each secondary page to a distinct keyword/intent: "
    "rewrite its title tag, H1 and opening paragraph so they no longer chase the same term.",
    "**Fix internal links** — make internal anchor text for this keyword point to the single chosen "
    "URL only; remove or re-anchor links that feed the losing pages.",
    "**Re-index & monitor** — update the sitemap, request re-indexing in Search Console, then watch "
    "rankings and clicks consolidate onto the winner over the next 2–4 weeks.",
]


def render_cluster_links(urls: list[str], titles: list[str] | None = None):
    """Render competing URLs as real, always-clickable links (open in a new tab).

    Uses HTML anchors rather than markdown `[t](u)` so URLs containing `)`,
    spaces or other markdown-breaking characters still link correctly.
    """
    titles = titles or [""] * len(urls)
    items = []
    for u, t in zip(urls, titles):
        label = html.escape(t or u)
        safe_url = html.escape(u, quote=True)
        raw = html.escape(u)
        items.append(
            f'<li style="margin-bottom:6px">'
            f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{label}</a>'
            f'<br><span style="color:#9ca3af;font-size:0.8em">{raw}</span></li>'
        )
    st.markdown(f'<ul style="margin:0;padding-left:1.1em">{"".join(items)}</ul>',
                unsafe_allow_html=True)


def crawl_cannibalization_reason(keyword: str, titles: list[str]) -> str:
    """Explain *why* a crawled cluster competes, from the shared title/H1 keywords."""
    sigs = [keyword_signature({"title": t}) for t in titles if t]
    shared = set.intersection(*sigs) if len(sigs) >= 2 else set()
    if shared:
        terms = ", ".join(f"`{w}`" for w in sorted(shared)[:6])
        return (
            f"These {len(titles)} pages target the same intent — their titles/H1s all share "
            f"the keywords {terms}. Google can't tell which one should rank for **{keyword}**, "
            f"so it splits ranking authority, clicks and CTR between them and often ranks the "
            f"weaker page."
        )
    return (
        f"These pages compete for **{keyword}**: their titles and headings overlap enough that "
        f"they target the same search intent, so Google splits ranking signals between them."
    )


def gsc_cannibalization_reason(keyword: str, n_urls: int, pos_spread: float, impressions: int) -> str:
    """Explain *why* a GSC keyword is cannibalized, from the observed ranking data."""
    return (
        f"**{n_urls} URLs** on your site are ranking for **{keyword}** at the same time "
        f"({impressions:,} impressions). Their positions are spread by **{pos_spread:.1f}** places, "
        f"which means Google keeps swapping which page it shows and dilutes clicks — a strong page "
        f"could rank higher if these signals were consolidated onto one URL."
    )


# --------------------------------------------------------------------------- #
# Data-specific fix plans (per cannibalization, not a generic list)
# --------------------------------------------------------------------------- #
# Priority buckets, colour-coded like the reference audit.
PLAN_BUCKETS = [
    ("immediate", "🔴 Fix immediately — on-page targeting & meta bugs"),
    ("cannibalization", "🟠 Fix soon — resolve the cannibalization"),
    ("content", "🟡 Strengthen the pillar page — content depth"),
    ("technical", "🟢 Technical / structural"),
]


def _path(url: str) -> str:
    """Short, readable form of a URL for inline references (path only when possible)."""
    try:
        p = urlparse(url)
        short = (p.path or "/") + (f"?{p.query}" if p.query else "")
        return short or url
    except Exception:
        return url


def _q(s: str, limit: int = 90) -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) > limit:
        s = s[:limit - 1] + "…"
    return f"“{s}”" if s else "—"


def _empty_plan() -> dict:
    return {"diagnosis": [], "checklist": {k: [] for k, _ in PLAN_BUCKETS}}


def build_crawl_fix_plan(keyword: str, pages: list[dict]) -> dict:
    """A concrete, evidence-based fix plan for one crawled cluster.

    Reads the real per-page signals (title / og:title / twitter:title / H1 /
    word count / alt coverage / schema) and names the actual URLs and mismatches
    — so the checklist is specific to *this* cannibalization, not generic advice.
    """
    plan = _empty_plan()
    if not pages:
        return plan

    # Pillar candidate = the page most ON-TOPIC for the keyword (its title/H1/URL
    # slug overlap the keyword most), tie-broken by content depth. Using relevance
    # rather than raw length avoids crowning a long but off-topic page (e.g. a
    # suburb/window-tint page that only *mentions* the keyword).
    kw_tokens = keyword_signature({"title": keyword})

    def _relevance(p: dict) -> int:
        slug = urlparse(p["url"]).path.replace("-", " ").replace("/", " ")
        sig = keyword_signature({"title": f"{p.get('title','')} {p.get('h1','')} {slug}"})
        return len(kw_tokens & sig)

    primary = max(pages, key=lambda p: (_relevance(p), p.get("word_count", 0)))
    secondaries = [p for p in pages if p["url"] != primary["url"]]
    pu = primary["url"]

    diag = plan["diagnosis"]
    imm = plan["checklist"]["immediate"]
    cann = plan["checklist"]["cannibalization"]
    cont = plan["checklist"]["content"]
    tech = plan["checklist"]["technical"]

    # 1) Inconsistent targeting across the tags Google weighs for relevance.
    tags = {
        "<title>": primary.get("title", ""),
        "og:title": primary.get("og_title", ""),
        "twitter:title": primary.get("tw_title", ""),
        "H1": primary.get("h1", ""),
    }
    present = {k: v for k, v in tags.items() if v}
    distinct = {v.strip().lower() for v in present.values()}
    if len(present) >= 2 and len(distinct) >= 2:
        listed = "; ".join(f"{k} = {_q(v)}" for k, v in present.items())
        diag.append(
            f"**Inconsistent targeting on `{_path(pu)}`** — the tags Google weighs most for "
            f"relevance disagree: {listed}. Pick one target and make them match."
        )
        imm.append(f"Decide ONE primary term for `{_path(pu)}` (align it to **{keyword}** or your best geo/volume pick).")
        imm.append(f"Rewrite so <title>, og:title, twitter:title and H1 on `{_path(pu)}` all say the same thing (they currently read {listed}).")

    # 2) Copy-paste meta leftovers on secondary pages (e.g. the Edina/twitter:title bug).
    for p in pages:
        for tag, val in (("og:title", p.get("og_title", "")), ("twitter:title", p.get("tw_title", ""))):
            own = keyword_signature({"title": f"{p.get('title','')} {p.get('h1','')}"})
            valsig = keyword_signature({"title": val})
            if val and own and valsig and not (own & valsig):
                diag.append(
                    f"**Possible copy-pasted meta on `{_path(p['url'])}`** — its {tag} reads {_q(val)}, "
                    f"which doesn't match the page's own title/H1. Looks like a leftover template tag."
                )
                imm.append(f"Fix the {tag} on `{_path(p['url'])}` — it currently says {_q(val)}, unrelated to that page.")

    # 3) The cannibalization itself.
    diag.append(crawl_cannibalization_reason(keyword, [p.get("title", "") for p in pages]))
    cann.append(
        f"Choose `{_path(pu)}` as the single pillar for **{keyword}** (closest match to the intent) "
        f"and point everything else at it."
    )
    for p in secondaries:
        cann.append(
            f"Trim the **{keyword}** section on `{_path(p['url'])}` (~{p.get('word_count', 0):,} words, "
            f"near-duplicate) to 1–2 sentences + a link to `{_path(pu)}`."
        )
    cann.append(f"Add internal links from every page above to `{_path(pu)}` using anchor text like {_q(keyword, 40)}.")
    cann.append(f"In Search Console, filter Performance by {_q(keyword, 40)} and noindex/merge any URL that's clearly redundant.")

    # 4) Content depth on the pillar.
    wc = primary.get("word_count", 0)
    if wc < 500:
        cont.append(
            f"Expand `{_path(pu)}` from ~{wc:,} to 500–800+ words: real pricing/ranges, process "
            f"specifics, maintenance, and comparisons — out-depth the pages currently outranking you."
        )
    cont.append(f"Add an FAQ section to `{_path(pu)}` (captures long-tail queries + enables FAQ schema).")
    if primary.get("n_imgs", 0) and primary.get("n_missing_alt", 0):
        cont.append(
            f"Add descriptive, keyword-relevant alt text on `{_path(pu)}` — "
            f"{primary['n_missing_alt']}/{primary['n_imgs']} images are missing alt attributes."
        )

    # 5) Technical.
    stypes = primary.get("schema_types", [])
    if not any(t in {"Service", "LocalBusiness", "Product"} for t in stypes):
        found = ", ".join(stypes) if stypes else "none detected"
        tech.append(f"Add Service / LocalBusiness schema to `{_path(pu)}` (schema currently: {found}).")
    tech.append("Add FAQPage schema once the FAQ section is live.")
    can = primary.get("canonical", "")
    if can and _path(can) != _path(pu):
        tech.append(f"Fix the canonical on `{_path(pu)}` — it points to `{_path(can)}`, not itself.")
    else:
        tech.append(f"Confirm the canonical on `{_path(pu)}` points to itself.")
    tech.append(f"Re-check rankings for {_q(keyword, 40)} 3–4 weeks after these changes ship.")

    return plan


def build_gsc_fix_plan(keyword: str, sub: pd.DataFrame, pos_spread: float, impressions: int) -> dict:
    """A concrete fix plan for one cannibalized GSC keyword, built from the real
    per-URL clicks / impressions / position numbers — names the winner and each
    page to consolidate, with its actual metrics.
    """
    plan = _empty_plan()
    if sub.empty:
        return plan

    ranked = sub.sort_values("position").reset_index(drop=True)  # best (lowest) position first
    winner = ranked.iloc[0]
    wu = str(winner["page"])
    n = len(ranked)

    diag = plan["diagnosis"]
    imm = plan["checklist"]["immediate"]
    cann = plan["checklist"]["cannibalization"]
    cont = plan["checklist"]["content"]
    tech = plan["checklist"]["technical"]

    diag.append(gsc_cannibalization_reason(keyword, n, pos_spread, impressions))
    diag.append(
        f"Strongest page is `{_path(wu)}` (avg pos **{winner['position']:.1f}**, "
        f"{int(winner['clicks']):,} clicks) — the other {n - 1} split the same query."
    )
    top_clicks = ranked.sort_values("clicks", ascending=False).iloc[0]
    if str(top_clicks["page"]) != wu:
        diag.append(
            f"Note the split: `{_path(str(top_clicks['page']))}` gets the most clicks "
            f"({int(top_clicks['clicks']):,}) but `{_path(wu)}` ranks better — decide which one becomes the winner."
        )

    imm.append(f"Confirm all {n} URLs truly target the same intent for {_q(keyword, 40)}; if one differs, re-target it instead of merging.")

    cann.append(f"Pick `{_path(wu)}` as the winner (best position {winner['position']:.1f}, {int(winner['clicks']):,} clicks).")
    for _, r in ranked.iloc[1:].iterrows():
        cann.append(
            f"Consolidate `{_path(str(r['page']))}` (pos {r['position']:.1f}, {int(r['clicks']):,} clicks, "
            f"{int(r['impressions']):,} impr) into `{_path(wu)}` — 301-redirect it, or add rel=\"canonical\" → `{_path(wu)}`."
        )
    cann.append(f"Re-anchor internal links for {_q(keyword, 40)} to point at `{_path(wu)}` only.")
    cann.append("If any page must stay live, differentiate it: rewrite its title, H1 and intro to target a distinct query.")

    cont.append(f"Before redirecting, merge any unique sections from the other URLs into `{_path(wu)}` so no content is lost.")

    tech.append(f"After consolidating, update the sitemap and request re-indexing for `{_path(wu)}` in Search Console.")
    tech.append(f"Re-check {_q(keyword, 40)} in 2–4 weeks — clicks and position should consolidate onto `{_path(wu)}`.")

    return plan


def groq_fix_plan(client, model: str, context: str) -> dict | None:
    """Ask Groq to turn concrete page signals into a prioritized, evidence-cited plan.
    Returns the same shape as the heuristics, or None to fall back."""
    if client is None:
        return None
    system = (
        "You are a senior technical-SEO consultant resolving keyword cannibalization. "
        "Using ONLY the concrete signals provided (never invent URLs, tags or numbers), produce a "
        "specific, evidence-cited fix plan. Cite the actual URLs, tag values and metrics you were given. "
        "Respond ONLY as JSON: "
        '{"diagnosis":["short bullet citing specific evidence", ...],'
        '"checklist":{"immediate":["actionable item", ...],"cannibalization":[...],'
        '"content":[...],"technical":[...]}}. '
        "Each checklist item must be a single concrete action referencing a real URL/tag/number from the input. "
        "immediate = on-page targeting & meta-tag bugs; cannibalization = pick winner / consolidate / internal links; "
        "content = depth on the winning page; technical = schema, canonical, sitemap, re-index, monitor."
    )
    data = groq_json(client, model, system, context)
    if not isinstance(data, dict) or "checklist" not in data:
        return None
    cl = data.get("checklist") or {}
    out = {"diagnosis": [str(d) for d in (data.get("diagnosis") or []) if d],
           "checklist": {k: [str(i) for i in (cl.get(k) or []) if i] for k, _ in PLAN_BUCKETS}}
    if not any(out["checklist"].values()):
        return None
    return out


def _crawl_plan_context(keyword: str, pages: list[dict]) -> str:
    slim = [
        {
            "url": p["url"], "title": p.get("title", ""), "og_title": p.get("og_title", ""),
            "twitter_title": p.get("tw_title", ""), "h1": p.get("h1", ""),
            "word_count": p.get("word_count", 0), "images": p.get("n_imgs", 0),
            "images_missing_alt": p.get("n_missing_alt", 0), "schema_types": p.get("schema_types", []),
            "canonical": p.get("canonical", ""), "meta_description": p.get("meta", "")[:160],
        }
        for p in pages
    ]
    return f"Keyword/intent: {keyword}\nPages competing for it:\n" + json.dumps(slim, ensure_ascii=False)


def _gsc_plan_context(keyword: str, sub: pd.DataFrame, pos_spread: float, impressions: int) -> str:
    rows = [
        {"url": str(r["page"]), "clicks": int(r["clicks"]),
         "impressions": int(r["impressions"]), "avg_position": round(float(r["position"]), 1)}
        for _, r in sub.sort_values("position").iterrows()
    ]
    return (
        f"Keyword: {keyword}\nTotal impressions: {impressions}\nPosition spread: {pos_spread}\n"
        f"URLs ranking for it (best position first):\n" + json.dumps(rows, ensure_ascii=False)
    )


def render_fix_plan(plan: dict, anchor_key: str):
    """Render a data-specific plan: a multi-point 'problem' diagnosis + a priority-grouped checklist."""
    render_diagnosis(plan)

    st.markdown("#### ✅ Actionable checklist")
    checklist = plan.get("checklist", {})
    rendered = False
    for slug, heading in PLAN_BUCKETS:
        items = [it for it in checklist.get(slug, []) if it]
        if not items:
            continue
        rendered = True
        st.markdown(f"**{heading}**")
        st.markdown("\n".join(f"- [ ] {it}" for it in items))
    if not rendered:  # ultimate fallback
        st.markdown("\n".join(f"- [ ] {s}" for s in FIX_CHECKLIST))


def render_diagnosis(plan: dict):
    """Render just the multi-point 'why this is holding the page back' diagnosis."""
    diag = [d for d in plan.get("diagnosis", []) if d]
    if diag:
        st.markdown("#### 🔴 Why this is holding the page back")
        st.markdown("\n".join(f"{i}. {d}" for i, d in enumerate(diag, 1)))


# --------------------------------------------------------------------------- #
# Per-link (per-URL) checklists — what to do for THIS specific page
# --------------------------------------------------------------------------- #
# Instead of one checklist for the whole cluster, each competing URL gets its
# own to-do list, keyed to its role (the pillar/winner you keep vs. a secondary
# you consolidate) and its actual signals/metrics.
def _crawl_primary(keyword: str, pages: list[dict]) -> dict:
    """Pick the pillar page for a crawled cluster: most on-topic for the keyword,
    tie-broken by content depth (same logic as build_crawl_fix_plan)."""
    kw_tokens = keyword_signature({"title": keyword})

    def _relevance(p: dict) -> int:
        slug = urlparse(p["url"]).path.replace("-", " ").replace("/", " ")
        sig = keyword_signature({"title": f"{p.get('title','')} {p.get('h1','')} {slug}"})
        return len(kw_tokens & sig)

    return max(pages, key=lambda p: (_relevance(p), p.get("word_count", 0)))


def build_crawl_link_checklist(keyword: str, page: dict, primary: dict) -> tuple[str, list[str]]:
    """What to do for ONE crawled URL, given its role in the cluster.

    Returns (role_label, [checklist items]) using the page's real signals
    (title / og:title / twitter:title / H1 / word count / alt coverage / schema).
    """
    is_primary = page["url"] == primary["url"]
    pth = _path(page["url"])
    actions: list[str] = []

    if is_primary:
        role = "🏆 Primary / pillar — keep this and consolidate the others into it"
        tags = {
            "<title>": page.get("title", ""),
            "og:title": page.get("og_title", ""),
            "twitter:title": page.get("tw_title", ""),
            "H1": page.get("h1", ""),
        }
        present = {k: v for k, v in tags.items() if v}
        distinct = {v.strip().lower() for v in present.values()}
        if len(present) >= 2 and len(distinct) >= 2:
            listed = "; ".join(f"{k} = {_q(v)}" for k, v in present.items())
            actions.append(
                f"Align every targeting tag to **{keyword}** — <title>, og:title, twitter:title and H1 "
                f"currently disagree ({listed})."
            )
        else:
            actions.append(f"Confirm <title>, og:title, twitter:title and H1 all target **{keyword}**.")
        wc = page.get("word_count", 0)
        if wc < 500:
            actions.append(
                f"Expand from ~{wc:,} to 500–800+ words — real pricing/ranges, process specifics and "
                f"comparisons so it out-depths the pages consolidating into it."
            )
        actions.append("Add an FAQ section (captures long-tail queries + enables FAQ schema).")
        if page.get("n_imgs", 0) and page.get("n_missing_alt", 0):
            actions.append(
                f"Add descriptive, keyword-relevant alt text — "
                f"{page['n_missing_alt']}/{page['n_imgs']} images are missing alt attributes."
            )
        stypes = page.get("schema_types", [])
        if not any(t in {"Service", "LocalBusiness", "Product"} for t in stypes):
            found = ", ".join(stypes) if stypes else "none detected"
            actions.append(f"Add Service / LocalBusiness schema (schema currently: {found}).")
        can = page.get("canonical", "")
        if can and _path(can) != pth:
            actions.append(f"Fix the canonical — it points to `{_path(can)}`, not itself.")
        else:
            actions.append("Confirm the canonical points to itself.")
        actions.append(
            f"Re-check rankings for {_q(keyword, 40)} 3–4 weeks after the other pages are consolidated in."
        )
    else:
        pp = _path(primary["url"])
        role = "↪️ Secondary — consolidate into the pillar"
        actions.append(
            f"Trim the **{keyword}** section here (~{page.get('word_count', 0):,} words, near-duplicate) "
            f"to 1–2 sentences + a link to the pillar."
        )
        actions.append(
            f"Add an internal link to the pillar `{pp}` using anchor text like {_q(keyword, 40)}."
        )
        actions.append(
            f"301-redirect to `{pp}`, or add rel=\"canonical\" → `{pp}` if it must stay live."
        )
        for tag, val in (("og:title", page.get("og_title", "")), ("twitter:title", page.get("tw_title", ""))):
            own = keyword_signature({"title": f"{page.get('title','')} {page.get('h1','')}"})
            valsig = keyword_signature({"title": val})
            if val and own and valsig and not (own & valsig):
                actions.append(
                    f"Fix the {tag} — it reads {_q(val)}, unrelated to this page (leftover template tag)."
                )
        actions.append(
            "If it must stay live, differentiate it: rewrite title, H1 and intro to target a distinct query."
        )
    return role, actions


def build_gsc_link_checklist(keyword: str, row: dict, winner: dict, is_winner: bool) -> tuple[str, list[str]]:
    """What to do for ONE GSC URL, given its role (winner vs consolidate) and its
    real clicks / impressions / position numbers."""
    pth = _path(str(row["page"]))
    wu = _path(str(winner["page"]))
    actions: list[str] = []

    if is_winner:
        role = "🏆 Winner — keep this and consolidate the others into it"
        actions.append(
            f"Keep as the single winner (best position {float(row['position']):.1f}, "
            f"{int(row['clicks']):,} clicks)."
        )
        actions.append("Merge any unique sections from the other URLs in here before redirecting them.")
        actions.append("Update the sitemap and request re-indexing for this URL in Search Console.")
        actions.append(
            f"Re-check {_q(keyword, 40)} in 2–4 weeks — clicks and position should consolidate here."
        )
    else:
        role = "↪️ Consolidate into the winner"
        actions.append(
            f"Confirm it targets the same intent as {_q(keyword, 40)}; if it differs, re-target it "
            f"instead of merging."
        )
        actions.append(f"301-redirect to `{wu}`, or add rel=\"canonical\" → `{wu}`.")
        actions.append(f"Re-anchor internal links for {_q(keyword, 40)} to point at `{wu}` only.")
        actions.append(
            f"This page holds pos {float(row['position']):.1f}, {int(row['clicks']):,} clicks, "
            f"{int(row['impressions']):,} impr — merge that value into the winner so nothing is lost."
        )
    return role, actions


def render_link_checklist(url: str, role: str, actions: list[str], meta_md: str | None = None):
    """Render one competing URL as a click-to-open panel holding ITS OWN checklist.

    The expander label is the URL, so clicking a link reveals what to do for that
    specific page. (Expanders can't be nested, so the caller must place this inside
    a plain container, not another expander.)"""
    with st.expander(f"🔗 {url}"):
        st.caption(role)
        if meta_md:
            st.markdown(meta_md)
        st.markdown("**✅ Checklist for this URL:**")
        st.markdown("\n".join(f"- [ ] {a}" for a in actions))
        safe_url = html.escape(url, quote=True)
        st.markdown(
            f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">Open page ↗</a>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("🔍 Keyword Cannibalization Detector")
st.caption("Crawl a site or import Search Console data to find keywords competing across multiple URLs — powered by Groq.")


with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API key", type="password",
                            value=os.environ.get("GROQ_API_KEY", ""),
                            help="Get one free at console.groq.com. Optional — a heuristic fallback runs without it.")
    model = st.text_input("Groq model", value=DEFAULT_MODEL)
    client = get_groq_client(api_key)
    if api_key and client is None:
        st.warning("Groq client unavailable (check key / `pip install groq`). Using heuristic fallback.")
    st.markdown("---")
    st.markdown("**Deploy:** `streamlit run app/dashboard.py`")

def render_crawl_results(crawl: dict):
    """Render a stored crawl result. Reads everything from session_state so the
    report survives Streamlit reruns (clicking an expander, switching tabs, etc.)
    instead of vanishing the moment the Crawl button is no longer 'pressed'."""
    pages = crawl.get("pages") or []
    groups = crawl.get("groups") or []
    rep = crawl.get("rep")
    plans = crawl.get("plans") or []

    if not pages:
        st.error("No pages could be fetched. Check the URL, or the site may block crawlers.")
        return
    st.success(f"Fetched {len(pages)} pages.")
    if not groups or rep is None:
        st.success("✅ No keyword cannibalization detected across crawled pages.")
        return

    st.warning(f"Found {len(groups)} cannibalization cluster(s).")

    # Graphical at-a-glance overview.
    render_crawl_overview(rep)

    # Full table (the raw url list isn't clickable here — the real
    # clickable links live in each cluster's expander below).
    with st.expander("📋 Full table", expanded=False):
        st.dataframe(severity_style(rep.drop(columns=["urls"])), use_container_width=True)

    sig_by_url = {p["url"]: p for p in pages}
    st.markdown("### Affected keywords")
    st.caption("Each cluster lists its competing pages — **click a URL to see the checklist for that page.**")
    for g, plan in zip(groups, plans):
        cps = [sig_by_url[u] for u in g["urls"] if u in sig_by_url]
        primary = _crawl_primary(g["keyword"], cps) if cps else None
        titles = g.get("titles", [""] * len(g["urls"]))
        title_by_url = {u: t for u, t in zip(g["urls"], titles)}
        with st.container(border=True):
            st.markdown(f"#### [{g['severity']}] {g['keyword']} — {len(g['urls'])} URLs")
            render_diagnosis(plan)
            if g.get("recommendation"):
                st.info("💡 " + g["recommendation"])
            st.markdown("**🔗 Competing pages — click one for its checklist:**")
            for u in g["urls"]:
                page = sig_by_url.get(u)
                if page and primary:
                    role, actions = build_crawl_link_checklist(g["keyword"], page, primary)
                else:  # no crawled signals for this URL — generic guidance
                    role, actions = "Competing page", list(FIX_CHECKLIST)
                t = title_by_url.get(u, "")
                meta = f"*{html.escape(t)}*" if t else None
                render_link_checklist(u, role, actions, meta_md=meta)

    # Downloads — CSV / JSON / PDF.
    counts = _sev_counts(rep)
    summary = [
        f"Pages crawled: {len(pages)}",
        f"Cannibalization clusters: {len(rep)}",
        f"Critical: {counts['Critical']}   High: {counts['High']}   "
        f"Medium: {counts['Medium']}   Low: {counts['Low']}",
        f"Total competing pages: {int(rep['competing_urls'].sum())}",
    ]
    sections = [
        {
            "heading": f"[{g['severity']}] {g['keyword']} — {len(g['urls'])} URLs",
            "severity": g["severity"],
            "urls": g["urls"],
            "recommendation": g.get("recommendation", ""),
            "diagnosis": plan.get("diagnosis", []),
            "checklist": plan.get("checklist", {}),
        }
        for g, plan in zip(groups, plans)
    ]
    df_download_buttons(rep, "cannibalization_crawl_report",
                        pdf_sections=sections,
                        pdf_title="Keyword Cannibalization — Crawl Report",
                        pdf_summary=summary)


tab_crawl, tab_gsc = st.tabs(["🌐 Crawl a website", "📊 Search Console CSV"])


# ---- Tab 1: crawl -------------------------------------------------------- #
with tab_crawl:
    st.subheader("Scan a website for content cannibalization")
    url_in = st.text_input("Website URL", placeholder="https://example.com")
    c1, c2 = st.columns(2)
    max_pages = c1.slider("Max pages to crawl", 5, 5000, 60, step=5)
    use_sitemap = c2.toggle("Use sitemap.xml if available", value=True)

    run = st.button("🚀 Crawl & analyze", type="primary", disabled=not url_in)
    if st.session_state.get("crawl") is not None and not run:
        if st.button("🗑️ Clear results"):
            st.session_state.pop("crawl", None)
            st.rerun()

    if run:
        prog = st.progress(0.0, text="Starting…")

        def _p(done, total, cur):
            prog.progress(min(done / total, 1.0), text=f"Crawled {done}/{total} — {cur[:70]}")

        with st.spinner("Crawling…"):
            pages = crawl_site(url_in, max_pages, use_sitemap, progress=_p)
        prog.empty()

        groups: list[dict] = []
        rep = None
        plans: list[dict] = []
        if pages:
            with st.spinner("Analyzing keyword overlap…"):
                groups = analyze_crawl_with_groq(client, model, pages)
            if groups:
                rows = [
                    {
                        "keyword": g["keyword"],
                        "severity": g["severity"],
                        "competing_urls": len(g["urls"]),
                        "urls": g["urls"],
                        "recommendation": g.get("recommendation", ""),
                    }
                    for g in groups
                ]
                rep = pd.DataFrame(rows).sort_values(
                    "severity", key=lambda s: s.map(SEVERITY_ORDER)
                ).reset_index(drop=True)
                # Keep the cluster/plan order aligned to the sorted report.
                order = {kw: i for i, kw in enumerate(rep["keyword"])}
                groups = sorted(groups, key=lambda g: order.get(g["keyword"], 0))

                # Build a concrete, evidence-based fix plan per cluster from the
                # real page signals (Groq-written when a key is set, else heuristic).
                sig_by_url = {p["url"]: p for p in pages}
                with st.spinner("Building per-cluster fix plans…"):
                    for g in groups:
                        cps = [sig_by_url[u] for u in g["urls"] if u in sig_by_url]
                        plan = (groq_fix_plan(client, model, _crawl_plan_context(g["keyword"], cps))
                                if client and cps else None)
                        if not plan:
                            plan = build_crawl_fix_plan(g["keyword"], cps)
                        plans.append(plan)

        # Persist so the report survives reruns (expander clicks, tab switches…).
        st.session_state["crawl"] = {"pages": pages, "groups": groups,
                                     "rep": rep, "plans": plans}

    crawl = st.session_state.get("crawl")
    if crawl is not None:
        render_crawl_results(crawl)

# ---- Tab 2: GSC ---------------------------------------------------------- #
with tab_gsc:
    st.subheader("Detect keywords ranking with multiple URLs")
    source = st.radio("Data source", ["Upload CSV", "Fetch via Search Console API"], horizontal=True)
    df = None

    if source == "Upload CSV":
        st.caption("Export a **query × page** report from Search Console (or any SEO tool) with clicks, "
                   "impressions, CTR and position columns, then upload it here.")
        file = st.file_uploader("Upload Search Console CSV", type=["csv"])
        if file is not None:
            try:
                df = load_gsc_csv(file)
            except ValueError as e:
                st.error(str(e))

    else:  # live API via service account
        with st.expander("ℹ️ One-time setup"):
            st.markdown(
                "1. In **Google Cloud** create a *service account* + JSON key and enable the "
                "**Search Console API**.\n"
                "2. In **Search Console → Settings → Users and permissions**, add the service "
                "account's `client_email` as a **Full** or **Restricted** user.\n"
                "3. Upload the JSON key below. Nothing leaves your session."
            )
        sa = st.file_uploader("Service account JSON key", type=["json"])
        site = st.text_input("Property (site URL)", placeholder="https://example.com/  or  sc-domain:example.com")
        c1, c2, c3 = st.columns(3)
        start = c1.date_input("Start date", value=date.today() - timedelta(days=90))
        end = c2.date_input("End date", value=date.today() - timedelta(days=2))
        max_rows = c3.number_input("Max rows", 100, 25000, 25000, step=1000)
        if st.button("📥 Fetch from API", type="primary", disabled=not (sa and site)):
            try:
                with st.spinner("Querying Search Console…"):
                    df = fetch_gsc_api(sa.read(), site.strip(), str(start), str(end), int(max_rows))
                if df is None or df.empty:
                    st.warning("No rows returned for that property / date range.")
                    df = None
            except Exception as e:
                st.error(f"API error: {e}")
                df = None

    if df is not None:
            st.success(f"Loaded {len(df):,} query-page rows.")
            report = analyze_gsc(df)
            if report.empty:
                st.success("✅ No cannibalization: every query maps to a single URL.")
            else:
                # Graphical at-a-glance overview.
                render_gsc_overview(report)

                use_ai = st.toggle(
                    "🤖 Deepen checklists with Groq (uses your API key)",
                    value=False, disabled=not client,
                    help="Off = fast, data-specific heuristic plans. On = Groq rewrites the top plans with richer prose.",
                )

                with st.expander("📋 Full table", expanded=False):
                    display = report.drop(columns=["urls"])
                    st.dataframe(severity_style(display), use_container_width=True)

                # Build one concrete fix plan per keyword from the real per-URL metrics.
                plans: dict[str, tuple] = {}
                groq_budget = 15  # cap Groq calls to the worst offenders
                with st.spinner("Building per-keyword fix plans…"):
                    for _, r in report.iterrows():
                        kw = r["keyword"]
                        sub = df[df["query"] == kw].groupby("page").agg(
                            clicks=("clicks", "sum"), impressions=("impressions", "sum"),
                            position=("position", "mean")).reset_index()
                        spread, impr = float(r["position_spread"]), int(r["impressions"])
                        plan = None
                        if use_ai and client and groq_budget > 0:
                            plan = groq_fix_plan(client, model, _gsc_plan_context(kw, sub, spread, impr))
                            groq_budget -= 1
                        if not plan:
                            plan = build_gsc_fix_plan(kw, sub, spread, impr)
                        plans[kw] = (sub, plan)

                st.markdown("### Affected keywords")
                st.caption("Each keyword lists the URLs competing for it — **click a URL to see the checklist for that page.**")
                for i, (_, r) in enumerate(report.iterrows()):
                    sub, plan = plans[r["keyword"]]
                    ranked = sub.sort_values("position").reset_index(drop=True)
                    winner = ranked.iloc[0].to_dict() if not ranked.empty else None
                    with st.container(border=True):
                        st.markdown(
                            f"#### [{r['severity']}] {r['keyword']} — {r['competing_urls']} URLs · "
                            f"{r['impressions']:,} impr · pos {r['avg_position']}"
                        )
                        # Why the keyword is cannibalized (shared across its URLs).
                        render_diagnosis(plan)
                        # Per-link checklists: click a URL to open its own to-do list.
                        st.markdown("**🔗 Competing pages — click one for its checklist:**")
                        for _, s in ranked.iterrows():
                            row = s.to_dict()
                            is_winner = winner is not None and str(row["page"]) == str(winner["page"])
                            role, actions = build_gsc_link_checklist(r["keyword"], row, winner, is_winner)
                            meta = (f"**{int(row['clicks']):,}** clicks · **{int(row['impressions']):,}** impr · "
                                    f"pos **{float(row['position']):.1f}**")
                            render_link_checklist(str(row["page"]), role, actions, meta_md=meta)

                counts = _sev_counts(report)
                summary = [
                    f"Cannibalized keywords: {len(report)}",
                    f"Critical: {counts['Critical']}   High: {counts['High']}   "
                    f"Medium: {counts['Medium']}   Low: {counts['Low']}",
                    f"Impressions affected: {int(report['impressions'].sum()):,}",
                    f"Clicks affected: {int(report['clicks'].sum()):,}",
                ]
                sections = [
                    {
                        "heading": f"[{r['severity']}] {r['keyword']} — {r['competing_urls']} URLs · "
                                   f"{r['impressions']:,} impr · pos {r['avg_position']}",
                        "severity": r["severity"],
                        "urls": list(r["urls"]),
                        "recommendation": "",
                        "diagnosis": plans[r["keyword"]][1].get("diagnosis", []),
                        "checklist": plans[r["keyword"]][1].get("checklist", {}),
                    }
                    for _, r in report.iterrows()
                ]
                df_download_buttons(report, "cannibalization_gsc_report",
                                    pdf_sections=sections,
                                    pdf_title="Keyword Cannibalization — Search Console Report",
                                    pdf_summary=summary)

 

