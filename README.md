# 🔍 Keyword Cannibalization Detector

A single-file Streamlit dashboard that finds **keyword cannibalization** two ways:

1. **Crawl a website** — enter a URL, it reads `sitemap.xml` (falling back to
   internal-link crawling), extracts each page's title / H1 / meta / H2s, and
   uses **Groq** to cluster pages competing for the same keyword intent.
2. **Import Search Console data** — upload a *query × page* CSV to detect
   keywords that rank with multiple URLs, with **clicks, impressions, CTR,
   average position**, **severity scoring**, and **exportable reports**.

## Features
- Detect keyword cannibalization (crawl + GSC)
- Identify competing URLs with direct links
- Keyword performance metrics (clicks, impressions, CTR, avg position)
- Issue severity (Critical / High / Medium / Low)
- AI fix suggestions via Groq (consolidate / redirect / canonical / differentiate)
- Export reports as CSV **and** JSON

## Run locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_groq_key_here   # optional; heuristic fallback works without it
streamlit run app/dashboard.py
```

## Deploy on Render (free)
This repo ships a [`render.yaml`](render.yaml) blueprint, so deploying is one click:

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect this repo. Render reads `render.yaml` and creates the web service.
4. (Optional) Under the service's **Environment** tab, add `GROQ_API_KEY` for
   AI clustering. Without it, the app uses a heuristic fallback.
5. Click **Apply**. Render builds and serves the app at your `*.onrender.com` URL.

> The free plan spins the app down after inactivity; the first request afterward
> takes ~30s to wake it back up.

## Deploy on Streamlit Community Cloud (free)
Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → point at
`app/dashboard.py` → add `GROQ_API_KEY` under **Secrets**. Done.

## Search Console data — two ways

**A) Upload a CSV** — any CSV with a **query** column, a **page/URL** column, and
(optionally) **clicks / impressions / CTR / position**. Column names are
auto-detected. Get a query×page export from the GSC Performance report or any SEO tool.

**B) Fetch live via the Search Console API** (service account — no browser OAuth):
1. In **Google Cloud**, create a *service account* + JSON key, and enable the **Search Console API**.
2. In **Search Console → Settings → Users and permissions**, add the service
   account's `client_email` as a **Full** or **Restricted** user of the property.
3. In the app's **Search Console CSV** tab, pick *Fetch via Search Console API*,
   upload the JSON key, enter the property (`https://example.com/` or `sc-domain:example.com`),
   choose a date range, and hit **Fetch**. It pulls query×page rows (paged, up to 25k) directly.
