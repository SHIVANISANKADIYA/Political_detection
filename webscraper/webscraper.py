#!/usr/bin/env python3
"""
Indian Express - Political Pulse scraper
Saves articles to a JSONL file with fields: url, title, date, author, body, tags (if any)
"""
import requests
from bs4 import BeautifulSoup
import time, random, json, re, os, argparse, logging
from urllib.parse import urljoin
from tqdm import tqdm

BASE_SECTION = "https://indianexpress.com/section/political-pulse/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PoliticalPulseScraper/1.0; +https://example.com/bot)",
    "Accept-Language": "en-US,en;q=0.9"
}

BODY_SELECTORS = [
    'script[type="application/ld+json"]',
    'div[itemprop="articleBody"]',
    'div.article-content',
    'div.full-details',
    'div.story-details',
    'div.articleBody',
    'div.articleText',
    'article',
    'div#content',
    'div#maincontent'
]

AUTHOR_SELECTORS = [
    'meta[name="author"]',
    'meta[property="author"]',
    'a[rel="author"]',
    '.auth-nm',
    '.author',
    '.byline',
]

DATE_SELECTORS = [
    'meta[property="article:published_time"]',
    'meta[name="ptime"]',
    'time',
    '.dateline',
    '.date',
]

# polite defaults
MIN_DELAY = 1.0
MAX_DELAY = 2.5
RETRY_DELAY = 3.0
MAX_RETRIES = 3

# helper functions
def request_get(url, timeout=12, session=None, retries=MAX_RETRIES):
    s = session or requests
    attempt = 0
    while attempt < retries:
        try:
            r = s.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            elif r.status_code in (429, 503):
                # too many requests / service unavailable -> backoff
                sleep_t = RETRY_DELAY * (attempt + 1)
                logging.warning("Status %s for %s — sleeping %ss", r.status_code, url, sleep_t)
                time.sleep(sleep_t)
            else:
                logging.warning("Non-OK status %s for %s", r.status_code, url)
                return r
        except requests.RequestException as e:
            logging.warning("Request error %s (attempt %d) for %s", e, attempt + 1, url)
            time.sleep(RETRY_DELAY * (attempt + 1))
        attempt += 1
    return None

def extract_json_ld(soup):
    """Parse JSON-LD blocks and return a dict for NewsArticle/Article if found."""
    scripts = soup.find_all("script", type="application/ld+json")
    for s in scripts:
        text = s.string
        if not text:
            continue
        # some pages have multiple JSON-LD objects or invalid trailing commas; try safe parse
        try:
            loaded = json.loads(text.strip())
        except Exception:
            # try to be forgiving: sometimes content is multiple JSON objects; try to extract the first {...}
            try:
                # naive: find first { and last } pair
                m = re.search(r'\{.*\}', text, flags=re.S)
                if m:
                    loaded = json.loads(m.group(0))
                else:
                    continue
            except Exception:
                continue
        # loaded can be a list or single dict
        candidates = loaded if isinstance(loaded, list) else [loaded]
        for obj in candidates:
            if not isinstance(obj, dict):
                continue
            t = obj.get("@type") or obj.get("type") or obj.get("headline")
            # handle arrays or names like ["NewsArticle", ...]
            if isinstance(t, list):
                ok = any('NewsArticle' in x or 'Article' in x for x in t)
            else:
                ok = (t and (('NewsArticle' in str(t)) or ('Article' in str(t))))
            if ok or ("articleBody" in obj) or ("headline" in obj):
                return obj
    return None

def get_text_from_element(el):
    if not el:
        return ""
    # gather paragraphs only for better structure
    pars = el.find_all('p')
    if pars:
        texts = [p.get_text(" ", strip=True) for p in pars if p.get_text(strip=True)]
        return "\n\n".join(texts).strip()
    # fallback to whole element text
    return el.get_text(" ", strip=True).strip()

def extract_body(soup, json_ld=None):
    # 1) If JSON-LD has articleBody use it
    if json_ld:
        ab = json_ld.get("articleBody") or json_ld.get("description")
        if ab and isinstance(ab, str) and len(ab.strip())>50:
            return ab.strip()

    # 2) Try selectors for article body
    for sel in BODY_SELECTORS[1:]:
        el = soup.select_one(sel)
        if el:
            text = get_text_from_element(el)
            if text and len(text) > 50:
                return text

    # 3) As fallback, collect big <p> blocks under main content/article tags
    article_tag = soup.find('article')
    if article_tag:
        text = get_text_from_element(article_tag)
        if text and len(text) > 50:
            return text

    # 4) fallback: use all <p> within main content area
    body_candidates = []
    for parent in soup.select('div'):
        # heuristic: parent that contains many <p> children
        ps = parent.find_all('p')
        if len(ps) >= 3:
            t = "\n\n".join(p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True))
            if len(t) > 50:
                body_candidates.append((len(t), t))
    if body_candidates:
        # pick largest candidate
        body_candidates.sort(reverse=True)
        return body_candidates[0][1]

    return ""

def extract_title(soup, json_ld=None):
    # try JSON-LD
    if json_ld:
        title = json_ld.get("headline") or json_ld.get("name")
        if title:
            return re.sub(r'\s+', ' ', title).strip()
    # try h1
    h1 = soup.find('h1')
    if h1 and h1.get_text(strip=True):
        return h1.get_text(" ", strip=True)
    # fallback to meta title
    m = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name':'title'})
    if m and m.get('content'):
        return m.get('content').strip()
    # last resort: <title> tag
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""

def extract_author(soup, json_ld=None):
    # JSON-LD author might be dict or list
    if json_ld:
        author = json_ld.get("author")
        if isinstance(author, dict):
            name = author.get("name")
            if name:
                return name.strip()
        if isinstance(author, list) and author:
            first = author[0]
            if isinstance(first, dict) and first.get("name"):
                return first.get("name").strip()
            if isinstance(first, str):
                return first.strip()
    # try meta author
    m = soup.find('meta', attrs={'name':'author'})
    if m and m.get('content'):
        return m.get('content').strip()
    # try rel=author link
    a = soup.find('a', rel='author')
    if a and a.get_text(strip=True):
        return a.get_text(strip=True)
    # try elements with 'author' in class name
    auth = soup.find(attrs={"class": re.compile(r".*\bauthor\b.*", re.I)})
    if auth and auth.get_text(strip=True):
        return auth.get_text(" ", strip=True)
    # fallback empty
    return ""

def extract_date(soup, json_ld=None):
    if json_ld:
        dp = json_ld.get("datePublished") or json_ld.get("dateCreated") or json_ld.get("pubDate")
        if dp:
            return dp.strip()
    # meta property
    m = soup.find('meta', property='article:published_time') or soup.find('meta', attrs={'name':'ptime'})
    if m and m.get('content'):
        return m.get('content').strip()
    # <time> tag
    t = soup.find('time')
    if t and t.get('datetime'):
        return t.get('datetime').strip()
    if t and t.get_text(strip=True):
        return t.get_text(" ", strip=True)
    # look for dateline pattern near top
    top_text = " ".join([x.get_text(" ", strip=True) for x in soup.find_all(['p','div'], limit=20)])
    m = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b.*\d{4}', top_text)
    if m:
        return m.group(0)
    return ""

def scrape_article(url, session=None):
    r = request_get(url, session=session)
    if not r or r.status_code != 200:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    json_ld = extract_json_ld(soup)
    title = extract_title(soup, json_ld=json_ld)
    date = extract_date(soup, json_ld=json_ld)
    author = extract_author(soup, json_ld=json_ld)
    body = extract_body(soup, json_ld=json_ld)
    # tags/categories if available in JSON-LD or page
    tags = []
    if json_ld:
        kws = json_ld.get("keywords") or json_ld.get("about")
        if isinstance(kws, str):
            tags = [x.strip() for x in re.split(r'[;,]', kws) if x.strip()]
        elif isinstance(kws, list):
            tags = [str(x).strip() for x in kws if str(x).strip()]
    # fallback: meta keywords
    mk = soup.find('meta', attrs={'name': 'keywords'})
    if mk and mk.get('content') and not tags:
        tags = [x.strip() for x in mk.get('content').split(',') if x.strip()]
    # clean body minimally
    body = re.sub(r'\n{3,}', '\n\n', body).strip()
    return {
        # "url": url,
        "title": title,
        # "date": date,
        # "author": author,
        "body": body
    }

def gather_article_links_from_listing(listing_html, base_url):
    """Return a list of article URLs found in this listing page HTML."""
    soup = BeautifulSoup(listing_html, "html.parser")
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        # include only full article urls in Political Pulse section (pattern observed)
        if '/article/political-pulse/' in href:
            full = urljoin(base_url, href)
            links.add(full)
    return sorted(links)

def save_jsonl(filename, record):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_seen_urls(filename):
    seen = set()
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("url"):
                        seen.add(obj["url"])
                except Exception:
                    continue
    return seen

def main(max_pages=200, out_file="articles.jsonl"):
    session = requests.Session()
    seen = load_seen_urls(out_file)
    collected = 0
    pbar = tqdm(range(1, max_pages+1), desc="Pages")
    for page in pbar:
        if page == 1:
            url = BASE_SECTION
        else:
            url = urljoin(BASE_SECTION, f"page/{page}/")
        pbar.set_postfix(page=page)
        r = request_get(url, session=session)
        if not r or r.status_code != 200:
            logging.info("Stopping: cannot fetch listing page %s (status=%s)", url, getattr(r,"status_code",None))
            break
        links = gather_article_links_from_listing(r.text, BASE_SECTION)
        if not links:
            logging.info("No article links found on listing page %d — assuming end", page)
            break

        new_links = [u for u in links if u not in seen]
        if not new_links:
            # likely repeating older pages; continue but we can stop early if desired
            logging.info("No NEW links on page %d", page)
        for article_url in new_links:
            # polite delay
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            try:
                article = scrape_article(article_url, session=session)
                if article and article.get("body"):
                    save_jsonl(out_file, article)
                    seen.add(article_url)
                    collected += 1
                    print(f"[SAVED] {article['title'][:80]} -> {article_url}")
                else:
                    print(f"[SKIP] empty content: {article_url}")
            except Exception as e:
                logging.exception("Failed scraping %s: %s", article_url, e)
        # optional: stop if we've collected many articles (uncomment if needed)
        # if collected > 5000:
        # break
    print(f"Done. Collected {collected} new articles. Output file: {out_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=200, help="max listing pages to crawl")
    ap.add_argument("--out", type=str, default="articles.jsonl", help="output JSONL file")
    args = ap.parse_args()
    main(max_pages=args.max_pages, out_file=args.out)