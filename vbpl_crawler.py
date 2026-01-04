import os, re, time, json, hashlib
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from bs4 import BeautifulSoup

BASE = "https://vbpl.vn"

SEED_LISTS = [
  # nhóm “cốt lõi”
  {"doc_type": "Hiến pháp", "issuer_default": "Quốc hội", "list_url": "https://vbpl.vn/nganhangnhanuoc/Pages/vanban.aspx?idLoaiVanBan=15&dvid=326"},
  {"doc_type": "Bộ luật",   "issuer_default": "Quốc hội", "list_url": "https://vbpl.vn/nganhangnhanuoc/Pages/vanban.aspx?idLoaiVanBan=16&dvid=326"},
  {"doc_type": "Luật",      "issuer_default": "Quốc hội", "list_url": "https://vbpl.vn/nganhangnhanuoc/Pages/vanban.aspx?idLoaiVanBan=17&dvid=326"},
  {"doc_type": "Nghị quyết","issuer_default": "Quốc hội", "list_url": "https://vbpl.vn/nganhangnhanuoc/Pages/vanban.aspx?idLoaiVanBan=18&dvid=326"},
  {"doc_type": "Pháp lệnh", "issuer_default": "UBTVQH",   "list_url": "https://vbpl.vn/nganhangnhanuoc/Pages/vanban.aspx?idLoaiVanBan=19&dvid=326"},
  {"doc_type": "Lệnh",      "issuer_default": "Chủ tịch nước", "list_url": "https://vbpl.vn/nganhangnhanuoc/Pages/vanban.aspx?idLoaiVanBan=2&dvid=326"},
  
  # nhóm “dưới luật”
  {"doc_type": "Nghị định", "issuer_default": "Chính phủ", "list_url": "https://vbpl.vn/nganhangnhanuoc/Pages/vanban.aspx?idLoaiVanBan=20&dvid=326"},
  {"doc_type": "Quyết định","issuer_default": "Chính phủ", "list_url": "https://vbpl.vn/nganhangnhanuoc/Pages/vanban.aspx?idLoaiVanBan=21&dvid=326"},
  {"doc_type": "Thông tư",  "issuer_default": "Bộ/Ngành",  "list_url": "https://vbpl.vn/nganhangnhanuoc/Pages/vanban.aspx?idLoaiVanBan=22&dvid=326"},
  # nhóm liên tịch
  {"doc_type": "Nghị quyết liên tịch", "issuer_default": "Liên tịch", "list_url": "https://vbpl.vn/nganhangnhanuoc/Pages/vanban.aspx?idLoaiVanBan=3&dvid=326"},
  {"doc_type": "Thông tư liên tịch",   "issuer_default": "Liên tịch", "list_url": "https://vbpl.vn/nganhangnhanuoc/Pages/vanban.aspx?idLoaiVanBan=23&dvid=326"},
]

CHECKPOINT_FILE = os.getenv("VBPL_CHECKPOINT_FILE", "data/vbpl_checkpoint_vn.json")

def _atomic_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_checkpoint(seed_key: str):
    if not os.path.exists(CHECKPOINT_FILE):
        return {"page": 1, "offset": 0}
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(seed_key, {"page": 1, "offset": 0})
    except Exception:
        return {"page": 1, "offset": 0}

def save_checkpoint(seed_key: str, page: int, offset: int):
    data = {}
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[seed_key] = {"page": page, "offset": offset}
    _atomic_write_json(CHECKPOINT_FILE, data)

def set_query(url: str, **params) -> str:
    u = urlparse(url)
    q = parse_qs(u.query)
    for k, v in params.items():
        q[k] = [str(v)]
    return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(q, doseq=True), u.fragment))


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://vbpl.vn/nganhangnhanuoc/Pages/Home.aspx",
    })
    # Warm-up to get cookies/session
    try:
        s.get("https://vbpl.vn/nganhangnhanuoc/Pages/Home.aspx", timeout=10)
    except Exception:
        pass # Non-fatal if warmup fails, but good to try
    return s

@retry(
    retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError)),
    stop=stop_after_attempt(6),
    wait=wait_exponential(min=1, max=30),
)
def _fetch(session: requests.Session, url: str, timeout: int) -> str:
    r = session.get(url, timeout=timeout, allow_redirects=True)
    if r.status_code != 200:
        # Debug 404 details
        snippet = (r.text or "")[:300].replace("\n", " ")
        raise RuntimeError(f"HTTP {r.status_code} final_url={r.url} len={len(r.text)} snippet={snippet}")
    
    r.encoding = r.apparent_encoding or r.encoding
    return r.text

def strip_param(url, key):
    u = urlparse(url)
    q = parse_qs(u.query)
    q.pop(key, None)
    return urlunparse((u.scheme,u.netloc,u.path,u.params,urlencode(q,doseq=True),u.fragment))

def toggle_case_path(url):
    return url.replace("VanBan.aspx","vanban.aspx") if "VanBan.aspx" in url else url.replace("vanban.aspx","VanBan.aspx")

def _fetch_with_fallback(session: requests.Session, url: str, timeout: int) -> str:
    candidates = [
        url, 
        strip_param(url, "dvid"), 
        toggle_case_path(url), 
        strip_param(toggle_case_path(url), "dvid")
    ]
    # Remove duplicates preserving order
    candidates = list(dict.fromkeys(candidates))
    
    last_err = None
    for u in candidates:
        try:
            return _fetch(session, u, timeout)
        except Exception as e:
            last_err = e
            # If 404/500, try next candidate. checking message string is brittle but simple for now
            # strictly _fetch raises RuntimeError on non-200.
            pass
    raise last_err

def _parse_last_page(html: str) -> int:
    soup = BeautifulSoup(html, "lxml")
    pages = set()
    for a in soup.find_all("a", href=True):
        m = re.search(r"[?&]Page=(\d+)", a["href"])
        if m:
            pages.add(int(m.group(1)))
    return max(pages) if pages else 1

def _extract_item_ids(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    ids = set()

    # href query
    for a in soup.find_all("a", href=True):
        href = a["href"]
        q = parse_qs(urlparse(href).query)
        for k in ("ItemID","ItemId","itemid","itemId"):
            if k in q and q[k]:
                v = q[k][0]
                if v.isdigit():
                    ids.add(v)
                break

    # onclick/script fallbacks
    ids.update(re.findall(r"ItemID\s*=\s*(\d+)", html, flags=re.I))
    ids.update(re.findall(r"ItemID=(\d+)", html, flags=re.I))

    return sorted(ids)

def _build_toanvan_url(list_url: str, item_id: str) -> str:
    q = parse_qs(urlparse(list_url).query)
    dvid = (q.get("dvid") or ["13"])[0]
    return f"{BASE}/nganhangnhanuoc/Pages/vbpq-toanvan.aspx?ItemID={item_id}&dvid={dvid}"

def download_html(url: str, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    name = hashlib.md5(url.encode()).hexdigest() + ".html"
    path = os.path.join(save_dir, name)
    if os.path.exists(path):
        return path

    session = _make_session()
    # fulltext đôi khi chậm -> timeout lớn hơn
    html = _fetch(session, url, timeout=int(os.getenv("VBPL_FULLTEXT_TIMEOUT", "100")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path

def iter_toanvan_urls(list_url: str, seed_key: str, check_known_func=None):
    """
    Yield (page, offset, toanvan_url). Offset là index trong danh sách item_ids của page.
    - Resume từ checkpoint.
    - Early-stop nếu check_known_func(urls)->(should_stop, known_ratio) báo ratio cao nhiều trang liên tiếp.
    """
    session = _make_session()
    limit_pages = int(os.getenv("VBPL_LIMIT_PAGES", "0"))  # 0=crawl hết
    rate = float(os.getenv("VBPL_RATE_LIMIT_SEC", "0.7"))

    ck = load_checkpoint(seed_key)
    start_page = int(ck.get("page", 1))
    start_offset = int(ck.get("offset", 0))


    try:
        # Use fallback for the SEED list URL which is prone to param changes
        html0 = _fetch_with_fallback(session, list_url, timeout=int(os.getenv("VBPL_LIST_TIMEOUT", "60")))
    except Exception as e:
        print(f"Failed to fetch extracted list {list_url}: {e}")
        return

    last = _parse_last_page(html0)
    max_page = last if limit_pages <= 0 else min(last, limit_pages)
    print(f"[{seed_key}] Found {last} pages in total. Resuming page {start_page} offset {start_offset}")

    high_known_streak = 0

    for page in range(start_page, max_page + 1):
        page_url = set_query(list_url, Page=page)
        try:
            html = _fetch(session, page_url, timeout=int(os.getenv("VBPL_LIST_TIMEOUT", "60")))
            item_ids = _extract_item_ids(html)
            urls = [_build_toanvan_url(list_url, iid) for iid in item_ids]

            # early stop ratio check (batch)
            if check_known_func and urls:
                should_stop, known_ratio = check_known_func(urls)
                if known_ratio >= 0.95:
                    high_known_streak += 1
                else:
                    high_known_streak = 0
                if high_known_streak >= 5:
                    print(f"[EARLY-STOP] known_ratio>=0.95 for 5 pages. Stopping seed {seed_key}.")
                    return
                if should_stop:
                    return

            # resume offset
            offset0 = start_offset if page == start_page else 0
            if page == start_page and offset0 > 0:
                urls = urls[offset0:]

            print(f"  Fetching page {page}/{max_page}... Found {len(urls)} fulltext urls (after offset).")

            for i, u in enumerate(urls, start=offset0):
                yield page, i, u

            start_offset = 0
            time.sleep(rate)
        except Exception as e:
             print(f"Error iterating page {page}: {e}")
             time.sleep(5)
