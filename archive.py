import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import quote, unquote, urlencode, urlparse

import requests
from lxml import etree
from PIL import Image
from playwright.async_api import TimeoutError as PWTimeoutError
from playwright.async_api import async_playwright
from tqdm import tqdm

CONFIG_PATH = os.environ.get("ARCHIVE_CONFIG", "config.json")

DEFAULT_HTTP_USER_AGENT = "cli-website-archiver/1.0"

SKIP_EXT_RE = re.compile(r"\.(pdf|xml|jpg|jpeg|png|gif|svg|webp|zip|rar|7z|docx?|xlsx?|pptx?)$", re.I)


@dataclass
class Config:
    out_dir: str
    domain: Optional[str]
    sitemap_urls: List[str]
    skip_path_prefixes: List[str]

    concurrency: int
    nav_timeout_ms: int
    post_idle_wait_ms: int
    scroll_wait_ms: int

    viewport_width: int
    viewport_height: int
    user_agent: Optional[str]

    accept_cookies: bool
    nuke_cookie_banner: bool
    consent_accept_selector: str
    consent_modal_selector: str

    save_html: bool
    optimize_png: bool
    png_opt_level: int
    png_max_concurrent_opt: int

    max_urls: Optional[int]
    retries: int
    skip_existing: bool

    index_path: str


@dataclass
class Stats:
    processed: int = 0
    success: int = 0
    skipped: int = 0
    failed: int = 0
    optimized: int = 0
    opt_failed: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_config(path: str = CONFIG_PATH) -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: {path}. Copy config.example.json to config.json or set ARCHIVE_CONFIG."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return Config(
        out_dir=raw.get("out_dir", "archive_out"),
        domain=raw.get("domain"),
        sitemap_urls=raw.get("sitemap_urls", []),
        skip_path_prefixes=raw.get("skip_path_prefixes", []),
        concurrency=int(raw.get("concurrency", 6)),
        nav_timeout_ms=int(raw.get("nav_timeout_ms", 45000)),
        post_idle_wait_ms=int(raw.get("post_idle_wait_ms", 500)),
        scroll_wait_ms=int(raw.get("scroll_wait_ms", 600)),
        viewport_width=int(raw.get("viewport_width", 1920)),
        viewport_height=int(raw.get("viewport_height", 1080)),
        user_agent=raw.get("user_agent"),
        accept_cookies=bool(raw.get("accept_cookies", True)),
        nuke_cookie_banner=bool(raw.get("nuke_cookie_banner", False)),
        consent_accept_selector=str(raw.get("consent_accept_selector", "")),
        consent_modal_selector=str(raw.get("consent_modal_selector", "")),
        save_html=bool(raw.get("save_html", False)),
        optimize_png=bool(raw.get("optimize_png", True)),
        png_opt_level=int(raw.get("png_opt_level", 9)),
        png_max_concurrent_opt=int(raw.get("png_max_concurrent_opt", 2)),
        max_urls=raw.get("max_urls"),
        retries=int(raw.get("retries", 2)),
        skip_existing=bool(raw.get("skip_existing", True)),
        index_path=raw.get("index_path", "_logs/filename_url_index.json"),
    )


def normalize_domain_to_base_url(domain_or_url: str) -> str:
    value = domain_or_url.strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        return f"{parsed.scheme}://{parsed.netloc}"
    return f"https://{value}"


def discover_sitemaps_from_robots(domain_or_url: str, timeout: int = 20) -> List[str]:
    base_url = normalize_domain_to_base_url(domain_or_url)
    if not base_url:
        return []

    robots_url = f"{base_url.rstrip('/')}/robots.txt"
    resp = requests.get(robots_url, timeout=timeout, headers={"User-Agent": DEFAULT_HTTP_USER_AGENT})
    if resp.status_code >= 400:
        return []

    sitemaps: List[str] = []
    for line in resp.text.splitlines():
        m = re.match(r"^\s*Sitemap\s*:\s*(\S+)\s*$", line, flags=re.I)
        if m:
            sitemaps.append(m.group(1).strip())

    if not sitemaps:
        sitemaps.append(f"{base_url.rstrip('/')}/sitemap.xml")

    return sitemaps


def safe_slug_from_url(url: str) -> str:
    u = urlparse(url)
    host = u.netloc.lower() or "nohost"
    path = (u.path or "/").strip("/") or "home"
    query = u.query.strip()

    base = f"{host}__{path}"
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)

    if query:
        q = re.sub(r"[^a-zA-Z0-9._-]+", "_", query)
        base = f"{base}__q_{q}"

    if len(base) > 180:
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        base = base[:160] + "__" + h

    return base


def domain_key(url: str) -> str:
    return (urlparse(url).netloc or "nohost").lower()


def fetch_xml(url: str, timeout: int = 30) -> bytes:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": DEFAULT_HTTP_USER_AGENT})
    r.raise_for_status()
    return r.content


def parse_sitemap(xml_bytes: bytes) -> Tuple[List[str], List[str]]:
    root = etree.fromstring(xml_bytes)
    tag = etree.QName(root).localname.lower()

    child_sitemaps: List[str] = []
    urls: List[str] = []

    if tag == "sitemapindex":
        for loc in root.findall(".//{*}sitemap/{*}loc"):
            if loc.text:
                child_sitemaps.append(loc.text.strip())
    elif tag == "urlset":
        for loc in root.findall(".//{*}url/{*}loc"):
            if loc.text:
                urls.append(loc.text.strip())
    else:
        for loc in root.findall(".//{*}loc"):
            if loc.text:
                urls.append(loc.text.strip())

    return child_sitemaps, urls


def collect_urls_from_sitemaps(sitemap_urls: Iterable[str], max_urls: Optional[int] = None) -> List[str]:
    seen_sitemaps: Set[str] = set()
    seen_urls: Set[str] = set()
    queue: List[str] = list(dict.fromkeys(sitemap_urls))

    while queue:
        sm = queue.pop(0)
        if sm in seen_sitemaps:
            continue
        seen_sitemaps.add(sm)

        xml = fetch_xml(sm)
        child_sitemaps, urls = parse_sitemap(xml)

        for c in child_sitemaps:
            if c not in seen_sitemaps:
                queue.append(c)

        for u in urls:
            if u not in seen_urls:
                seen_urls.add(u)
                if max_urls is not None and len(seen_urls) >= max_urls:
                    return sorted(seen_urls)

    return sorted(seen_urls)


def should_skip_url_by_prefix(url: str, prefixes: List[str]) -> bool:
    if not prefixes:
        return False
    path = urlparse(url).path or "/"
    normalized_path = path if path.startswith("/") else f"/{path}"
    for prefix in prefixes:
        p = prefix.strip()
        if not p:
            continue
        p = p if p.startswith("/") else f"/{p}"
        if normalized_path == p or normalized_path.startswith(p.rstrip("/") + "/"):
            return True
    return False


def storage_state_path(out_dir: str, domain: str) -> str:
    return os.path.join(out_dir, "_state", f"storage_state__{domain}.json")


def expected_files(cfg: Config, out_png: str, out_html: str) -> Tuple[bool, bool]:
    ok_png = os.path.exists(out_png)
    ok_html = (not cfg.save_html) or os.path.exists(out_html)
    return ok_png, ok_html


def optimize_png_inplace(path_png: str, opt_level: int = 9) -> None:
    with Image.open(path_png) as im:
        im.save(path_png, format="PNG", optimize=True, compress_level=opt_level)


def load_index(cfg: Config) -> Dict[str, str]:
    p = os.path.join(cfg.out_dir, cfg.index_path)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def normalize_filename_input(s: str) -> str:
    s = s.strip().strip('"').strip("'")
    base = os.path.basename(s)
    if not base.lower().endswith(".png"):
        base += ".png"
    return base


def find_existing_path_for_png(cfg: Config, filename_png: str) -> Optional[str]:
    if "__" in filename_png:
        host = filename_png.split("__", 1)[0]
        cand = os.path.join(cfg.out_dir, host, filename_png)
        if os.path.exists(cand):
            return cand

    for root, _, files in os.walk(cfg.out_dir):
        if filename_png in files:
            return os.path.join(root, filename_png)
    return None


def host_from_existing_path(cfg: Config, png_path: str) -> Optional[str]:
    try:
        rel = os.path.relpath(png_path, cfg.out_dir)
        parts = rel.split(os.sep)
        if len(parts) >= 2:
            return parts[0]
    except Exception:
        pass
    return None


def parse_slug_parts(filename_png: str) -> Tuple[Optional[str], str, Optional[Dict[str, str]]]:
    base = filename_png[:-4] if filename_png.lower().endswith(".png") else filename_png
    if "__" not in base:
        return None, base, None

    host, rest = base.split("__", 1)
    path_part = rest
    query = None

    if "__q_" in rest:
        path_part, qpart = rest.split("__q_", 1)
        tokens = qpart.split("_")
        if len(tokens) >= 2 and len(tokens) % 2 == 0:
            query = {tokens[i]: tokens[i + 1] for i in range(0, len(tokens), 2)}
        else:
            query = {"q": qpart}

    return host, path_part, query


def build_url(host: str, path_part: str, query: Optional[Dict[str, str]]) -> str:
    path = path_part.replace("_", "/").strip("/")
    url = f"https://{host}"
    if path:
        url += "/" + "/".join(quote(unquote(p), safe=":@") for p in path.split("/"))
    if query:
        url += "?" + urlencode(query, doseq=True)
    return url


async def maybe_accept_cookies(page, cfg: Config) -> bool:
    if not cfg.accept_cookies or not cfg.consent_accept_selector.strip():
        return False
    try:
        await page.wait_for_selector(cfg.consent_accept_selector, timeout=4000)
        await page.click(cfg.consent_accept_selector, timeout=4000)
        try:
            if cfg.consent_modal_selector.strip():
                await page.wait_for_selector(cfg.consent_modal_selector, state="detached", timeout=10000)
        except PWTimeoutError:
            pass
        return True
    except PWTimeoutError:
        return False


async def nuke_cookie_banner(page, cfg: Config) -> None:
    if not cfg.nuke_cookie_banner or not cfg.consent_modal_selector.strip():
        return
    await page.add_style_tag(
        content=f"""
      {cfg.consent_modal_selector}, .ot-sdk-container, .ot-overlay-backdrop {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
      }}
      body {{ overflow: auto !important; }}
    """
    )


async def trigger_lazy_load(page, cfg: Config) -> None:
    try:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(cfg.scroll_wait_ms)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(int(cfg.scroll_wait_ms * 0.6))
    except Exception:
        pass


async def prepare_for_snapshot(page, cfg: Config) -> bool:
    await page.wait_for_load_state("domcontentloaded", timeout=cfg.nav_timeout_ms)
    await page.wait_for_load_state("networkidle", timeout=cfg.nav_timeout_ms)
    await page.wait_for_timeout(cfg.post_idle_wait_ms)

    consent_clicked = await maybe_accept_cookies(page, cfg)
    await nuke_cookie_banner(page, cfg)
    await trigger_lazy_load(page, cfg)

    return consent_clicked


async def snapshot_one(context, url: str, cfg: Config, out_png: str, out_html: str) -> bool:
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=cfg.nav_timeout_ms)
        consent_clicked = await prepare_for_snapshot(page, cfg)

        await page.screenshot(path=out_png, full_page=True)

        if cfg.save_html:
            html = await page.content()
            with open(out_html, "w", encoding="utf-8") as f:
                f.write(html)

        return consent_clicked
    finally:
        await page.close()


async def worker(
    name: str,
    queue: asyncio.Queue,
    cfg: Config,
    browser,
    log_dir: str,
    stats: Stats,
    pbar,
    opt_sem: asyncio.Semaphore,
) -> None:
    contexts = {}
    errors_path = os.path.join(log_dir, f"errors_{name}.log")
    opt_errors_path = os.path.join(log_dir, f"png_opt_errors_{name}.log")

    async def get_context_for_domain(domain: str):
        if domain in contexts:
            return contexts[domain]

        ensure_dir(os.path.join(cfg.out_dir, "_state"))
        state_path = storage_state_path(cfg.out_dir, domain)

        kwargs = {"viewport": {"width": cfg.viewport_width, "height": cfg.viewport_height}}
        if cfg.user_agent:
            kwargs["user_agent"] = cfg.user_agent
        if os.path.exists(state_path):
            kwargs["storage_state"] = state_path

        context = await browser.new_context(**kwargs)
        contexts[domain] = (context, state_path)
        return contexts[domain]

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        url = item
        dom = domain_key(url)

        slug = safe_slug_from_url(url)
        dom_dir = os.path.join(cfg.out_dir, dom)
        ensure_dir(dom_dir)

        out_png = os.path.join(dom_dir, f"{slug}.png")
        out_html = os.path.join(dom_dir, f"{slug}.html")

        ok_png, ok_html = expected_files(cfg, out_png, out_html)
        already_done = ok_png and ok_html

        if cfg.skip_existing and already_done:
            async with stats.lock:
                stats.processed += 1
                stats.skipped += 1
                pbar.update(1)
            queue.task_done()
            continue

        (context, state_path) = await get_context_for_domain(dom)

        ok = False
        last_err = None
        consent_clicked_any = False

        for attempt in range(cfg.retries + 1):
            try:
                consent_clicked = await snapshot_one(context, url, cfg, out_png, out_html)
                consent_clicked_any = consent_clicked_any or consent_clicked
                ok = True
                break
            except Exception as e:
                last_err = repr(e)
                await asyncio.sleep(0.8 + attempt * 0.7)

        if not ok:
            with open(errors_path, "a", encoding="utf-8") as f:
                f.write(f"{url}\t{last_err}\n")
        else:
            if cfg.optimize_png and os.path.exists(out_png):
                try:
                    async with opt_sem:
                        await asyncio.to_thread(optimize_png_inplace, out_png, cfg.png_opt_level)
                    async with stats.lock:
                        stats.optimized += 1
                except Exception as e:
                    with open(opt_errors_path, "a", encoding="utf-8") as f:
                        f.write(f"{url}\t{repr(e)}\n")
                    async with stats.lock:
                        stats.opt_failed += 1

        if consent_clicked_any:
            try:
                await context.storage_state(path=state_path)
            except Exception:
                pass

        async with stats.lock:
            stats.processed += 1
            if ok:
                stats.success += 1
            else:
                stats.failed += 1
            pbar.update(1)

        queue.task_done()

    for (context, _) in contexts.values():
        try:
            await context.close()
        except Exception:
            pass


def resolve_targets_for_url(cfg: Config, url: str) -> Tuple[str, str]:
    dom = domain_key(url)
    slug = safe_slug_from_url(url)
    dom_dir = os.path.join(cfg.out_dir, dom)
    ensure_dir(dom_dir)
    png_path = os.path.join(dom_dir, f"{slug}.png")
    html_path = os.path.join(dom_dir, f"{slug}.html")
    return png_path, html_path


def resolve_url_and_targets_from_filename(cfg: Config, filename_input: str, index: Dict[str, str]) -> Tuple[str, str, str]:
    filename_png = normalize_filename_input(filename_input)

    existing_png_path = find_existing_path_for_png(cfg, filename_png)
    if existing_png_path:
        png_path = existing_png_path
        html_path = os.path.join(os.path.dirname(png_path), os.path.splitext(filename_png)[0] + ".html")
        host_hint = host_from_existing_path(cfg, png_path)
    else:
        host_hint = None
        if "__" in filename_png:
            host_hint = filename_png.split("__", 1)[0]
        dom_dir = os.path.join(cfg.out_dir, host_hint or "nohost")
        ensure_dir(dom_dir)
        png_path = os.path.join(dom_dir, filename_png)
        html_path = os.path.join(dom_dir, os.path.splitext(filename_png)[0] + ".html")

    url = index.get(filename_png)
    if not url:
        host_from_name, path_part, query = parse_slug_parts(filename_png)
        host = host_hint or host_from_name or "nohost"
        url = build_url(host, path_part, query)

    return url, png_path, html_path


async def run_archive_flow(cfg: Config) -> None:
    ensure_dir(cfg.out_dir)
    log_dir = os.path.join(cfg.out_dir, "_logs")
    ensure_dir(log_dir)

    sitemap_sources = list(cfg.sitemap_urls)
    if cfg.domain:
        sitemap_sources.extend(discover_sitemaps_from_robots(cfg.domain))
    sitemap_sources = list(dict.fromkeys(sitemap_sources))

    if not sitemap_sources:
        raise SystemExit("No sitemap source found. Set domain and/or sitemap_urls in config.json.")

    urls = collect_urls_from_sitemaps(sitemap_sources, max_urls=cfg.max_urls)
    urls = [u for u in urls if not SKIP_EXT_RE.search(u)]
    urls = [u for u in urls if not should_skip_url_by_prefix(u, cfg.skip_path_prefixes)]

    with open(os.path.join(log_dir, "urls.json"), "w", encoding="utf-8") as f:
        json.dump(urls, f, ensure_ascii=False, indent=2)

    total = len(urls)
    stats = Stats()

    q: asyncio.Queue = asyncio.Queue()
    for u in urls:
        await q.put(u)

    for _ in range(cfg.concurrency):
        await q.put(None)

    pbar = tqdm(total=total, desc="Snapshot", unit="url", dynamic_ncols=True)
    opt_sem = asyncio.Semaphore(max(1, cfg.png_max_concurrent_opt))

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        tasks = [
            asyncio.create_task(worker(f"w{i + 1}", q, cfg, browser, log_dir, stats, pbar, opt_sem))
            for i in range(cfg.concurrency)
        ]

        await q.join()
        await asyncio.gather(*tasks)

        await browser.close()

    pbar.close()

    missing = 0
    missing_urls: List[str] = []

    for u in urls:
        dom = domain_key(u)
        slug = safe_slug_from_url(u)
        dom_dir = os.path.join(cfg.out_dir, dom)
        out_png = os.path.join(dom_dir, f"{slug}.png")
        out_html = os.path.join(dom_dir, f"{slug}.html")

        ok_png, ok_html = expected_files(cfg, out_png, out_html)
        if not (ok_png and ok_html):
            missing += 1
            missing_urls.append(u)

    if missing_urls:
        with open(os.path.join(log_dir, "missing_urls.json"), "w", encoding="utf-8") as f:
            json.dump(missing_urls, f, ensure_ascii=False, indent=2)

    report = {
        "sitemap_sources": sitemap_sources,
        "total_urls": total,
        "processed": stats.processed,
        "success": stats.success,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "png_optimized": stats.optimized,
        "png_opt_failed": stats.opt_failed,
        "save_html": cfg.save_html,
        "optimize_png": cfg.optimize_png,
        "missing_files": missing,
        "complete": (missing == 0),
        "output_dir": os.path.abspath(cfg.out_dir),
        "logs_dir": os.path.abspath(log_dir),
    }

    with open(os.path.join(log_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Snapshot Report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed"] > 0:
        print(f"\nError details: {log_dir}/errors_w*.log")
    if report["png_opt_failed"] > 0:
        print(f"\nPNG optimization errors: {log_dir}/png_opt_errors_w*.log")
    if missing_urls:
        print(f"\nMissing URL list: {log_dir}/missing_urls.json")
    print(f"Report saved to: {log_dir}/report.json")


async def run_refresh_flow(cfg: Config) -> None:
    ensure_dir(cfg.out_dir)
    index = load_index(cfg)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        kwargs = {"viewport": {"width": cfg.viewport_width, "height": cfg.viewport_height}}
        if cfg.user_agent:
            kwargs["user_agent"] = cfg.user_agent
        context = await browser.new_context(**kwargs)

        while True:
            raw = input("\nEnter URL or PNG filename (exit to quit): ").strip()
            if raw.lower() == "exit":
                break

            if raw.startswith("http://") or raw.startswith("https://"):
                url = raw.strip()
                png_path, html_path = resolve_targets_for_url(cfg, url)
            else:
                url, png_path, html_path = resolve_url_and_targets_from_filename(cfg, raw, index)

            print(f"→ goto: {url}")
            print(f"→ png:  {png_path}")

            try:
                await snapshot_one(context, url, cfg, png_path, html_path)
                if cfg.optimize_png and os.path.exists(png_path):
                    await asyncio.to_thread(optimize_png_inplace, png_path, cfg.png_opt_level)
                print("✔ Snapshot updated")
            except Exception as e:
                print("Error:", repr(e))

        await context.close()
        await browser.close()


def choose_flow() -> str:
    print("Select mode:")
    print("1) Archive from sitemap(s)")
    print("2) Refresh individual pages (URL or filename)")

    while True:
        choice = input("Choice [1/2]: ").strip().lower()
        if choice in {"1", "archive", "a"}:
            return "archive"
        if choice in {"2", "refresh", "r"}:
            return "refresh"
        print("Invalid choice. Please enter 1 or 2.")


async def main() -> None:
    cfg = load_config(CONFIG_PATH)
    flow = choose_flow()

    if flow == "archive":
        await run_archive_flow(cfg)
    else:
        await run_refresh_flow(cfg)


if __name__ == "__main__":
    asyncio.run(main())
