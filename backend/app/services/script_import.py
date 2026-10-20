"""Text / blog URL -> scenes. Heuristic extractive summariser; optional LLM when ANTHROPIC_API_KEY is set."""
import ipaddress
import json
import re
import socket
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

from ..config import get_settings

STOP = set("a an the and or but if of to in on at for with by from as is are was were be been it this that these those i you he she we they not can will would should could have has had do does did so than then there their our your its into about more most".split())


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.title = ""
        self._in_title = False
        self.blocks: list[str] = []
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header", "aside", "noscript", "form"):
            self.skip += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header", "aside", "noscript", "form"):
            self.skip = max(0, self.skip - 1)
        if tag == "title":
            self._in_title = False
        if tag in ("p", "li", "h1", "h2", "h3", "div", "br"):
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self._buf.append(data)

    def _flush(self):
        t = " ".join("".join(self._buf).split())
        if len(t) > 40:
            self.blocks.append(t)
        self._buf = []


def _check_public(url: str) -> None:
    host = urlparse(url).hostname or ""
    if urlparse(url).scheme not in ("http", "https") or not host:
        raise ValueError("Only http(s) URLs are allowed")
    for info in socket.getaddrinfo(host, None):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("That address is not allowed")


def fetch_url_text(url: str) -> tuple[str, str]:
    current = url
    with httpx.Client(timeout=10, follow_redirects=False, headers={"User-Agent": "VideoAI-importer/1.0"}) as client:
        for _ in range(4):  # follow redirects manually so every hop is checked
            _check_public(current)
            r = client.get(current)
            if r.is_redirect and r.headers.get("location"):
                current = str(httpx.URL(current).join(r.headers["location"]))
                continue
            r.raise_for_status()
            if "html" not in r.headers.get("content-type", "") and "text" not in r.headers.get("content-type", ""):
                raise ValueError("URL is not a web page")
            p = _Text()
            p.feed(r.text[:1_500_000])
            p._flush()
            return p.title.strip(), "\n".join(p.blocks)
    raise ValueError("Too many redirects")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 4]


def heuristic_scenes(text: str, max_scenes: int = 6, max_chars: int = 260) -> list[str]:
    sents = _sentences(text)
    if not sents:
        return [text.strip()[:max_chars]] if text.strip() else []
    freq = Counter(w for s in sents for w in re.findall(r"[a-zA-Z']+", s.lower()) if w not in STOP and len(w) > 2)
    scored = [(sum(freq[w] for w in re.findall(r"[a-zA-Z']+", s.lower())) / (len(s.split()) ** 0.8), i) for i, s in enumerate(sents)]
    budget = max_scenes * 2
    keep = sorted(i for _, i in sorted(scored, reverse=True)[:budget])
    if 0 not in keep:
        keep = [0] + keep[:-1]
    chosen = [sents[i] for i in keep]
    scenes, cur = [], ""
    for s in chosen:
        if cur and len(cur) + len(s) + 1 > max_chars:
            scenes.append(cur)
            cur = ""
        cur = f"{cur} {s}".strip()
    if cur:
        scenes.append(cur)
    return [sc[:600] for sc in scenes[:max_scenes]]


def llm_scenes(text: str, max_scenes: int) -> list[str] | None:
    key = get_settings().anthropic_api_key
    if not key:
        return None
    prompt = (
        f"Turn the following content into a spoken video script of at most {max_scenes} scenes. "
        "Each scene is 1-3 short, natural sentences meant to be read aloud. Reply with ONLY a JSON array of strings.\n\n"
        + text[:12000]
    )
    try:
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={"model": get_settings().llm_model, "max_tokens": 1500, "messages": [{"role": "user", "content": prompt}]},
            timeout=40,
        )
        r.raise_for_status()
        out = r.json()["content"][0]["text"]
        arr = json.loads(out[out.index("["): out.rindex("]") + 1])
        scenes = [str(x).strip()[:900] for x in arr if str(x).strip()]
        return scenes[:max_scenes] or None
    except Exception:
        return None


def to_scenes(text: str, max_scenes: int) -> tuple[list[str], str]:
    llm = llm_scenes(text, max_scenes)
    if llm:
        return llm, "llm"
    return heuristic_scenes(text, max_scenes), "heuristic"
