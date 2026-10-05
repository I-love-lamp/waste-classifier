"""Download freely licensed sample photos from Wikimedia Commons for the app's sample gallery.

Writes images to app/samples/ and a manifest (samples.json) with attribution for each file.
Usage: python scripts/fetch_samples.py
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "waste-classifier-samples/0.1 (local dev script)"
OUT = Path(__file__).resolve().parent.parent / "app" / "samples"
ALLOWED_LICENSES = re.compile(r"^(CC0|CC BY|CC BY-SA|Public domain|PD)", re.I)

# (slug, display name, expected class, Commons file title)
SAMPLES = [
    ("banana-peel", "Banana peel", "O", "Banana peel 2.jpg"),
    ("rotten-apple", "Rotten apple", "O", "Rotten apple under the tree.jpg"),
    ("kitchen-scraps", "Kitchen scraps", "O", "Kitchen food waste, vegetable peelings & coffee grounds in biodegradable bag.jpg"),
    ("eggshell", "Broken eggshell", "O", "Broken eggshell.jpg"),
    ("mouldy-bread", "Mouldy bread", "O", "Food waste with moldy bread.jpg"),
    ("plastic-bottle", "Plastic bottle", "R", "Botella de plástico - PET.jpg"),
    ("crushed-cans", "Crushed cans", "R", "Crushed cans (9691248780).jpg"),
    ("glass-jar", "Glass jar", "R", "Empty Clear Jar (51330638811).jpg"),
    ("newspapers", "Newspapers", "R", "A stack of newspapers.jpg"),
    ("tin-can", "Tin can", "R", "Empty tin can2009-01-19.jpg"),
    ("packaging", "Food packaging", "R", "Several Cartons.jpg"),
]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"rate limited: {url}")


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def image_info(title):
    params = {"action": "query", "titles": f"File:{title}", "prop": "imageinfo",
              "iiprop": "url|extmetadata", "iiurlwidth": 640, "format": "json", "formatversion": "2"}
    page = json.loads(fetch(f"{API}?{urllib.parse.urlencode(params)}"))["query"]["pages"][0]
    if "imageinfo" not in page:
        return None
    ii = page["imageinfo"][0]
    meta = ii.get("extmetadata", {})
    lic = meta.get("LicenseShortName", {}).get("value", "")
    if not ALLOWED_LICENSES.match(lic):
        return None
    return {
        "thumb": ii["thumburl"],
        "source": ii["descriptionurl"],
        "license": lic,
        "author": strip_html(meta.get("Artist", {}).get("value", ""))[:120] or "Unknown",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for slug, name, label, title in SAMPLES:
        info = image_info(title)
        time.sleep(1)
        if not info:
            print(f"skip {slug}: missing or not freely licensed")
            continue
        path = OUT / f"{slug}.jpg"
        path.write_bytes(fetch(info["thumb"]))
        time.sleep(1)
        manifest.append({"id": slug, "name": name, "expected": label, "file": path.name,
                         "source": info["source"], "license": info["license"], "author": info["author"]})
        print(f"ok   {slug} ({info['license']})")
    (OUT / "samples.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
