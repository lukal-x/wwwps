import functools
import hashlib
import ipaddress
import itertools
import json
import multiprocessing
import operator
import random
import re
import requests
import socket
import subprocess
import time
import typing
import urllib.parse
import uvicorn
import zlib

from asgiref.wsgi import WsgiToAsgi
from bs4 import BeautifulSoup
from contextlib import suppress
from collections import deque
from dataclasses import dataclass
from flask import Flask, Response, request as req
from os.path import dirname, exists, join
from sentence_transformers import SentenceTransformer
from types import SimpleNamespace
from urllib.parse import urlparse

# from selenium import webdriver
# from selenium.common.exceptions import TimeoutException
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.expected_conditions import staleness_of
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.firefox.options import Options


cors_headers = dict(
    {
        "Access-Control-Allow-Origin": "http://localhost:4567",
        "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,PATCH,OPTIONS",
        "Access-Control-Allow-Headers": "authorization, content-type, cookie",
        "Access-Control-Allow-Credentials": "true",
    }
)


Store = typing.Literal["urls", "queries"]

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@functools.lru_cache(maxsize=9999)
def encoded(text: str):
    raise DeprecationWarning("!") # return model.encode([text], convert_to_tensor=True)


def inform(info) -> None:
    print(info)
    if False: # 30 > len(str(info)):
        subprocess.run(["espeak", str(info)])


def relative(filepath: str) -> str:
    return join(dirname(__file__), filepath)


class F:

    @staticmethod
    def ngram(text: str, size: int = 2) -> list[str]:

        grams: list[str] = [""]

        for i, el in enumerate(text):
            for j in range(i, i + size):
                if j < len(text):
                    grams[-1] += text[j]

            grams.append("")

        return grams

    @staticmethod
    def trigram(text) -> list[str]:
        return F.ngram(text, 3)

    @staticmethod
    def similarity(a: str, b: str) -> float:
        pattern = r"[^a-zA-Z0-9]"

        left, right = map(
            F.trigram,
            map(functools.partial(re.sub, pattern, " "), map(str.lower, [a, b])),
        )

        total = len(left + right)
        distinct = len(set(left + right))
        same = total - distinct

        return same / total


@dataclass
class LinkData:
    href: str
    text: str
    odds: int


class Page(typing.NamedTuple):
    current_location: str
    page_title: str
    page_metadata: str
    page_content: str
    discovered: list[LinkData]


@dataclass
class Walker:

    # runtime statistics
    stats: object

    # the actual contents
    driver: BeautifulSoup

    # ip ranges to avoid
    nogo_ranges: list[str]

    # common file extensions
    exts: dict

    # N most common TLDs
    tlds: list[str]

    # N most common words
    words: list[str]

    # locations, in memory, to look at
    locations: dict[str, LinkData]

    # N last seen locations
    seen: deque[str]

    # user query
    focus: str  # typing.Any  # tensor

    # how deep(!wide) should the scanning be
    depth: float = 0.3

    @classmethod
    def create(cls):
        ua = "Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0"

        _ = """
        options = Options()
        options.add_argument("--headless")

        options.set_preference("general.useragent.override", ua)
        options.set_preference("browser.download.folderList", 2)
        options.set_preference("javascript.enabled", False)

        driver = webdriver.Firefox(options=options)
        driver.set_page_load_timeout(20)
        """

        driver = requests.Session()
        driver.headers.update({"User-Agent": ua})

        nogo_ranges = open(relative("resources/nogo_ranges.txt")).read().splitlines()

        exts = json.loads(open(relative("resources/file-extensions.json")).read())
        tlds = open(relative("resources/tlds.txt")).read().splitlines()
        words = open(relative("resources/google-10000-en.txt")).read().splitlines()

        return cls(
            SimpleNamespace(from_storage=0, from_web=0, failed=0, total=0),
            driver,
            nogo_ranges,
            exts,
            tlds,
            words,
            {},
            deque([], maxlen=300),
            None,
        )

    def estimated(self, estimations: dict[str, LinkData] | None) -> None:
        if estimations is None:
            return None

        for url, link in estimations.items():
            self.locations[url] = link

    def is_ok(self, url: str) -> bool:
        time.sleep(random.randint(0, 24))

        ip = None
        with suppress(Exception):
            ip = ipaddress.ip_address(socket.gethostbyname(urlparse(url).netloc))

        if ip is None:
            inform(f"Moving on... No IP for {url}")
            return False

        if any(
            filter(
                lambda iprange: ip in ipaddress.ip_network(iprange),
                self.nogo_ranges,
            )
        ):
            inform(f"Moving on... {url} is in a nogo range, skipping these")
            return False

        return True

    def location(self, depth: float):
        if (1 == random.randint(0, 5)) or (not any(self.locations)):
            tld = random.choice(self.tlds)
            word = random.choice(self.words).lower()
            distance = F.similarity(self.focus, word)

            attempts = 0
            started = time.time()
            depth = self.depth
            while depth > float(distance):
                attempts += 1
                word = random.choice(self.words).lower()
                distance = F.similarity(self.focus, word)

                if attempts > 5_000:
                    depth -= 0.0001

            finished = time.time()
            inform(
                [
                    "Word ",
                    word,
                    " distance ",
                    ("%.2f" % float(distance)),
                    " attempts ",
                    attempts,
                    " seconds ",
                    round(finished - started, 3),
                ]
            )

            # if 1 == random.randint(0, 50):
            #     return f"https://wikipedia.org/wiki/{word}"

            return f"http://{word}{tld}"

        def not_pointless(e: dict) -> bool:
            if e.get("href") in self.seen:
                return False

            return e.get("odds", 0.0) > depth

        def by_odds(e: dict) -> float:
            return e["odds"]

        def by_distance(e: dict) -> float:
            if not any(self.seen):
                return 0.0

            a = e["href"]
            relevant = list(map(str, list(self.seen)[-100:]))

            scores = list(
                map(
                    float,
                    map(
                        functools.partial(F.similarity, a),
                        relevant,
                    ),
                )
            )

            chance = (random.randint(-3, 3)) / 10
            avg = sum(scores) / len(scores)

            if 0.0 == chance:
                return avg

            return avg  # * chance

        inform(["Choosing..."])
        chosen = next(
            iter(
                sorted(
                    sorted(
                        filter(not_pointless, self.locations.values()),
                        key=by_odds,
                        reverse=True,
                    ),
                    key=by_distance,
                )
            ),
            None,
        )

        if chosen is None:
            inform(["No usable locations, depth ", depth])
            return self.location(depth - 0.1)

        inform(["Chose ", chosen.get("href"), chosen.get("odds")])
        self.seen.append(chosen.get("href"))

        return chosen.get("href")

    def follow(self, url: str) -> None:
        self.soup = BeautifulSoup(
            self.driver.get(url, allow_redirects=True).content, "html.parser"
        )
        time.sleep(random.randint(0, 28))

    def collect_links(self, limit: int = 99) -> dict[str, LinkData]:
        per_netloc = {"found": {}, "collected": {}}

        links = []
        for e_idx, e in enumerate(self.soup.find_all("a")):
            netloc = urllib.parse.urlparse(e.get("href")).netloc
            per_netloc["found"][netloc] = per_netloc["found"].get(netloc, 0) + 1

        for e_idx, e in enumerate(self.soup.find_all("a")):
            if 0 != limit and e_idx > limit:
                break

            link = dict(
                href=e.get("href") or "#",
                text=e.get_text(),
                odds="?",
            )

            if not isinstance(link["href"], str):  # ??
                continue

            if link["href"] is None:
                continue

            if not link["href"].startswith("http"):
                continue

            loc, *fragment = link["href"].split("#")
            loc, *query_string = loc.split("?")
            link["href"] = loc

            if link in links:
                continue

            if link["href"] in self.seen:
                continue

            if link["href"] in self.locations.keys():
                continue

            if any(
                map(
                    lambda e: link["href"].lower().endswith("." + e.lower()),
                    filter(lambda e: e not in ["HTM", "HTML"], self.exts.keys()),
                )
            ):
                continue

            netloc = urllib.parse.urlparse(link["href"]).netloc
            ratio = per_netloc["found"].get(netloc, 0) / sum(
                per_netloc["found"].values()
            )
            if (
                10 < per_netloc["found"].get(netloc, 0)
                and (0.6 < ratio)
                and 99 < per_netloc["collected"].get(netloc, 0)
            ):
                continue

            per_netloc["collected"][netloc] = per_netloc["collected"].get(netloc, 0) + 1
            links.append(link)

        return links

    def collect_text(self, page: int):
        body = next(iter(self.soup.find_all("body")), None)
        if body is None:
            return ""

        chars_per_page = 200
        lower = (page - 1) * chars_per_page
        upper = lower + chars_per_page
        return body.get_text()[lower : lower + upper]

    def collect_metas(self):
        metadata = []
        for e in self.soup.find_all("meta"):
            if e.get("name") in ["description"]:
                metadata.append(f"""{e.get("name")}: {e.get("content")}""")

        return "; ".join(metadata)

    def step(self) -> Page:
        url = self.location(self.depth)
        inform(["Runtime statistics ", self.stats.__dict__, " seen ", len(self.seen)])
        inform(["Looking at ", url])
        while not self.is_ok(url):
            url = self.location(self.depth)
            inform(["Looking at ", url])

        inform(["Checking ", url])
        self.stats.total += 1

        stored_only = False
        while stored_only and not self.has_url_stored(url):
            inform(["Skipping... Found no storage for ", url])
            url = self.location(self.depth)

        if self.has_url_stored(url) and (1 != random.randint(0, 10)):
            inform(["Got storage for url", url])
            try:
                restored = self.from_storage(url)

                if not any(
                    filter(
                        lambda e: not e["href"].startswith("http"), restored.discovered
                    )
                ):
                    self.stats.from_storage += 1
                    return restored
            except Exception as ex:
                inform([f"Failed to read url {url} from storage", ex])

        loaded_ok = False
        while not loaded_ok:
            try:
                self.follow(url)
                loaded_ok = True
                self.stats.from_web += 1
                inform(["Loaded OK: ", url])
            except Exception as ex:
                inform(["Failed to open ", url, " got ", ex])
                self.stats.failed += 1
                url = self.location(self.depth)

        content = ""
        page = 1
        paged = "..."
        while any(paged) and page < 2:
            paged = self.collect_text(page)
            content += paged
            page += 1

        links = self.collect_links(999)
        found = Page(
            **{
                "current_location": url,
                "page_title": (
                    (self.soup.title.string or url)
                    if self.soup.title is not None
                    else f"No title: {url}"
                ),
                "page_metadata": self.collect_metas() or f"No information",
                "page_content": content,
                "discovered": links,
            }
        )

        self.to_storage(found)
        return found

    def storage_filename(self, url: str, store: Store = "urls") -> str:
        return join(
            f"storage/{store}", hashlib.md5(bytes(url, "utf-8")).hexdigest() + ".bin"
        )

    def has_url_stored(self, url: str) -> bool:
        return exists(relative(self.storage_filename(url)))

    def read_storage(self, value: str, store: Store = "urls") -> typing.Any:
        with open(relative(self.storage_filename(value, store)), "rb") as f:
            result = json.loads(zlib.decompress(f.read()).decode("utf-8"))
            inform(["Read url", value, " from storage"])

            return result

    def write_storage(self, key: str, data: dict, store: Store = "urls") -> None:
        assert isinstance(data, dict)
        with open(relative(self.storage_filename(key, store)), "wb") as f:
            f.write(zlib.compress(bytes(json.dumps(data), "utf-8")))

    def from_storage(self, url: str) -> Page:
        return Page(**self.read_storage(url, store="urls"))

    def to_storage(self, page: Page) -> None:
        assert isinstance(page, Page)
        self.write_storage(page.current_location, data=page._asdict(), store="urls")


content = {}
walkers = {}
procs = {}
queries = {}
manager = multiprocessing.Manager()
sharedresults = manager.dict()
web = Flask("walker")

form = """<form method="GET" action="/search"><input type="text" placeholder="search" name="query" /><input type="submit" /></form>"""


@web.route("/", methods=["GET"])
def home():
    return form


def crawl(query: str, depth: float | None = None, once: bool = False):

    assert query

    sharedresults.setdefault(query, [])

    if query not in walkers:
        walkers[query] = Walker.create()

    form = f"""<form method="GET" action="/search"><input type="text" placeholder="search" name="query" value="{query}"/><input type="submit" /></form>"""
    message = f"""<p>Searching for "{query}"... Refresh this page to see results & progress.</p>"""

    if query not in queries:
        queries[query] = query  # model.encode([query], convert_to_tensor=True)
        inform(["Encoded query into embeddings"])

    in_history = False
    if exists(relative("storage/queries.txt")):
        with open(relative("storage/queries.txt"), "r") as f:
            for _line in f:  # look for exact query to load its links
                if once:
                    continue

                line = _line.strip()
                if query.strip() != line:
                    continue

                in_history = True

                inform(["Found an exact query in engine history"])
                path = walkers[query].storage_filename(query, "queries")
                if not exists(relative(path)):
                    inform(["Did not find in storage ", path])
                    continue

                for k, v in walkers[query].read_storage(line, "queries").items():
                    v["odds"] = min(0.3, v["odds"])
                    walkers[query].locations[k] = v

                inform(["Locations loaded ", len(walkers[query].locations)])

            if not in_history:
                f.seek(0)

                highest = 0.0
                minimum = 0.4
                closest = {}
                for _line in f:  # look for closest match to load its links
                    line = _line.strip()
                    # vec = model.encode([line], convert_to_tensor=True)
                    similarity = round(
                        F.similarity(queries[query], line).item(),
                        4,
                    )

                    path = walkers[query].storage_filename(line, "queries")

                    if minimum < similarity and highest < similarity:
                        inform(
                            [
                                "Previous query ",
                                line,
                                " similarity ",
                                similarity,
                                " / ",
                                minimum,
                            ]
                        )
                        highest = similarity
                        if not exists(relative(path)):
                            inform(["Did not find in storage ", path])
                            continue

                        for k, v in (
                            walkers[query].read_storage(line, "queries").items()
                        ):
                            v["odds"] = min(0.1, v["odds"])
                            walkers[query].locations[k] = v

                        inform(["Locations loaded ", len(walkers[query].locations)])

                if any(closest):
                    walkers[query].locations = closest

    if not in_history:
        with open(relative("storage/queries.txt"), "a") as f:
            f.write(f"{query}\n")

    if depth is not None:
        with suppress(Exception):
            walkers[query].depth = float(depth)

    focus: str = ", ".join(
        itertools.filterfalse(
            operator.methodcaller("startswith", "not "),
            map(str.strip, query.split(",")),
        )
    )

    assert focus, focus
    walkers[query].focus = focus

    best = 0.0

    content[query] = walkers[query].step()._asdict()

    if any(walkers[query].locations):
        best = max(map(operator.itemgetter("odds"), walkers[query].locations.values()))

    level = next(
        filter(
            functools.partial(operator.lt, best),
            map(functools.partial(operator.mul, 0.1), range(18, 99)),
        )
    )

    while True:  # level > max(0.17, best):
        try:
            content[query] = walkers[query].step()._asdict()

            for link_idx, discovered in enumerate(content[query]["discovered"]):
                vals = []

                for fact in map(str.strip, query.split(",")):
                    # fact_tensor = encoded(fact)
                    # if fact.strip().startswith("not "):
                    #     fact_tensor = torch.flip(fact_tensor, [0, 1])

                    second = f"""{discovered["href"]} ({discovered["text"]}) {content[query]["page_metadata"]}"""
                    # second = model.encode([f"""{discovered["href"]} ({discovered["text"]}) {content[query]["page_metadata"]}"""], convert_to_tensor=True)
                    vals.append(round(F.similarity(fact, second), 4))

                val = 0.0
                if any(vals):
                    val = round(sum(vals) / len(vals), 4)

                inform(
                    ["Answer: ", val, " for ", discovered["href"], discovered["text"]]
                )

                content[query]["discovered"][link_idx][
                    "description"
                ] = f"""from {content[query]["current_location"]}, {content[query]["page_metadata"]}"""
                content[query]["discovered"][link_idx]["odds"] = val

            walkers[query].estimated(
                {e["href"]: e for e in content[query]["discovered"]}
            )

            if any(walkers[query].locations):
                best = max(
                    map(operator.itemgetter("odds"), walkers[query].locations.values())
                )

            inform(f"Best found so far {best}")
            inform(
                [
                    "Best found so far ",
                    best,
                    " goal is ",
                    round(level, 4),
                    " query ",
                    query,
                ]
            )
            walkers[query].write_storage(
                query, walkers[query].locations, store="queries"
            )

            level -= 0.0005  # gradually reduce goal
            walkers[query].depth -= 0.005  # gradually reduce depth

            if any(content[query]["discovered"]):
                highest_found = max(
                    map(operator.itemgetter("odds"), content[query]["discovered"])
                )
                if highest_found > walkers[query].depth:
                    walkers[query].depth = highest_found  # reset depth when necessary

            inform(["Depth ", walkers[query].depth])
        except Exception as ex:
            inform(["Got an error ", ex])
            break

        time.sleep(random.randint(0, 66))
        sharedresults[query] = sorted(
            [*sharedresults[query], *content[query].get("discovered", [])],
            key=operator.itemgetter("odds"),
            reverse=True,
        )[0:50]

        del content[query]
    #

    walkers[query].depth = (
        float(depth) if isinstance(depth, int) else 0.3
    )  # reset depth

    inform(["Best found so far ", best, " goal is ", round(level, 4), " query ", query])
    inform(["Depth ", walkers[query].depth])
    inform("\n--\n\n")


cache_build_queries = deque([], maxlen=999)


@web.route("/search", methods=["GET"])
async def search():
    query = req.args.get("query")
    depth = req.args.get("depth")
    once = bool(req.args.get("once"))

    return sharedresults.get(query, [])


@web.route("/halt", methods=["PATCH", "OPTIONS"])
async def halt():
    query = req.args.get("query")

    assert query is not None

    if query in walkers:
        del walkers[query]

    if query in content:
        del content[query]

    if query in procs:
        procs[query].terminate()
        del procs[query]

    return Response("Halted")  # , headers=cors_headers)


@web.route("/start", methods=["PATCH", "OPTIONS"])
async def start():
    query = req.args.get("query")

    assert query is not None

    if query not in walkers or walkers[query].driver is None:
        procs[query] = multiprocessing.Process(
            target=crawl, args=[query, sharedresults]
        )
        procs[query].start()

    return Response("Started")  # , headers=cors_headers)


# @web.route("/build-cache", methods=["GET"])
async def build_cache():
    """deprecated, pick apart to pre-create some data"""
    iters = 10

    with suppress(Exception):
        iters = int(req.args.get("iters"))

    query = "lorem ipsum dolor sit amet"

    if not exists(relative("storage/queries.txt")):
        with open(relative("storage/queries.txt"), "a") as f:
            f.write(f"{query}\n")

    for i in range(0, iters):
        inform(["Iteration ", i + 1, "/", iters])

        # take a random previous query, if any
        query = random.choice(
            open(relative("storage/queries.txt"), "r").readlines()
        ).strip()

        searched(query, depth=0.3, once=True)

        if query not in sharedresults:
            continue

        for content in sharedresults[query]:
            text = content.get("page_content", None)
            if text is None:
                text = content.get("page_metadata", None)
                if text is None:
                    continue

            phrases = re.split("\,|\.|\:", text)
            for phrase in phrases:
                size = len(phrase.split(" "))
                phrase = re.sub("\W", " ", phrase).strip()

                if not phrase:
                    continue

                if 4 > size:
                    continue

                if 8 < size:
                    continue

                if phrase not in open(relative("storage/queries.txt")).read():
                    inform(["Phrase ", phrase])
                    with open(relative("storage/queries.txt"), "a") as f:
                        f.write(f"{phrase}\n")

    return "Done"


# a_web = WsgiToAsgi(web)

if __name__ == "__main__":
    web.run(port=1337)
    # uvicorn.run("main:a_web", reload=False, workers=1, port=1337)
