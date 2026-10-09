"""Новости российских изданий из CC-News (HuggingFace ruggsea/infini-news-corpus) → data/news/.

Корпус — 1,8 ТБ parquet (≈30 МБ на файл, одна row group, строки не упорядочены по сайту), поэтому
файлы целиком не качаются: по футеру parquet считаются смещения колонок и HTTP range-запросами берутся
только нужные (url, сайт, дата, заголовок, лид, язык, тема IPTC) — ≈3–4 МБ на файл вместо ≈30.
Колонка text (≈80% объёма) не читается: для признаков хватает заголовка и лида.

Файлы корпуса разбиты по месяцу краулинга, а не публикации, поэтому берутся и соседние месяцы,
а статьи отбираются по publish_date.

Результат:
  data/news/infini_ru_raw/<год-месяц>/<файл>.parquet — русскоязычные статьи изданий из HOSTS (заголовок, лид, тема IPTC);
  data/news/infini_ru_census/<год-месяц>/<файл>.parquet — число русскоязычных статей по сайту и дате
    публикации для всех сайтов (чтобы расширить список изданий без повторной загрузки).
Готовые файлы пропускаются, поэтому прерванный прогон можно просто перезапустить.

Запуск (≈2,8 МБ трафика на файл, ≈15 тыс. файлов; на медленном канале удобнее
notebooks/news_infini_colab.ipynb — там же, в Colab, и готовый файл на скачивание):
  .venv/bin/python -m src.data.news_infini                       # краул-месяцы 2022-12…2025-01
  .venv/bin/python -m src.data.news_infini --no-description      # только заголовки, трафик ≈2,7× меньше
  .venv/bin/python -m src.data.news_infini --sample 0.5          # половина файлов корпуса
  .venv/bin/python -m src.data.news_infini --months 2024-03 --limit 5 --workers 8   # проба
  .venv/bin/python -m src.data.news_infini --collect             # склеить в data/news/infini_ru.parquet
Файлы обходятся в перемешанном порядке: остановленный на середине прогон — равномерная выборка по месяцам.
Прогон останавливается, если на диске осталось меньше MIN_FREE_GB.
Токен HF_TOKEN берётся из окружения или из файла .env / ,env в корне.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

REPO = "ruggsea/infini-news-corpus"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "news"

# Федеральные издания и крупные региональные сети. Ключ — сайт без www./m.
HOSTS = {
    "ria.ru": "РИА Новости", "tass.ru": "ТАСС", "interfax.ru": "Интерфакс", "lenta.ru": "Лента.ру",
    "rbc.ru": "РБК", "kommersant.ru": "Коммерсантъ", "vedomosti.ru": "Ведомости", "iz.ru": "Известия",
    "rg.ru": "Российская газета", "gazeta.ru": "Газета.ру", "kp.ru": "Комсомольская правда",
    "mk.ru": "Московский комсомолец", "aif.ru": "Аргументы и факты", "1prime.ru": "Прайм",
    "regnum.ru": "Regnum", "news.ru": "News.ru", "russian.rt.com": "RT", "life.ru": "Life",
    "vz.ru": "Взгляд", "360.ru": "360", "360tv.ru": "360", "m24.ru": "Москва 24", "tvzvezda.ru": "Звезда",
    "news.mail.ru": "Новости Mail", "news.rambler.ru": "Рамблер/новости", "banki.ru": "Банки.ру",
    "frankmedia.ru": "Frank Media", "rbc.ru/rbcfreenews": "РБК",
    # региональные
    "ura.news": "URA.RU", "fontanka.ru": "Фонтанка", "ngs.ru": "НГС", "e1.ru": "E1", "74.ru": "74.ru",
    "59.ru": "59.ru", "163.ru": "163.ru", "72.ru": "72.ru", "29.ru": "29.ru", "v1.ru": "V1",
    "ngs24.ru": "НГС24", "ngs42.ru": "НГС42", "sibkray.ru": "Сибкрай", "business-gazeta.ru": "БИЗНЕС Online",
    "realnoevremya.ru": "Реальное время", "vl.ru": "VL.ru", "znak.com": "Znak",
}

# Колонки: [url, url_hostname] в начале файла, [publish_date … description] перед text, остальное — в хвосте
COLS = ["url_hostname", "publish_date", "title", "description", "lang", "iptc_topic", "iptc_score"]
GAP = 32 * 1024  # соседние нужные колонки с промежутком меньше GAP читаются одним запросом
FOOTER_GUESS = 64 * 1024
MIN_FREE_GB = 10  # остановиться, если на диске осталось меньше


def hf_token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for name in (".env", ",env"):
        p = ROOT / name
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    return None


def session() -> requests.Session:
    s = requests.Session()
    tok = hf_token()
    if tok:
        s.headers["Authorization"] = f"Bearer {tok}"
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=3))
    return s


S = session()


def norm_host(h: str) -> str:
    h = (h or "").lower()
    for p in ("www.", "m."):
        if h.startswith(p):
            h = h[len(p):]
    return h


def list_files(crawl_months: list[str]) -> list[dict]:
    out = []
    for ym in crawl_months:
        y, m = ym.split("-")
        u = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data/year={y}/month={m}"
        while u:
            r = S.get(u, timeout=60)
            r.raise_for_status()
            out += [{"path": f["path"], "size": f["size"], "ym": ym} for f in r.json() if f["type"] == "file"]
            u = r.links.get("next", {}).get("url")
    return out


CDN = requests.Session()  # подписанные ссылки CDN — без токена
CDN.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))


def _retry(fn, what: str):
    """Повторы при сетевых ошибках, 429 (ждём Retry-After — лимит запросов к хабу) и 5xx."""
    for attempt in range(8):
        try:
            r = fn()
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 30)) + 1)
                continue
            if r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 7:
                raise
            time.sleep(2 ** min(attempt, 5))
    raise requests.HTTPError(f"{what}: слишком много 429")


def resolve(path: str) -> str:
    """Один запрос к хабу на файл: прямая ссылка на CDN, дальше range-запросы идут туда."""
    u = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    r = _retry(lambda: S.get(u, headers={"Range": "bytes=0-0"}, allow_redirects=False, timeout=60), path)
    return r.headers.get("Location") or u


def get_range(url: str, start: int, end: int) -> bytes:
    """Байты [start, end)."""
    r = _retry(lambda: CDN.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120), url)
    if len(r.content) != end - start:
        raise requests.HTTPError(f"short read {len(r.content)} != {end - start}")
    return r.content


class Sparse(io.RawIOBase):
    """Файл из заранее скачанных кусков: pyarrow читает только их."""

    def __init__(self, size: int, parts: list[tuple[int, bytes]]):
        self.size, self.parts, self.pos = size, parts, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        for s, b in self.parts:
            if s <= self.pos and self.pos + n <= s + len(b):
                out = b[self.pos - s:self.pos - s + n]
                self.pos += n
                return out
        raise IOError(f"range {self.pos}+{n} not prefetched")

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[:len(d)] = d
        return len(d)


def read_file(f: dict, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Только колонки cols одного файла: футер + по запросу на группу соседних колонок."""
    url = resolve(f["path"])
    size = f["size"]
    tail_n = min(size, FOOTER_GUESS)
    tail = get_range(url, size - tail_n, size)
    nbytes = tail_n
    footer_len = struct.unpack("<I", tail[-8:-4])[0]
    if footer_len + 8 > tail_n:
        tail_n = footer_len + 8
        tail = get_range(url, size - tail_n, size)
        nbytes += tail_n
    tail_start = size - tail_n
    rg = pq.read_metadata(Sparse(size, [(tail_start, tail)])).row_group(0)
    need = []
    for i in range(rg.num_columns):
        c = rg.column(i)
        if c.path_in_schema.split(".")[0] in cols:
            start = c.dictionary_page_offset if c.has_dictionary_page and c.dictionary_page_offset \
                else c.data_page_offset
            need.append((start, start + c.total_compressed_size))
    merged: list[list[int]] = []
    for s, e in sorted(need):
        e = min(e, tail_start)  # конец колонки может уже лежать в скачанном хвосте
        if s >= e:
            continue
        if merged and s - merged[-1][1] <= GAP:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    parts = [(tail_start, tail)]
    for s, e in merged:
        parts.append((s, get_range(url, s, e)))
        nbytes += e - s
    # колонка, заходящая в хвост, читается одним куском: склеиваем её с хвостом
    parts.sort()
    glued = [parts[0]]
    for s, b in parts[1:]:
        ps, pb = glued[-1]
        if ps + len(pb) == s:
            glued[-1] = (ps, pb + b)
        else:
            glued.append((s, b))
    t = pq.ParquetFile(Sparse(size, glued)).read(columns=cols).to_pandas()
    return t[t["lang"].eq("rus")].drop(columns="lang"), nbytes


class DiskFull(RuntimeError):
    pass


def process(f: dict, cols: list[str], force: bool = False) -> tuple[int, int, int]:
    name = Path(f["path"]).stem + ".parquet"
    raw_p = OUT / "infini_ru_raw" / f["ym"] / name
    cen_p = OUT / "infini_ru_census" / f["ym"] / name
    if raw_p.exists() and cen_p.exists() and not force:
        return 0, 0, 0
    free_gb = shutil.disk_usage(OUT).free / 1e9
    if free_gb < MIN_FREE_GB:
        raise DiskFull(f"на диске осталось {free_gb:.1f} ГБ < {MIN_FREE_GB} ГБ")
    t, nbytes = read_file(f, cols)
    t["host"] = t["url_hostname"].map(norm_host)
    t["pub_date"] = pd.to_datetime(t["publish_date"].str.slice(0, 10), errors="coerce").dt.date
    census = t.groupby(["host", "pub_date"], dropna=False).size().rename("n").reset_index()
    keep = t[t["host"].isin(HOSTS.keys())].drop(columns=["url_hostname", "publish_date"])
    for p, d in ((raw_p, keep), (cen_p, census)):
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        d.to_parquet(tmp, index=False)
        tmp.replace(p)
    return len(keep), len(t), nbytes


def month_range(a: str, b: str) -> list[str]:
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def collect(date_from: str, date_to: str) -> None:
    raw = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "infini_ru_raw").rglob("*.parquet"))],
                    ignore_index=True)
    raw["pub_date"] = pd.to_datetime(raw["pub_date"])
    raw = raw[raw["pub_date"].between(date_from, date_to)]
    n0 = len(raw)
    # один и тот же материал краулится повторно
    raw = raw.drop_duplicates(["host", "pub_date", "title"])
    raw["outlet"] = raw["host"].map(HOSTS)
    raw = raw.sort_values("pub_date").reset_index(drop=True)
    # мелкие row group по отсортированной дате: можно читать по месяцу, не загружая всё
    # pd.read_parquet(path, filters=[("pub_date", ">=", pd.Timestamp("2024-03-01"))])
    raw.to_parquet(OUT / "infini_ru.parquet", index=False, row_group_size=50_000)
    cen = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "infini_ru_census").rglob("*.parquet"))],
                    ignore_index=True)
    cen["pub_date"] = pd.to_datetime(cen["pub_date"])
    cen = cen[cen["pub_date"].between(date_from, date_to)]
    cen = cen.groupby(["host", cen["pub_date"].dt.to_period("M").astype(str)])["n"].sum().unstack(fill_value=0)
    cen.assign(total=cen.sum(1)).sort_values("total", ascending=False).to_csv(OUT / "infini_ru_census.csv")
    print(f"{len(raw):,} статей (до дедупликации {n0:,}) → {OUT / 'infini_ru.parquet'}")
    print(raw.groupby(raw["pub_date"].dt.to_period("M")).size().to_string())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", help="краул-месяцы YYYY-MM (по умолчанию 2022-12…2025-01)")
    ap.add_argument("--limit", type=int, help="не больше N файлов на месяц (проба)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--collect", action="store_true", help="только склеить уже скачанное")
    ap.add_argument("--date-from", default="2023-01-01")
    ap.add_argument("--date-to", default="2024-12-31")
    ap.add_argument("--no-description", action="store_true", help="только заголовки: трафик ≈2,5× меньше")
    ap.add_argument("--sample", type=float, default=1.0, help="доля файлов корпуса (0–1), одна и та же при перезапуске")
    a = ap.parse_args(argv)
    cols = [c for c in COLS if not (a.no_description and c == "description")]
    if a.collect:
        return collect(a.date_from, a.date_to)
    months = a.months or month_range("2022-12", "2025-01")
    files = list_files(months)
    if a.limit:
        files = [f for ym in months for f in [x for x in files if x["ym"] == ym][:a.limit]]
    # порядок по хэшу пути: прерванный прогон — всё равно равномерная выборка по месяцам
    h = {f["path"]: int(hashlib.md5(f["path"].encode()).hexdigest(), 16) for f in files}
    files = sorted((f for f in files if h[f["path"]] % 10_000 < a.sample * 10_000), key=lambda f: h[f["path"]])
    print(f"{len(files)} файлов, {sum(f['size'] for f in files) / 1e9:.0f} ГБ в корпусе", file=sys.stderr)
    OUT.mkdir(parents=True, exist_ok=True)
    t0, done, kept, rus, nb, failed = time.time(), 0, 0, 0, 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, f, cols, a.force): f for f in files}
        for fut in as_completed(futs):
            done += 1
            try:
                k, r, b = fut.result()
                kept, rus, nb = kept + k, rus + r, nb + b
            except DiskFull as e:
                print(f"СТОП: {e}", file=sys.stderr, flush=True)
                ex.shutdown(wait=True, cancel_futures=True)
                sys.exit(2)
            except Exception as e:  # noqa: BLE001 — один битый файл не должен ронять прогон
                failed.append(futs[fut]["path"])
                print(f"ошибка {futs[fut]['path']}: {e}", file=sys.stderr)
            if done % 50 == 0 or done == len(files):
                el = time.time() - t0
                print(f"{done}/{len(files)} файлов, {nb / 1e9:.1f} ГБ, {nb / 1e6 / el:.1f} МБ/с, "
                      f"рус. статей {rus:,}, из списка {kept:,}, "
                      f"осталось ~{el / done * (len(files) - done) / 60:.0f} мин", file=sys.stderr, flush=True)
    if failed:
        print(f"не удалось: {len(failed)} файлов — перезапустите команду", file=sys.stderr)


if __name__ == "__main__":
    main()
