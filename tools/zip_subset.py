import os
import sys
import io
import time
import struct
import zlib
import argparse
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
from tqdm import tqdm

COMP = "rsna-knee-abnormality-detection"
DATA_ROOT = "/mnt/ssd/rsna-knee/data"
CD_CACHE = os.path.join(DATA_ROOT, "zip_central_dir.bin")
TLS = threading.local()


def bearer():
    return {"Authorization": "Bearer " + open(os.path.expanduser("~/.kaggle/access_token")).read().strip()}


def get_signed_url():
    backoff = 5.0
    for attempt in range(10):
        r = requests.get(
            f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMP}",
            headers=bearer(), allow_redirects=False, timeout=60)
        if r.status_code == 302:
            return r.headers["Location"]
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            time.sleep(float(ra) if ra and ra.replace(".", "").isdigit() else backoff + random.uniform(0, 1))
            backoff = min(backoff * 2, 60.0)
            continue
        r.raise_for_status()
    raise RuntimeError("could not fetch signed url")


def zip64_eocd(signed):
    s = requests.Session()
    tail = s.get(signed, headers={"Range": "bytes=-262144"}, timeout=120)
    tail.raise_for_status()
    buf = tail.content
    i = buf.rfind(b"PK\x06\x06")
    if i < 0:
        raise RuntimeError("no zip64 eocd")
    cd_size = struct.unpack_from("<Q", buf, i + 40)[0]
    cd_off = struct.unpack_from("<Q", buf, i + 48)[0]
    return cd_off, cd_size


def fetch_central_dir(signed, cd_off, cd_size):
    if os.path.exists(CD_CACHE) and os.path.getsize(CD_CACHE) == cd_size:
        with open(CD_CACHE, "rb") as f:
            return f.read()
    s = requests.Session()
    t0 = time.time()
    r = s.get(signed, headers={"Range": f"bytes={cd_off}-{cd_off + cd_size - 1}"}, timeout=1800)
    r.raise_for_status()
    blob = r.content
    with open(CD_CACHE + ".part", "wb") as f:
        f.write(blob)
    os.replace(CD_CACHE + ".part", CD_CACHE)
    print(f"central dir: {len(blob)/1e6:.1f}MB in {time.time()-t0:.0f}s", flush=True)
    return blob


def parse_central_dir(blob):
    names_blob = bytearray()
    name_offs = []
    offs = []
    csizes = []
    usizes = []
    methods = []
    crcs = []
    pos = 0
    while pos + 46 <= len(blob):
        if blob[pos:pos + 4] != b"PK\x01\x02":
            raise RuntimeError(f"bad cd entry at {pos}")
        crc = struct.unpack_from("<I", blob, pos + 16)[0]
        csize = struct.unpack_from("<I", blob, pos + 20)[0]
        usize = struct.unpack_from("<I", blob, pos + 24)[0]
        method = struct.unpack_from("<H", blob, pos + 10)[0]
        fnl, exl, cml = struct.unpack_from("<HHH", blob, pos + 28)
        lho = struct.unpack_from("<I", blob, pos + 42)[0]
        name_start = pos + 46
        name_end = name_start + fnl
        extra = blob[name_end:name_end + exl]
        epos = 0
        while epos + 4 <= len(extra):
            hsig, hsize = struct.unpack_from("<HH", extra, epos)
            if hsig == 1:
                fields = []
                fp = epos + 4
                vals = []
                for cur in (usize, csize, lho):
                    if cur == 0xFFFFFFFF:
                        vals.append(struct.unpack_from("<Q", extra, fp)[0])
                        fp += 8
                    else:
                        vals.append(cur)
                usize, csize, lho = vals
                break
            epos += 4 + hsize
        names_blob += blob[name_start:name_end]
        name_offs.append(len(names_blob))
        offs.append(lho)
        csizes.append(csize)
        usizes.append(usize)
        methods.append(method)
        crcs.append(crc)
        pos = name_end + exl + cml
    names_blob = bytes(names_blob)
    return {
        "names_blob": names_blob,
        "name_offs": name_offs,
        "offs": offs,
        "csizes": csizes,
        "usizes": usizes,
        "methods": methods,
        "crcs": crcs,
        "count": len(name_offs),
    }


def session():
    if not hasattr(TLS, "s"):
        s = requests.Session()
        s.headers.update({"User-Agent": "rsna-zip-subset/1.0"})
        TLS.s = s
    return TLS.s


def fetch_member(signed, cd, idx, dest):
    off = cd["offs"][idx]
    csize = cd["csizes"][idx]
    usize = cd["usizes"][idx]
    method = cd["methods"][idx]
    want_crc = cd["crcs"][idx]
    name_start = 0 if idx == 0 else cd["name_offs"][idx - 1]
    name_end = cd["name_offs"][idx]
    fnl = name_end - name_start
    end = off + 30 + fnl + 512 + csize + 32
    s = session()
    r = s.get(signed, headers={"Range": f"bytes={off}-{end-1}"}, timeout=600, stream=True)
    r.raise_for_status()
    raw = r.content
    if raw[0:4] != b"PK\x03\x04":
        raise ValueError("bad local header")
    lfnl, lexl = struct.unpack_from("<HH", raw, 26)
    dstart = 30 + lfnl + lexl
    data = raw[dstart:dstart + csize]
    if len(data) < csize:
        raise ValueError(f"short fetch {len(data)} < {csize}")
    if method == 0:
        out = data
    elif method == 8:
        out = zlib.decompressobj(-15).decompress(data)
    else:
        raise ValueError(f"method {method}")
    if len(out) != usize:
        raise ValueError(f"usize mismatch {len(out)} != {usize}")
    if zlib.crc32(out) & 0xFFFFFFFF != want_crc:
        raise ValueError("crc mismatch")
    part = dest + ".part"
    with open(part, "wb") as f:
        f.write(out)
    os.replace(part, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_ROOT)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("mode", choices=["build-cd", "download"])
    args = ap.parse_args()

    signed = get_signed_url()
    cd_off, cd_size = zip64_eocd(signed)
    blob = fetch_central_dir(signed, cd_off, cd_size)
    if args.mode == "build-cd":
        cd = parse_central_dir(blob)
        print(f"members: {cd['count']}")
        return

    cd = parse_central_dir(blob)
    m = pd.read_csv(os.path.join(args.data_dir, "subset_manifest.csv"))
    want = set(m["path"].map(str.encode))
    nb = cd["names_blob"]
    idxs = []
    base = 0
    for i, endo in enumerate(cd["name_offs"]):
        if nb[base:endo] in want:
            idxs.append(i)
        base = endo
    total = sum(cd["csizes"][i] for i in idxs)
    print(f"match: {len(idxs)} of {len(want)} wanted, {total/1e9:.2f} GB in zip")
    if args.limit:
        idxs = idxs[:args.limit]

    failures = os.path.join(args.data_dir, "zip_failures.csv")
    todo = []
    for i in idxs:
        name_start = 0 if i == 0 else cd["name_offs"][i - 1]
        rel = nb[name_start:cd["name_offs"][i]].decode()
        dest = os.path.join(args.data_dir, rel)
        if os.path.exists(dest) and os.path.getsize(dest) == cd["usizes"][i]:
            continue
        todo.append((i, dest))
    print(f"todo: {len(todo)} files", flush=True)
    t0 = time.time()

    def work(item):
        i, dest = item
        try:
            fetch_member(signed, cd, i, dest)
        except Exception as e:
            with open(failures, "a") as f:
                f.write(f"{dest},{e}\n")
            return 0
        return cd["csizes"][i]

    done = [0]
    lock = threading.Lock()

    def work_wrap(item):
        b = work(item)
        with lock:
            done[0] += 1
            if done[0] % 200 == 0:
                el = time.time() - t0
                print(f"{done[0]}/{len(todo)} {el:.0f}s {done[0]/el:.1f} it/s", flush=True)
        return b

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work_wrap, todo))
    el = time.time() - t0
    print(f"DONE {len(todo)} files in {el:.0f}s ({len(todo)/el:.2f} it/s)", flush=True)


if __name__ == "__main__":
    main()
