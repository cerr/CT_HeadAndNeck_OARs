"""Install the CT_HeadAndNeck_OARs weights into weights/CT_HeadAndNeck_OARs/.
-
-    python get_weights.py --model CT_HeadAndNeck_OARs --url <hashed link>
-    python get_weights.py --model CT_HeadAndNeck_OARs --file /path/to/weights.tar.gz
-"""
import argparse, base64, gzip, os, re, shutil, sys, tarfile, tempfile, urllib.request

MODEL = 'CT_HeadAndNeck_OARs'
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(REPO_DIR, 'weights')
DEST = os.path.join(WEIGHTS_DIR, MODEL)
ARCHIVE_PREFIX = 'models/'  # the archive holds 

def decode_url(hashed):
    """Hashed link as in model_installer: base64 of the gzipped URL (base64 -d | gunzip)."""
    try:
        url = gzip.decompress(base64.b64decode(hashed.strip())).decode().strip()
    except Exception:
        sys.exit("--url is not a valid hashed link (base64-encoded, gzipped URL).")
    if not url.startswith(('http://', 'https://')):
        sys.exit("--url did not decode to an http(s) link.")
    return url

def direct_download_url(url):
    """Box shared links 
    """
    m = re.match(r'^(https?://[^/]*box\.com)/s/([A-Za-z0-9]+)/?(?:[?#].*)?$', url.strip())
    return f'{m.group(1)}/shared/static/{m.group(2)}' if m else url


def download(url, dest):
    """Download url to dest. The link is not printed."""
    url = direct_download_url(url)
    print("Downloading weights")
    req = urllib.request.Request(url, headers={'User-Agent': 'CT_HeadAndNeck_OARs-get_weights'})
    with urllib.request.urlopen(req) as r, open(dest, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        done, last = 0, -1
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
            done += len(b)
            pct = int(100 * done / total) if total else -1
            if pct != last and (pct % 5 == 0 or pct < 0):
                print(f"  {done / 2**20:,.0f} MB" + (f" ({pct}%)" if total else ""), flush=True)
                last = pct
    print(f"Downloaded {done / 2**20:,.0f} MB")


def member_path(name):
    """Relative output path of an archive member, with ARCHIVE_PREFIX stripped; None to skip it.
    Exits on absolute paths and '..' components."""
    name = name.replace('\\', '/')
    while name.startswith('./'):
        name = name[2:]
    if name.startswith('/') or re.match(r'^[A-Za-z]:', name) or '..' in name.split('/'):
        sys.exit(f"Unsafe path in weights archive: {name}")
    if name.startswith('__MACOSX/') or os.path.basename(name).startswith('._'):
        return None  # macOS archive metadata
    if name.startswith(ARCHIVE_PREFIX):
        name = name[len(ARCHIVE_PREFIX):]
    return name or None


def extract(archive, out_dir):
    """Extract the regular files of a tar archive to out_dir. Returns the number of files written."""
    count = 0
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            if m.isdir():
                continue
            if not m.isfile():
                sys.exit(f"Unexpected link or special file in weights archive: {m.name}")
            rel = member_path(m.name)
            if rel is None:
                continue
            out = os.path.join(out_dir, *rel.split('/'))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with t.extractfile(m) as fi, open(out, 'wb') as fo:  # stream copy: no path traversal
                shutil.copyfileobj(fi, fo, 1 << 22)
            count += 1
    return count


def is_installed():
    return os.path.isdir(DEST) and any(os.scandir(DEST))



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=[MODEL], help='model name')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--url', help='hashed download link you received for this model')
    src.add_argument('--file', help='already-downloaded weights archive (.tar.gz)')
    ap.add_argument('--force', action='store_true', help='replace weights that are already installed')
    args = ap.parse_args()

    # Default to the weights source in models.yaml
    if not (args.url or args.file):
        with open(os.path.join(REPO_DIR, 'models.yaml')) as f:
            m = re.search(r'^\s*url:\s*(\S+)', f.read(), re.M)
        if not m:
            sys.exit("No --url or --file given, and models.yaml lists no weights source.")
        args.url = m.group(1)

    if is_installed() and not args.force:
        sys.exit(f"Already installed: {DEST} (use --force to replace)")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=WEIGHTS_DIR, prefix='.get_weights_') as tmp:
        if args.url:
            url = decode_url(args.url)
            raw = os.path.join(tmp, 'download')
            try:
                download(url, raw)
            except Exception as e:  # the message may contain the link
                sys.exit(f"Download failed ({type(e).__name__}). Check the hashed link, or download the "
                         f"file in a browser and install it with --file.")
        else:
            raw = args.file
            if not os.path.isfile(raw):
                sys.exit(f"No such file: {raw}")

        if not tarfile.is_tarfile(raw):
            sys.exit("The weights file is not a .tar / .tar.gz archive. Nothing was installed.")
        staged = os.path.join(tmp, MODEL)
        os.makedirs(staged)
        n = extract(raw, staged)
        if n == 0:
            sys.exit("The weights archive contains no files. Nothing was installed.")

        if os.path.exists(DEST):
            shutil.rmtree(DEST)
        os.replace(staged, DEST)
    print(f"Installed {MODEL} weights ({n} files): {DEST}")


if __name__ == '__main__':
    main()
