"""Static server for the browser demo with the headers a phone needs.

    python serve.py [port]   (default 8788, binds 0.0.0.0)

Python's plain http.server sends no Cache-Control, so browsers apply heuristic
freshness and can hold a stale model after a re-export. Here everything
revalidates (304 when unchanged; the runtime's Cache API store avoids the
download itself), and the cross-origin isolation headers let onnxruntime-web
use multi-threaded WASM, which halved per-sentence latency in testing.
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        path = self.path.split("?", 1)[0]
        # Always revalidate (ETag/Last-Modified -> 304). The runtime keeps model
        # files in its own Cache API store, so revalidation is the only cost, and a
        # re-exported file under the same name is picked up instead of a stale copy
        # that an immutable header would pin for a year.
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8788
    root = Path(__file__).resolve().parent
    ThreadingHTTPServer(("0.0.0.0", port), partial(Handler, directory=str(root))).serve_forever()
