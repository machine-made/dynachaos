import { serve } from "bun";
import { statSync, existsSync, readFileSync } from "fs";
import { join, extname } from "path";
import zlib from "zlib";

const SITE_DIR = join(import.meta.dir, "..", "site");
const PORT = Number(process.env.PORT || 8080);

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".woff2": "font/woff2",
  ".webp": "image/webp",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

const COMPRESSIBLE = new Set([".html", ".css", ".js", ".json", ".svg"]);

// In-memory cache for compressed static assets with mtime validation
const fileCache = new Map<string, { raw: Uint8Array; gz: Uint8Array; br: Uint8Array; mime: string; ext: string; mtime: number }>();

function getAsset(filePath: string) {
  if (!existsSync(filePath)) return null;
  const stat = statSync(filePath);
  if (!stat.isFile()) return null;

  let cached = fileCache.get(filePath);
  if (!cached || cached.mtime !== stat.mtimeMs) {
    const raw = readFileSync(filePath);
    const ext = extname(filePath).toLowerCase();
    const mime = MIME_TYPES[ext] || "application/octet-stream";
    const gz = COMPRESSIBLE.has(ext) ? Bun.gzipSync(raw) : raw;
    const br = COMPRESSIBLE.has(ext)
      ? zlib.brotliCompressSync(raw, { params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 11 } })
      : raw;
    cached = { raw, gz, br, mime, ext, mtime: stat.mtimeMs };
    fileCache.set(filePath, cached);
  }
  return cached;
}

serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    let pathname = decodeURIComponent(url.pathname);
    if (pathname === "/") pathname = "/index.html";

    if (pathname === "/favicon.ico") {
      const fav = getAsset(join(SITE_DIR, "favicon.ico"));
      if (!fav) {
        return new Response(null, { status: 204 });
      }
    }

    const filePath = join(SITE_DIR, pathname);
    const asset = getAsset(filePath);
    if (!asset) {
      return new Response("Not Found", { status: 404 });
    }

    const acceptEncoding = req.headers.get("accept-encoding") || "";
    const headers = new Headers();
    headers.set("Content-Type", asset.mime);
    headers.set("Vary", "Accept-Encoding");

    if (asset.ext === ".woff2" || asset.ext === ".webp" || asset.ext === ".png") {
      headers.set("Cache-Control", "public, max-age=31536000, immutable");
    } else {
      headers.set("Cache-Control", "no-cache, must-revalidate");
    }

    if (COMPRESSIBLE.has(asset.ext) && acceptEncoding.includes("br")) {
      headers.set("Content-Encoding", "br");
      headers.set("Content-Length", String(asset.br.length));
      return new Response(asset.br, { headers });
    }

    if (COMPRESSIBLE.has(asset.ext) && acceptEncoding.includes("gzip")) {
      headers.set("Content-Encoding", "gzip");
      headers.set("Content-Length", String(asset.gz.length));
      return new Response(asset.gz, { headers });
    }

    headers.set("Content-Length", String(asset.raw.length));
    return new Response(asset.raw, { headers });
  },
});

console.log(`Dev server running on http://localhost:${PORT}`);
