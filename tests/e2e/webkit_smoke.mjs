// Smoke test for the six live figures in WebKit across desktop and mobile viewports.
//
//   PLAYWRIGHT_DIR=<path> node tests/e2e/webkit_smoke.mjs <site dir>
//
// Serves the built site locally, opens WebKit via Playwright at 1440x900 and 390x844,
// scrolls each live figure into view, clicks interact, and verifies that the canvas
// paints non-background pixels within 120 seconds and figure.dataset.state is "live".

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import net from "node:net";
import { resolve } from "node:path";

const pwDir = process.env.PLAYWRIGHT_DIR;
if (!pwDir) {
  console.error("Error: PLAYWRIGHT_DIR environment variable is unset. It must point to a directory with node_modules/playwright inside.");
  process.exit(1);
}

let playwright;
try {
  const req = createRequire(resolve(pwDir, "index.js"));
  playwright = req("playwright");
} catch (err) {
  console.error(`Error: Failed to load playwright from PLAYWRIGHT_DIR=${pwDir}: ${err.message}`);
  process.exit(1);
}

const { webkit } = playwright;
if (!webkit) {
  console.error("Error: webkit browser type is not available in playwright.");
  process.exit(1);
}

const SITE = resolve(process.argv[2] || "site");
if (!existsSync(SITE)) {
  console.error(`Error: site directory does not exist: ${SITE}`);
  process.exit(1);
}

const FIGURE_TIMEOUT_MS = 120_000;
const POLL_INTERVAL_MS = 250;

const VIEWPORTS = [
  {
    name: "1440x900",
    contextOptions: {
      viewport: { width: 1440, height: 900 },
    },
  },
  {
    name: "390x844",
    contextOptions: {
      viewport: { width: 390, height: 844 },
      isMobile: true,
      hasTouch: true,
    },
  },
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const freePort = () =>
  new Promise((res) => {
    const s = net.createServer();
    s.listen(0, "127.0.0.1", () => {
      const p = s.address().port;
      s.close(() => res(p));
    });
  });

async function stopChild(child) {
  if (!child || child.pid == null) return;
  if (child.exitCode != null || child.signalCode != null) return;
  const pid = child.pid;
  const exited = new Promise((resolve) => child.once("exit", resolve));
  try {
    child.kill("SIGTERM");
  } catch {}
  const outcome = await Promise.race([exited.then(() => "exited"), sleep(2_000).then(() => "timeout")]);
  if (outcome === "timeout") {
    try {
      child.kill("SIGKILL");
    } catch {}
    await Promise.race([exited, sleep(1_000)]);
  }
}

async function waitForServer(url) {
  for (let i = 0; i < 150; i++) {
    try {
      await fetch(url);
      return;
    } catch {
      await sleep(100);
    }
  }
  throw new Error(`Server at ${url} not reachable after 15 seconds`);
}

const failures = [];

function check(name, ok, detail) {
  if (ok) {
    console.log(`ok   ${name}`);
  } else {
    console.log(`FAIL ${name}${detail !== undefined ? ` -- ${detail}` : ""}`);
    failures.push(name);
  }
}

let server = null;
let browser = null;

try {
  const httpPort = await freePort();
  server = spawn(
    "python3",
    ["-m", "http.server", String(httpPort), "--bind", "127.0.0.1", "--directory", SITE],
    { stdio: "ignore" },
  );

  const siteUrl = `http://127.0.0.1:${httpPort}/`;
  await waitForServer(siteUrl);

  try {
    browser = await webkit.launch({ headless: true });
  } catch (err) {
    console.error(`FAIL WebKit failed to launch: ${err.message}`);
    process.exit(1);
  }

  for (const vp of VIEWPORTS) {
    const context = await browser.newContext(vp.contextOptions);
    const page = await context.newPage();

    const pageErrors = [];
    page.on("pageerror", (err) => {
      pageErrors.push(err.message || String(err));
    });
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        pageErrors.push(msg.text());
      }
    });

    await page.goto(siteUrl, { waitUntil: "domcontentloaded" });

    const figureIds = await page.evaluate(() => {
      const figs = Array.from(document.querySelectorAll("figure[data-live]"));
      return figs.map((f) => f.id);
    });

    if (figureIds.length !== 6) {
      check(
        `found 6 live figures at ${vp.name}`,
        false,
        `found ${figureIds.length} figures: ${figureIds.join(", ")}`,
      );
    }

    for (const figId of figureIds) {
      const errorCountBefore = pageErrors.length;

      // Scroll the figure into view and click its interact button
      try {
        await page.evaluate((id) => {
          const fig = document.getElementById(id);
          if (!fig) throw new Error(`Figure element ${id} not found`);
          fig.scrollIntoView({ block: "center" });
          const btn = fig.querySelector(".act-interact");
          if (!btn) throw new Error(`Interact button not found in ${id}`);
          btn.click();
        }, figId);
      } catch (err) {
        check(`${figId} at ${vp.name}`, false, `click error: ${err.message}`);
        continue;
      }

      // Poll until canvas has non-background pixels and state is "live"
      const t0 = Date.now();
      let painted = false;
      let lastDetail = "timeout";

      while (Date.now() - t0 < FIGURE_TIMEOUT_MS) {
        if (pageErrors.length > errorCountBefore) {
          lastDetail = `page error: ${pageErrors.slice(errorCountBefore).join("; ")}`;
          break;
        }

        const status = await page.evaluate((id) => {
          const fig = document.getElementById(id);
          if (!fig) return { ready: false, reason: "figure not found" };
          if (fig.dataset.state !== "live") {
            return { ready: false, reason: `state is "${fig.dataset.state}", expected "live"` };
          }
          const canvases = Array.from(fig.querySelectorAll("canvas"));
          if (canvases.length === 0) {
            return { ready: false, reason: "no canvas element" };
          }
          for (const canvas of canvases) {
            if (canvas.width > 0 && canvas.height > 0) {
              try {
                const ctx = canvas.getContext("2d");
                if (!ctx) continue;
                const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imgData.data;
                const cR = data[0], cG = data[1], cB = data[2], cA = data[3];
                for (let i = 4; i < data.length; i += 4) {
                  if (
                    data[i] !== cR ||
                    data[i + 1] !== cG ||
                    data[i + 2] !== cB ||
                    data[i + 3] !== cA
                  ) {
                    return { ready: true };
                  }
                }
              } catch (e) {
                return { ready: false, reason: `canvas error: ${e.message}` };
              }
            }
          }
          return { ready: false, reason: "canvas pixels match corner pixel" };
        }, figId);

        if (status && status.ready) {
          painted = true;
          break;
        }
        if (status && status.reason) {
          lastDetail = status.reason;
        }

        await sleep(POLL_INTERVAL_MS);
      }

      if (pageErrors.length > errorCountBefore) {
        check(
          `${figId} at ${vp.name}`,
          false,
          `uncaught page error: ${pageErrors.slice(errorCountBefore).join("; ")}`,
        );
      } else if (painted) {
        check(`${figId} at ${vp.name}`, true);
      } else {
        check(
          `${figId} at ${vp.name}`,
          false,
          `did not paint live canvas within ${FIGURE_TIMEOUT_MS / 1000} s (${lastDetail})`,
        );
      }
    }

    await context.close();
  }
} catch (err) {
  console.error(`FAIL unhandled error: ${err.message}`);
  failures.push("unhandled");
} finally {
  if (browser) {
    try {
      await browser.close();
    } catch {}
  }
  await stopChild(server);
}

process.exit(failures.length > 0 ? 1 : 0);
