/**
 * Pure geometry for live (Omega, K) heatmaps.
 *
 * No DOM. Node tests import this. The paper page uses the same functions
 * to blit a tile and to answer fig._live.sample().
 *
 * Tile rows: row 0 is kMin, the bottom of the data rectangle. Canvas y
 * points down, so that row is the last row of an ImageData buffer.
 */

export const HEADER = 4;

const MAX_SIDE = 512;
const MAX_STEPS = 250_000_000;
const MIN_SPAN = 1e-9;

/**
 * Pixel rectangle of a tile on a canvas whose y axis points down.
 *
 * `plot` is the visible data rectangle in canvas pixels: data x maps onto
 * [left, left + width], data y maps onto [top + height, top] (kMin at the
 * bottom).
 *
 * @param {{ omegaMin: number, omegaMax: number, kMin: number, kMax: number }} tile
 * @param {{ x0: number, x1: number, y0: number, y1: number, left: number, top: number, width: number, height: number }} plot
 * @returns {{ x: number, y: number, w: number, h: number }}
 */
export function tilePixelRect(tile, plot) {
  const sx = (v) => plot.left + ((v - plot.x0) / (plot.x1 - plot.x0)) * plot.width;
  const sy = (v) => plot.top + plot.height - ((v - plot.y0) / (plot.y1 - plot.y0)) * plot.height;
  const x = sx(tile.omegaMin);
  const x2 = sx(tile.omegaMax);
  const yTop = sy(tile.kMax);
  const yBot = sy(tile.kMin);
  return { x, y: yTop, w: x2 - x, h: yBot - yTop };
}

/**
 * World rectangle of a scheduler tile id in a viewport.
 *
 * @param {string} id
 * @param {{ omegaMin: number, omegaMax: number, kMin: number, kMax: number }} viewport
 * @returns {object | null}
 */
export function tileWorld(id, viewport) {
  if (viewport == null || id == null) return null;
  const parts = String(id).split(":");
  if (parts.length < 3) return null;
  const level = Number(parts[0]);
  const ix = Number(parts[1]);
  const iy = Number(parts[2]);
  if (!Number.isFinite(level) || !Number.isFinite(ix) || !Number.isFinite(iy)) return null;
  const n = 2 ** level;
  if (n < 1 || !Number.isFinite(n)) return null;
  const dOmega = (viewport.omegaMax - viewport.omegaMin) / n;
  const dK = (viewport.kMax - viewport.kMin) / n;
  return {
    id: String(id),
    level,
    ix,
    iy,
    omegaMin: viewport.omegaMin + ix * dOmega,
    omegaMax: viewport.omegaMin + (ix + 1) * dOmega,
    kMin: viewport.kMin + iy * dK,
    kMax: viewport.kMin + (iy + 1) * dK,
  };
}

/**
 * Rotation number at (omega, K) from the finest current-generation tile
 * covering the point. `null` if none of `tiles` covers it.
 *
 * Row 0 of a tile is kMin. The last row is kMax.
 *
 * @param {object[]} tiles
 * @param {number} generation
 * @param {number} omega
 * @param {number} K
 * @returns {number | null}
 */
export function sampleAt(tiles, generation, omega, K) {
  if (!Array.isArray(tiles) || !Number.isFinite(omega) || !Number.isFinite(K)) return null;
  let best = null;
  for (let i = 0; i < tiles.length; i++) {
    const tile = tiles[i];
    if (!tile || tile.generation !== generation) continue;
    if (!covers(tile, omega, K)) continue;
    if (!best || tile.level > best.level) best = tile;
  }
  if (!best) return null;
  return cellValue(best, omega, K);
}

function covers(tile, omega, K) {
  return (
    omega >= tile.omegaMin &&
    omega <= tile.omegaMax &&
    K >= tile.kMin &&
    K <= tile.kMax
  );
}

function cellValue(tile, omega, K) {
  const header = tile.header;
  const data = tile.data;
  if (header == null || data == null) return null;
  const nOmega = Math.max(1, Math.floor(Number(header[0])));
  const nK = Math.max(1, Math.floor(Number(header[1])));
  const spanO = tile.omegaMax - tile.omegaMin;
  const spanK = tile.kMax - tile.kMin;
  let ix = spanO === 0 ? 0 : Math.floor(((omega - tile.omegaMin) / spanO) * nOmega);
  let iy = spanK === 0 ? 0 : Math.floor(((K - tile.kMin) / spanK) * nK);
  if (ix >= nOmega) ix = nOmega - 1;
  if (iy >= nK) iy = nK - 1;
  if (ix < 0) ix = 0;
  if (iy < 0) iy = 0;
  const v = data[HEADER + iy * nOmega + ix];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * Cache key for one live tile's coloured pixels.
 *
 * Live colour uses a fixed viridis ramp. The key does not include the page theme.
 *
 * The fingerprint is generation, id, the record's paint sequence, grid size,
 * sample count, and the first and last samples. The paint sequence is a
 * per-record counter stamped when the figure stores a painted tile: a record
 * that replaces another inside one generation keys differently, so a repaint
 * always misses the cache while an unchanged record still hits it. Hashing
 * the samples instead would cost a full-tile pass on the paint path.
 *
 * @param {object} tile
 * @returns {string}
 */
export function liveTileColorKey(tile) {
  if (tile == null) return "";
  const header = tile.header;
  const data = tile.data;
  const nOmega = header ? header[0] : "";
  const nK = header ? header[1] : "";
  const n = data && data.length ? data.length : 0;
  let first = "";
  let last = "";
  if (data && n > HEADER) {
    first = data[HEADER];
    last = data[n - 1];
  }
  return `${tile.generation}|${tile.id}|${tile.paintSeq == null ? "" : tile.paintSeq}|${nOmega}|${nK}|${n}|${first}|${last}`;
}

/**
 * Tile side (cells) so the finest pyramid level is about one cell per
 * device pixel, without exceeding the wasm side cap or the step budget
 * that would cut nIter below the requested value.
 *
 * @param {number} pixelWidth
 * @param {number} pixelHeight
 * @param {{ levels?: number, nIter?: number, nTransient?: number, maxSide?: number, maxSteps?: number }} [options]
 * @returns {number}
 */
export function tileCellsFor(pixelWidth, pixelHeight, options = {}) {
  const levels = Math.max(1, Math.floor(Number(options.levels)) || 3);
  const nIter = Math.max(1, Math.floor(Number(options.nIter)) || 2000);
  const nTransient = Math.max(0, Math.floor(Number(options.nTransient)) || 0);
  const maxSide = Math.max(1, Math.floor(Number(options.maxSide)) || MAX_SIDE);
  const maxSteps = Number(options.maxSteps);
  const budget = Number.isFinite(maxSteps) && maxSteps > 0 ? maxSteps : MAX_STEPS;
  const finest = 2 ** (levels - 1);
  const px = Math.max(1, Math.round(Math.max(Number(pixelWidth) || 1, Number(pixelHeight) || 1)));
  const target = Math.max(1, Math.round(px / finest));
  const perCell = nTransient + nIter;
  const maxByBudget = Math.max(1, Math.floor(Math.sqrt(budget / perCell)));
  return Math.max(1, Math.min(maxSide, maxByBudget, target));
}

export { MIN_SPAN, MAX_SIDE, MAX_STEPS };
