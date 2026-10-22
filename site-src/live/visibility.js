/**
 * Hidden-tab gate shared by every live figure.
 *
 * The worker pool and the direct kernel runner both call this module, so a
 * hidden tab pauses new tile issue and new kernel calls without a special
 * case in any one figure. An in-flight result still goes through the
 * generation counter, which drops it when the view has moved on.
 */

/**
 * @param {Document|object|null|undefined} [doc]
 * @returns {boolean}
 */
export function pageHidden(doc) {
  const d = doc === undefined ? globalDocument() : doc;
  if (!d) return false;
  return d.hidden === true || d.visibilityState === "hidden";
}

/**
 * @param {Document|object|null|undefined} [doc]
 * @param {(hidden: boolean) => void} onChange
 * @returns {() => void}
 */
export function watchVisibility(doc, onChange) {
  const d = doc === undefined ? globalDocument() : doc;
  if (!d || typeof d.addEventListener !== "function") return () => {};
  const handler = () => onChange(pageHidden(d));
  d.addEventListener("visibilitychange", handler);
  return () => {
    if (typeof d.removeEventListener === "function") {
      d.removeEventListener("visibilitychange", handler);
    }
  };
}

function globalDocument() {
  return typeof document !== "undefined" ? document : null;
}
