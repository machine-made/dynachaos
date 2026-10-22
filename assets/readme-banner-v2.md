# README Banner v2

Asset: `assets/readme-banner-v2.jpg` (1408x469, 3:1, 220 KB)

Tool/model: xAI Grok CLI, built-in `image_gen` tool, plus local compositing.

Replaces `readme-banner-v1.png`, which was generated with an earlier image model
and carried a model-rendered wordmark. This version adopts the shared language
now used across the openfluids repositories — `fftkit`, `chaos-atlas`,
`openmodalpy`, `dsgbr` — all 3:1, all on a charcoal ground with a warm off-white
lowercase wordmark and cyan/teal structure with coral accents.

## Approach

The wordmark is **not** generated. Image models render short lowercase words
unpredictably, and accepting whatever letterforms come back is most of what
makes a generated banner look cheap. v1 happened to spell `dynachaos` correctly,
which was luck rather than a property of the method. The artwork is now
generated deliberately textless and the type is set locally in Lato Light, sized
to a fixed fraction of the frame width so names of different lengths carry
comparable optical weight across the family.

## Subject

A period-doubling cascade: one filament splits into two, then four, then eight,
the splittings crowding together until the structure dissolves into a chaotic
band — with a few clean vertical windows where order briefly returns. The route
to chaos, which is the territory this package analyses.

The bifurcation motif stays with `dynachaos`. The sibling `chaos-atlas` banner
deliberately uses a strange attractor instead, so the two do not blur.

## Prompt (artwork only, no text)

```text
A stunning abstract scientific artwork, wide 2:1 landscape: a period-doubling
cascade. On the left a single smooth luminous filament travels rightward, then
splits cleanly into two, then each of those splits into four, then eight, the
splittings accelerating and crowding together until the structure dissolves into
a dense misty band of chaos on the right, shot through with a few sharp clean
vertical windows of order where the mist briefly resolves back into a small
number of crisp filaments. Rendered as exquisitely fine glowing lines and dense
stippled points. Brilliant electric cyan and teal, with the densest chaotic
bands flaring hot coral and warm amber. Deep near-black charcoal background with
a subtle gradient and very faint fine grid, volumetric glow, atmospheric depth
of field, fine film grain, rich deep blacks and luminous highlights. Cinematic,
elegant, expensive, gallery-quality scientific data art. ABSOLUTELY NO TEXT [...]
Leave the left third dark, calm and completely empty as negative space.
```

The framing clause matters: `image_gen` rejects a 3:1 request and returns 2:1,
so the result is cropped afterwards. Prompts must state that the image will be
letterboxed and that all subject matter belongs in the central band, or the crop
cuts through it.

## Post-processing

- Returned 1408x704, centre-cropped to 1408x469 for the family's 3:1 aspect.
- Wordmark composited locally: Lato Light, auto-sized to 30% of frame width
  (83 px here), tracking 6% of point size, warm off-white `#F7F3EC`, with a wide
  blurred dark halo underneath. Placed slightly above centre so it clears the
  incoming filament.
- JPEG q95, no chroma subsampling: 220 KB, against 2.1 MB for the v1 PNG. The
  image is a smooth gradient render with no flat colour fields, which is the
  case PNG handles worst and JPEG handles best.

## Rejected alternative

- **Recurrence plot** — a matrix of diagonal line structures with rectangular
  voids, beside a time-series trace. Conceptually apt, since RQA is a headline
  feature, but the square matrix cropped badly into a 3:1 frame and the diagonals
  read as decorative banding rather than as recurrence.
