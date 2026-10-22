# Contributing to dynachaos

Contributions are genuinely welcome, and that includes the ones that are not
code. A bug report, a confusing docstring, a README paragraph that turned out to
be wrong, a question that took you an hour to answer yourself — all of those are
worth opening an [issue](https://github.com/openfluids/dynachaos/issues) for.

If you are unsure whether something is worth reporting, it probably is. Open the
issue.

## Getting set up

```bash
git clone https://github.com/openfluids/dynachaos.git
cd dynachaos
uv sync
```

dynachaos has a Rust extension for the hot loops, with a pure-Python fallback
covering everything it does. You do **not** need a Rust toolchain to contribute
to the Python side — the fallback is fully supported and tested. If you are
working on the Rust code, or want to test against the compiled extension:

```bash
uv run maturin develop --release
```

Add `--extra viz` if you are touching the figure scripts; it pulls in matplotlib
and pillow, which nothing else needs.

## Before you open a pull request

The same checks CI runs. Both Python paths, because a change can easily pass one
and break the other:

```bash
uv run pytest tests/ -q -n auto                       # with the Rust extension
DYNACHAOS_NO_RUST=1 uv run pytest tests/ -q -n auto   # pure-Python fallback
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
```

If you touched the Rust code:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

`rust/wasm` has its own workspace, so those two commands do not reach it. If you
touched the browser build, add:

```bash
cargo fmt --manifest-path rust/wasm/Cargo.toml --all -- --check
cargo clippy --manifest-path rust/wasm/Cargo.toml --all-targets -- -D warnings
uv run python scripts/build_wasm.py && uv run python scripts/check_wasm_parity.py
```

If one fails for a reason you think is unrelated to your change, say so in the
pull request rather than working around it — that is useful information, and
sometimes it is CI that is wrong.

## What makes a pull request easy to review

- **One thing at a time.** A focused change gets reviewed quickly. A change that
  also reformats fifty unrelated lines is hard to read and slow to merge.
- **Say what you verified.** A pasted command and its output is worth more than
  "tested locally".
- **Ask early.** For anything substantial, open an issue first. It is much
  better to disagree about an approach before you have written it than after.
- **Draft PRs are fine.** Opening one early to ask "is this the right
  direction?" is welcome and costs nothing.

Reviews may take a few days — one maintainer, research alongside. A nudge on a
quiet pull request is welcome, not annoying.

## Conventions

Only the ones that are actually enforced:

- The Rust extension and the pure-Python fallback must agree. Anything added to
  one needs the other, and a test that passes under both.
- Figure scripts follow the compute/plot split with `.npz` caching; cache
  loaders use the shared `_safe_load()` helper.
- Use `np.random.default_rng(seed)`, not the legacy module-level `np.random`
  calls — reproducibility matters more here than convenience.
- Resolve paths relative to `__file__`. No hardcoded absolute paths.
- Formatting and import order are handled by `ruff` — do not hand-tune them.

## Conduct and licence

Everyone taking part is asked to follow the
[openfluids Code of Conduct](https://github.com/openfluids/.github/blob/main/CODE_OF_CONDUCT.md).
It is short.

dynachaos is licensed under Apache-2.0, and contributions are accepted under the
same licence. See `LICENSE` and `NOTICE`.

Found a security problem? Please do not open a public issue — see the
[security policy](https://github.com/openfluids/dynachaos/security/policy).
