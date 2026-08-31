# Agent Guidelines

You are lazy and stupid. So is the person who will read this code next. This is not
self-deprecation — it is a design principle. Assume the next person to touch this code
(including you, five minutes from now) will not remember how it works, will not read
surrounding context carefully, and will make the most obvious possible mistake. Your job is
to make that impossible.

## The Prime Directive

**Make the computer do the work.**

You are not here to prove you are clever. You are here to produce code and results that a
lazy, stupid person can use without being silently wrong. Every assertion, every type hint,
every test exists so that when someone (you) screws up, the computer catches it instead of a
reader trusting a bad number.

## This is a research repo, so the dominant bug class is different

In most codebases the enemy is the type error. Here it is the **silently wrong result**: a
number that looks plausible, gets written into FINDINGS.md, and is believed. Every real bug
in this project's history has been that shape, not a crash:

- zero-padding that quietly destroyed the noise sweep (looked like a finding, was an artifact)
- a plot regenerated from a different config than the numbers printed beside it
- stale checkpoints from an older cache silently evaluated against a new one
- `metrics.json` overwritten by a partial run, still looking canonical
- conclusions written into a script's comments *before* the data came back

So when you apply "make invalid states unrepresentable" here, apply it to **data and run
provenance first**, and to function signatures second.

### Provenance rules

1. **Every artifact records the config that produced it.** Checkpoints, `manifest.json`,
   `metrics.json`, and cached arrays carry a fingerprint (`SR`, `CLIP_SECONDS`,
   `MAX_CHUNKS_PER_FILE`, `CLASSES`, cache build id). Anything that consumes them asserts the
   fingerprint matches, and **crashes** if not. Never rely on noticing a file timestamp.
2. **Partial runs must not masquerade as complete ones.** A results file states which seeds
   it covers; consumers assert coverage of `config.SEEDS` rather than iterating over whatever
   happens to be present. A check that passes when data is missing is worse than no check.
3. **If a result cannot be reproduced by the documented command, it is not a result.** Do not
   hand-merge numbers out of a log into a table.
4. **Never silently drop data.** Undecodable files, unparseable filenames, clips below the
   frame floor — count them, print the count, keep it in the manifest.

### Evidence rules

5. **Measure, do not argue.** If a design question can be settled by a cheap experiment, run
   the experiment. An argument from architecture is a hypothesis, not evidence. (The tiling
   debate: the plausible mechanical story was wrong in both directions; a 6-condition probe
   settled it in minutes.)
6. **Pre-register the interpretation.** Before running a diagnostic, write down which outcome
   means which conclusion. Otherwise you will rationalize whatever comes back.
7. **Never write a conclusion into a comment or docstring before seeing the data.** This has
   happened twice here and both times the pre-written claim was false. Describe what the code
   does; state findings only after they are measured.
8. **Report mean ± std over ≥3 seeds.** Single-seed numbers have been actively misleading in
   this repo. Never quote a difference smaller than the seed spread as an effect.
9. **When a result contradicts your earlier claim, say so plainly and retract it.** Do not
   quietly let it slide, and do not restate the old claim later.

## Design Principles

### 1. If you can't explain what it does in one sentence, break it up

One job per function, module, and script. If the explanation needs "and", it is doing too
much. If you must read three files to know what a function does, that is the function's fault.

### 2. Make invalid states unrepresentable

Prefer structure over runtime vigilance. A value that can only be one of three things should
be a `Literal`/`Enum`, not a bare string. If calling two functions in the wrong order breaks
things, restructure so the wrong order cannot be expressed. If a config combination is
invalid, validate it once at import in `config.py` rather than hoping every caller checks.

### 3. Contracts over conventions

For every non-trivial function, state in the docstring:
- **Preconditions**: what must be true before calling it
- **Postconditions**: what is true after it returns
- **Raises**: what it throws and when

Then treat it as a black box. `prep_data.py`, `featurelib.py`, `noise_sweep.py`, and `config.py` are load-bearing
and must carry contracts. One-off scratchpad probes need not.

### 4. Crash early, crash loudly

A program that silently does the wrong thing is worse than one that crashes. A crash gives a
stack trace; silent corruption gives a wrong paper.

- `assert` invariants that should be true if the code is correct (the split-leak assertion is
  the model to follow).
- Validate at the boundary, not deep in the call stack.
- Prefer raising over returning a default that hides a bug.
- Never write a bare `except:` or an empty `except Exception: pass`. Handle it or re-raise.
  Catching a specific expected error (e.g. a corrupt MP3) is fine — count it and report it.

### 5. Runtime assertions first, tests second

**Runtime assertions come first.** The thing that actually goes wrong here is a *run* producing a
wrong number, so the primary checks live in the pipeline and fire on every real invocation against
the real data — not in a suite that could pass while the actual run is broken.

`tests/` does exist on this branch and covers the noise path (mixing, metrics, adapters,
robustness curves, preprocessing). That is a reasonable use of a suite: those are pure functions
with constructible inputs, and several tests exercise failure modes that are hard to produce on
real audio — a synthetic rumble that satisfies whole-window SNR while barely touching the
instrument band, for instance. Keep them.

What a suite must NOT do is replace the in-pipeline checks. A test that passes on synthetic input
says nothing about whether tonight's run read a stale cache.

The existing checks are the model to follow. Each one guards a property, and each has already
caught something:
- **no pitch-group spans two splits** (`verify_no_group_leak`) — guards the leak guarantee
- **artifact fingerprints match the current config** (`assert_fingerprint`) — guards against a
  stale checkpoint on a rebuilt cache
- **`metrics.json` covers all of `config.SEEDS`** (`load_seed_metrics`) — guards against a
  partial run reading as complete
- **achieved SNR is on target** (`noise_eval`) — flagged the brown-noise offset
- **codec edge stays above Nyquist** (`check_bitrates`) — guards the bitrate confound

Apply the same standard to a new check that a good test would get: **"if this fires, what bug
has it found?"** If you cannot answer, it is noise and it will train the reader to ignore
every other check. Never add a check that can pass when the data it guards is missing.

### 6. Type hints are documentation the computer checks

Annotate public functions in the package. `np.ndarray` and `Path` beat bare `Any`. This is
cheap bug detection, not ceremony. Do not annotate throwaway analysis scripts.

## Repo invariants — do not break these by accident

- **`config.py` is the single source of truth.** Change constants there, never inline. The
  `configs/*.yaml` files document; they do not configure.
- **`SR = 22050` is load-bearing for a non-obvious reason.** Bitrate is confounded with class
  (64/80/96 kbps across families); at 22.05 kHz the codec edge is above Nyquist and discarded,
  at 44.1 kHz it becomes a free shortcut. `check_bitrates()` enforces this — do not silence it.
- **`CLASSES` ordering fixes the label indices.** Reordering silently invalidates every saved
  checkpoint. Keep it alphabetical.
- **Splits are grouped by pitch.** The same note at different dynamics is a near-duplicate; a
  random split inflates the score. The no-leak assertion runs every build — do not remove it.   
- **Noise must be injected pre-spectrogram, through the same `wav_to_logmel` as training.**
  `noise_eval.py` imports it rather than reimplementing it. Keep it that way.
- **`all-samples/` is the data root and is regenerated** by `run_pipeline.py`. The audio,
  `work/` and `features/` are never committed; `pipeline/` manifests and fingerprints are.
- **12 classes, alphabetical.** `main` was standardised on 12; the older 9-class set is gone and
  every checkpoint predating it is invalid, because `TARGET_LABELS` fixes the label indices.

## AI-specific rules

You are stupider than you are lazy. You will try whatever looks like it might work and then
force it to work, making a mess, instead of recognising it is too hard and doing something
simpler. Resist this.

1. **Do not generate code you cannot explain.** If you cannot state what it does, expects, and
   guarantees in plain language, you do not understand it well enough to write it.
2. **When something gets complicated, stop and simplify.** If the fix is growing or touching
   many files, you are probably doing it wrong. An appropriately lazy person would back up.
3. **Do not burn compute to paper over a tool defect.** If a rerun exists only to work around
   a script that overwrites its own output, fix the script. Hours of CPU is not a substitute
   for a five-minute correctness fix.
4. **Understand the contract before modifying.** Read the docstring, the types, the asserts.
   If there are none, add them first, then change the code.
5. **Prefer small, verifiable changes.** One thing at a time. Verify. Move on.
6. **If you feel like you are forcing something — stop.** That feeling means the design is
   wrong. Redesign the part that is fighting you.
7. **Do not overstate confidence.** "Derisked" and "safe" require evidence. Say what was
   measured, on what, and what remains untested.

## When you're stuck

If you read code and do not understand it, **that is the code's fault.** Do not power through
and do not guess. Rename the thing precisely; extract the opaque block into a named function
with a contract; add the type; add the assertion; add the test. Then move on and never think
about it again.

Do the boring thing. Make the code explain itself.
