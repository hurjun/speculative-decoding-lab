# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Context

This is a personal research lab for studying LLM inference acceleration through speculative decoding. The owner is preparing for graduate study in ML systems (KAIST AI Graduate School and US PhD programs) with a focus on LLM inference efficiency. The repository will be visible to US faculty during graduate applications and research outreach.

**This means everything written here is a professional artifact.** Treat every commit message, code comment, note, and README as something a faculty member at UT Austin, MIT, or CMU might read.

## Language Policy

**All output in this repository is written in English.** This includes:
- Code comments and docstrings
- Commit messages
- Markdown notes (paper notes, code reading notes, experiment logs)
- Issue and PR descriptions
- README files at every level

The owner is actively improving English proficiency for graduate study (TOEFL preparation in progress). All communication that results in a written artifact must be in English.

## Writing Style

Match the conventions of US academic ML writing:

**Do:**
- Use precise, restrained language. Prefer "we observe a 1.4x speedup" over "we achieved a massive speedup."
- Use hedging where appropriate: "suggests," "appears to," "is consistent with."
- Use passive voice for methods sections when natural ("the model was evaluated on...").
- Use active voice for findings and contributions ("we find that...", "this analysis shows...").
- Define acronyms on first use, even common ones like LLM in formal documents.
- Cite specific papers with author and year (e.g., "Leviathan et al., 2023") in notes.

**Avoid:**
- Marketing language: "powerful," "cutting-edge," "revolutionary," "blazing fast."
- Excessive emojis. None in commit messages, code, or formal docs. Sparingly in casual notes if at all.
- First-person plural ("we") in solo project notes unless writing in paper-draft style.
- Vague qualifiers: "very," "really," "quite," "pretty much."
- Filler phrases: "it is important to note that," "needless to say."

When in doubt, read it aloud. If it sounds like a press release, rewrite it.

## Repository Structure

```
notes/           Paper reading notes and literature reviews
code-reading/    Architecture analysis of reference implementations
experiments/     Experiment code and reproducibility scripts
benchmarks/      Quantitative results and comparison tables
docs/            Diagrams, slides, and technical writeups
```

Each subdirectory should have its own README.md explaining what goes there once content exists.

## Git Workflow

### Commit Style

- All commit messages in lowercase.
- No "Claude" or "Co-authored-by" attribution. Do not add `Generated with Claude Code` or similar tags.
- Use conventional prefixes when helpful: `notes:`, `experiments:`, `docs:`, `fix:`, `refactor:`.
- Keep subject line under 60 characters. Add a body if the change needs context.

Examples:
```
notes: add eagle paper reading notes
experiments: reproduce eagle-2 on llama-3.1-8b mt-bench
docs: add draft tree architecture diagram
fix: correct acceptance length formula in benchmark script
```

### Auto-Commit and Auto-Push

When the user finishes a logical unit of work, automatically commit and push without waiting for explicit instruction. Triggers for auto-commit:

1. A new note file is created and contains substantive content.
2. An existing note or document receives a meaningful update (more than typo fixes).
3. A code file in `experiments/` is added or modified and runs without error.
4. A diagram or figure is added to `docs/`.
5. The user says "let's commit," "push it," or similar.

The standard auto-commit sequence:
```bash
git add -A
git status         # show what will be committed
git commit -m "<lowercase message following style above>"
git push origin main
```

If the working tree has unrelated changes, group them into separate commits rather than one large commit. Each commit should be narratable in a single sentence.

If `git push` fails due to remote changes, run `git pull --rebase` first, resolve any conflicts, then push.

### Branching

For Phase 1 (reproduction work), commit directly to `main`. This is a solo learning repository and a clean linear history is more valuable than branch hygiene at this stage. Once Phase 2 experiments begin, switch to feature branches per experiment (`exp/long-context-eagle`, `exp/quantized-draft-acceptance`, etc.).

## Working Conventions

### Paper Notes (`notes/`)

Each paper note follows this structure:

```markdown
# <Paper Title>

**Authors:** <author list>
**Venue:** <conference / arXiv with year>
**Link:** <arXiv URL>

## One-line summary
<a single sentence stating the contribution>

## Problem
<what limitation in prior work the paper addresses>

## Method
<core idea, in your own words; include equations if central>

## Results
<key numerical findings, with the benchmarks they were measured on>

## Strengths
<what is convincing>

## Limitations
<what is unconvincing or open>

## Connection to my work
<why this paper matters for this lab>
```

### Code Reading Notes (`code-reading/`)

When dissecting a reference implementation, document:
- The entry point and call graph for one representative inference step.
- The key data structures (e.g., draft tree buffer layout).
- Any non-obvious implementation choices that diverge from the paper.
- Open questions for follow-up.

### Experiment Logs (`experiments/<id>-<name>/`)

Each experiment directory contains:
- `README.md` — hypothesis, setup, command to reproduce, results.
- `run.py` or `run.sh` — single entry point.
- `configs/` — any config files used.
- `results/` — raw outputs and parsed metrics. Do not commit large model checkpoints; use `.gitignore`.

The experiment README must answer:
1. What question is this experiment asking?
2. What was the exact setup (hardware, model, dataset, hyperparameters)?
3. What did we find?
4. What is the next experiment this suggests?

## Environment

- Local development: macOS with Apple Silicon (M1, 16GB). Used for reading code, writing notes, small-scale dry runs with 1B-class models on MPS backend.
- Compute: rented GPUs on Vast.ai or RunPod (A100 80GB / H100) for actual benchmarks. Once a research internship begins (STAI Lab pending), institutional GPU resources will replace rentals.
- Python 3.11, PyTorch with MPS or CUDA depending on environment.

## When You Are Uncertain

If the user asks for something that could be done in multiple ways and the choice has non-trivial consequences for the project's direction (e.g., which paper to reproduce next, whether to publish a result, how to frame a finding), pause and ask before committing to one path. Default to producing rather than asking, but research direction is the exception.

If the user writes something that reads as non-academic in tone (marketing language, overclaims, casual hedges that weaken a real finding), flag it before committing. The owner is using this repo partly to internalize academic register; surfacing the issue is more useful than silently fixing it.