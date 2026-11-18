# PathSnap (ML_Hack)

GPS map-matching: match noisy GPS points to road segments and reconstruct a plausible route.
Status: first-review scaffold. Nothing here is a trained model until a component says so.

## Layout and owners
| Folder | Owner | Branch |
|---|---|---|
| `src/data/` | Suhaas | `suhaas/data` |
| `src/models/` | Pratham | `pratham/models` |
| `src/routing/` | Adithya | `adithya/routing` |
| `src/api/`, `src/ui/`, `src/visualization/`, root setup | Prithika | `prithika/app` |
| `docs/` | shared (contract changes need team agreement) | |
| `data/` | local only, git-ignored | |

Contract: see `docs/INTEGRATION_CONTRACT.md`.

## Rules
- Branch from the same initial commit. No force-push. Edit only your own folder.
- Do not add `__init__.py` files (namespace packages are used).
- Never commit data, `.env`, secrets or evaluator labels.
- Never use `.route` in inference. Synthetic fixtures must be clearly labeled and never reported as results.
- Each owner adds their own tests/examples/docs inside their folder and shares: branch, commit, one-line run command, whether output is mock / baseline / trained.

## Setup (all laptops)
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env             # then edit paths
    # place dataset under data/ (see data/README.md)

## Run
Filled in by each owner as slices land (API: `src/api/`, UI: `src/ui/`).
