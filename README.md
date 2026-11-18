# PathSnap

PathSnap matches GPS observations to directed road segments. The API and UI display individual matched segments, unmatched observations and decoder reset events. They do not claim to reconstruct every intermediate road.

## Setup

Use Python 3.9 or later and a recent Node.js version compatible with Vite 6 (Node 18, 20 or 22+). From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd src/ui && npm ci && cd ../..
```

On Windows activate `.venv\Scripts\activate`. Place permitted data outside Git. Each partition contains one folder per trip, and each trip folder contains matching `.track`, `.nodes` and `.arcs` files. References in `.route` are used only for offline development evaluation, never prediction. Evaluator data and labels must not be used in training or tuning.

## Run the two-laptop demo

On Laptop 1, point to the appropriate permitted partition, then start the API. For development, use the supplied `train` folder and trip `00000005`. Configure the final evaluator folder separately when supplied.

```sh
export PATHSNAP_DATA_DIR='/absolute/path/to/Student folder/train'
export PATHSNAP_DECODER=projected
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

With no `PATHSNAP_MODEL_ARTIFACT`, the scorer is the distance baseline. For a trusted trained heading artifact available on Laptop 1, set `PATHSNAP_MODEL_ARTIFACT` to its absolute path before starting the API. `PATHSNAP_DECODER=midpoint` selects the original decoder; `projected` selects the experimental position-aware decoder. The recommended demonstration candidate is the four-feature histogram model with projected decoding; verify that exact configuration with the team before the final demo. Artifacts and data remain local and are not included in Git.

On Laptop 2:

```sh
cd src/ui
npm run dev -- --host 0.0.0.0
```

Open the displayed UI address. Enter `http://<Laptop-1-LAN-IP>:8000` as the backend URL, select **Load trips**, choose a trip, and run the prediction. Both laptops need network access; firewall and CORS settings must permit the connection. `PATHSNAP_CORS_ORIGINS` can restrict allowed UI origins. Map tiles require network access. Do not use `localhost` as Laptop 1's address from Laptop 2.

The UI shows the scoring mode, decoder, unmatched points and reset markers. Purple resets begin new decoding blocks; road segments are displayed separately rather than connected into a fabricated continuous path. The API provides `GET /trips`, `GET /status` and `POST /predict`. Responses include `diagnostics.events` and `diagnostics.matched_blocks`; metrics remain empty when not calculated.

## Verify

```sh
python -m unittest discover -s src/data -p 'test_*.py'
python -m unittest discover -s src/models -p 'test_*.py'
python -m unittest discover -s src/routing -p 'test_*.py'
python -m unittest discover -s src/api -p 'test_*.py'
python -m src.api.integration_check --trip-id 00000005 --api-url http://127.0.0.1:8000
cd src/ui && npm run build
```

For the six available features and the selected four-feature demo see `src/models/FEATURE_EXPANSION.md`. For model experiments and caveats see `src/models/EXPERIMENT_RESULTS.md`, `src/models/HEADING_RESULTS.md` and `src/models/STUDENT_TEST_RESULTS.md`. For routing see `src/routing/STUDENT_TEST_EVALUATION.md`. All reported pointwise scores are agreement with approximate inferred labels, not verified accuracy. The student-test partition was inspected during development; reserved evaluator trips remain for final evaluation.

The two-laptop demo was verified on 18 November 2026 with the local four-feature trained artifact and projected decoder; see `docs/DEMO_VALIDATION.md`. The official evaluation rule remains to be confirmed. The current npm audit flags a `picomatch` regular-expression denial-of-service advisory through the development build toolchain; a compatible fixed release was not available at the time of testing. Do not expose the development server publicly.
