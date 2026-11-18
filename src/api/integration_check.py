"""One-command integration check. Run from the repository root:

    python -m src.api.integration_check                       # pipeline only, no server needed
    python -m src.api.integration_check --api-url http://192.168.1.23:8000   # also test the live API

It tells you which teammate modules are really integrated, runs one trip, and checks the contract.
It never claims accuracy: it only checks shapes, lengths, ordering and error behaviour.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

from src.api import config
from src.api.runner import PipelineError, resolve_components, run_prediction
from src.api.trips import list_trip_ids

failures = []


def check(ok: bool, message: str) -> None:
    print(("PASS  " if ok else "FAIL  ") + message)
    if not ok:
        failures.append(message)


def http(url: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip-id")
    ap.add_argument("--api-url")
    args = ap.parse_args()

    print(f"Data dir: {config.DATA_DIR}   Model artifact: {config.MODEL_ARTIFACT}")
    components, problems, sig_warnings = resolve_components()
    print("Integrated components:", sorted(components) or "none")
    for p in problems:
        print("  missing:", p)
    for w in sig_warnings:
        print("  SIGNATURE:", w)
    check(not sig_warnings, "teammate function signatures match the contract")

    ids = list_trip_ids(config.DATA_DIR)
    check(bool(ids), f"found {len(ids)} trips in the data dir")
    trip_id = args.trip_id or (ids[0] if ids else None)
    if trip_id is None:
        sys.exit(1)

    try:
        result = run_prediction(config.DATA_DIR / trip_id, config.MODEL_ARTIFACT)
        check(True, f"pipeline ran for trip {trip_id} in mode '{result['mode']}'")
        check(len(result["predictions"]) == len(result["observations"]), "one prediction per observation")
        check([p["observation_index"] for p in result["predictions"]] == list(range(len(result["observations"]))),
              "prediction order matches observation order")
        check(result["metrics"] == {} or isinstance(result["metrics"], dict), "metrics is a dict ({} = not calculated)")
        for w in result["warnings"]:
            print("  warning:", w)
    except PipelineError as exc:
        check(False, f"pipeline failed: {exc}")

    if len(components) < 4:
        print("NOTE  not all teammate modules are integrated yet: do not present this run as a real result.")

    if args.api_url:
        base = args.api_url.rstrip("/")
        status, body = http(f"{base}/trips")
        check(status == 200 and trip_id in body.get("trip_ids", []), "GET /trips lists the trip")
        status, body = http(f"{base}/predict", {"trip_id": trip_id})
        check(status == 200 and body.get("mode") in ("mock", "baseline", "trained"), "POST /predict works")
        check(set(body) >= {"trip_id", "observations", "predictions", "metrics", "mode", "warnings"}, "response has all contract fields")
        data_only = json.dumps({k: body.get(k) for k in ("observations", "predictions", "metrics")})
        check("reference" not in data_only and ".route" not in data_only, "no reference data in the response")
        status, body = http(f"{base}/predict", {"trip_id": "does-not-exist"})
        check(status == 404 and "detail" in body, "bad trip_id gives 404 with a message")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
