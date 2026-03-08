import json

def validate_traces(jsonl_path: str) -> dict:
    """Validate all traces in a jsonl file. Returns summary report."""
    traces = [json.loads(l) for l in open(jsonl_path)]

    errors = []
    valid_count = 0

    for i, trace in enumerate(traces):
        try:
            convs = trace["conversations"]

            # Must have at least 4 turns
            if len(convs) < 4:
                errors.append({"index": i, "error": "fewer than 4 turns"}); continue

            # First turn must be system
            if convs[0]["role"] != "system":
                errors.append({"index": i, "error": "first turn not system"}); continue

            # Must have at least 2 tool calls
            tool_calls = sum(
                1 for m in convs
                if m["role"] == "assistant" and "Action:" in m["content"]
            )
            if tool_calls < 2:
                errors.append({"index": i, "error": f"only {tool_calls} tool calls"}); continue

            # Must contain a Verdict
            has_verdict = any(
                "Verdict:" in m["content"]
                for m in convs if m["role"] == "assistant"
            )
            if not has_verdict:
                errors.append({"index": i, "error": "missing Verdict"}); continue

            # Must not contain hallucinated Observations
            hallucinated = any(
                "Observation:" in m["content"]
                for m in convs if m["role"] == "assistant"
            )
            if hallucinated:
                errors.append({"index": i, "error": "hallucinated Observation"}); continue

            valid_count += 1

        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    report = {
        "total": len(traces),
        "valid": valid_count,
        "invalid": len(errors),
        "error_rate": len(errors) / len(traces),
        "errors": errors[:10]
    }

    # Fail loudly if error rate exceeds 5%
    if report["error_rate"] > 0.05:
        raise ValueError(
            f"Data validation failed: {report['error_rate']:.1%} error rate "
            f"({report['invalid']}/{report['total']} traces invalid)"
        )

    return report


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "data", "train.jsonl")
    report = validate_traces(path)
    print(f"Valid: {report['valid']}/{report['total']}")
    print(f"Error rate: {report['error_rate']:.1%}")
    if report["errors"]:
        print("Sample errors:", report["errors"])