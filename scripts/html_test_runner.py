#!/usr/bin/env python3
"""HTML code generation test runner for Gemma 4 E2B via llama.cpp.

Sends each HTML test prompt to the llama.cpp server, runs deterministic
checks on the output, and records all errors. Outputs a structured report
that can be used to build an adaptive error database.

Usage:
    python3 html_test_runner.py --output results.json
    python3 html_test_runner.py --category profile --output results.json
    python3 html_test_runner.py --category game --max-tokens 4096
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error

# Add scripts dir to path so we can import html_prompts
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from html_prompts import ALL_PROMPTS, HTMLTestPrompt, get_prompts_by_category

LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8080
DEFAULT_MAX_TOKENS = 4096  # HTML is verbose — needs more tokens


def query_llama(prompt: str, max_tokens: int, temperature: float = 0.0,
                timeout: int = 300) -> tuple[str, str, dict]:
    """Send a prompt to llama.cpp and return (content, reasoning, usage_info)."""
    url = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )

    try:
        start = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            elapsed = time.time() - start
    except urllib.error.URLError as e:
        return "", "", {"error": f"Connection error: {e.reason}", "elapsed": 0}
    except Exception as e:
        return "", "", {"error": f"Query error: {e}", "elapsed": 0}

    content = ""
    reasoning = ""
    choices = data.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")

    usage = data.get("usage", {})
    usage["elapsed"] = elapsed

    return content, reasoning, usage


def run_checks(html_output: str, prompt: HTMLTestPrompt) -> list[dict]:
    """Run all deterministic checks on the model output.

    Returns list of check results:
      [{"check": name, "passed": bool, "detail": str}]
    """
    results = []
    for check_fn in prompt.checks:
        try:
            passed, detail = check_fn(html_output)
        except Exception as e:
            passed = False
            detail = f"Check error: {e}"
        results.append({
            "check": check_fn.__name__.lstrip("_"),
            "passed": passed,
            "detail": detail,
        })
    return results


def run_all_tests(prompts: list[HTMLTestPrompt], max_tokens: int,
                  temperature: float, verbose: bool) -> dict:
    """Run all prompts and return structured results."""
    all_results = []
    total_checks = 0
    total_passed = 0
    total_failed = 0

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt.id} ({prompt.category}/{prompt.difficulty})")
        print(f"  Prompt: {prompt.prompt[:80]}...")

        # Query the model
        content, reasoning, usage = query_llama(
            prompt.prompt, max_tokens, temperature
        )

        if usage.get("error"):
            print(f"  ❌ ERROR: {usage['error']}")
            all_results.append({
                "id": prompt.id,
                "category": prompt.category,
                "difficulty": prompt.difficulty,
                "prompt": prompt.prompt,
                "error": usage["error"],
                "html_output": "",
                "checks": [],
                "check_results": {"passed": 0, "failed": 0, "total": 0},
                "timing": {"elapsed": usage.get("elapsed", 0)},
            })
            continue

        # Run deterministic checks
        check_results = run_checks(content, prompt)
        passed = sum(1 for r in check_results if r["passed"])
        failed = sum(1 for r in check_results if not r["passed"])
        total_checks += len(check_results)
        total_passed += passed
        total_failed += failed

        status = "✅" if failed == 0 else f"❌ {failed} failures"
        elapsed = usage.get("elapsed", 0)
        tokens = usage.get("completion_tokens", "?")
        print(f"  {status} — {passed}/{len(check_results)} checks passed ({elapsed:.1f}s, {tokens} tokens)")

        if verbose or failed > 0:
            for r in check_results:
                marker = "✅" if r["passed"] else "❌"
                print(f"    {marker} {r['check']}: {r['detail']}")

        all_results.append({
            "id": prompt.id,
            "category": prompt.category,
            "difficulty": prompt.difficulty,
            "prompt": prompt.prompt,
            "html_output": content[:10000],  # Cap stored output (was 5000, caused truncation for long game code)
            "html_length": len(content),
            "reasoning_length": len(reasoning),
            "checks": check_results,
            "check_results": {"passed": passed, "failed": failed, "total": len(check_results)},
            "timing": {
                "elapsed": elapsed,
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
            },
        })

    return {
        "summary": {
            "total_prompts": len(prompts),
            "total_checks": total_checks,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "pass_rate": f"{total_passed}/{total_checks}" if total_checks > 0 else "0/0",
            "pass_percent": round(total_passed / total_checks * 100, 1) if total_checks > 0 else 0,
        },
        "results": all_results,
    }


def compile_error_list(results: dict) -> list[dict]:
    """Extract a compiled list of all errors found, grouped by check type."""
    errors_by_check: dict[str, list[dict]] = {}

    for result in results["results"]:
        if result.get("error"):
            continue
        for check in result.get("checks", []):
            if not check["passed"]:
                check_name = check["check"]
                if check_name not in errors_by_check:
                    errors_by_check[check_name] = []
                errors_by_check[check_name].append({
                    "prompt_id": result["id"],
                    "category": result["category"],
                    "detail": check["detail"],
                })

    error_list = []
    for check_name, occurrences in sorted(errors_by_check.items(),
                                           key=lambda x: -len(x[1])):
        error_list.append({
            "check": check_name,
            "count": len(occurrences),
            "affected_prompts": list(set(o["prompt_id"] for o in occurrences)),
            "details": occurrences,
        })

    return error_list


def main():
    parser = argparse.ArgumentParser(
        description="HTML code generation test runner for Gemma 4 E2B"
    )
    parser.add_argument("--category", choices=["profile", "game", "all"],
                        default="all", help="Which prompts to run")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max tokens per response (default {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature (default 0.0)")
    parser.add_argument("--output", "-o", default=None,
                        help="Save JSON results to file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all check details, not just failures")
    args = parser.parse_args()

    # Select prompts
    if args.category == "all":
        prompts = ALL_PROMPTS
    else:
        prompts = get_prompts_by_category(args.category)

    print(f"═" * 60)
    print(f"  HTML Code Generation Test Runner")
    print(f"  Model: Gemma 4 E2B via llama.cpp ({LLAMA_HOST}:{LLAMA_PORT})")
    print(f"  Prompts: {len(prompts)} ({args.category})")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"═" * 60)

    results = run_all_tests(prompts, args.max_tokens, args.temperature, args.verbose)

    # Print summary
    s = results["summary"]
    print(f"\n{'═' * 60}")
    print(f"  SUMMARY")
    print(f"  Prompts:  {s['total_prompts']}")
    print(f"  Checks:   {s['total_passed']}/{s['total_checks']} passed ({s['pass_percent']}%)")
    print(f"  Errors:   {s['total_failed']}")
    print(f"{'═' * 60}")

    # Compile and print error list
    errors = compile_error_list(results)
    if errors:
        print(f"\n  ERROR SUMMARY (grouped by check type):")
        print(f"  {'─' * 56}")
        for err in errors:
            print(f"  {err['check']:30s}  {err['count']:3d} occurrences  prompts: {', '.join(err['affected_prompts'])}")
        print(f"  {'─' * 56}")
    else:
        print(f"\n  ✅ No errors found!")

    # Save results
    results["errors"] = errors
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()