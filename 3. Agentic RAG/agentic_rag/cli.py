"""CLI.  python -m agentic_rag.cli {seed|ingest|ask|eval|serve}"""
import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentic_rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed", help="generate the demo database and documents")
    ing = sub.add_parser("ingest", help="(incrementally) index documents")
    ing.add_argument("--force", action="store_true", help="re-index every file")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--json", action="store_true")
    ev = sub.add_parser("eval", help="run the golden-set evaluation")
    ev.add_argument("--ids", nargs="*", help="only these case ids")
    srv = sub.add_parser("serve", help="start the API")
    srv.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.cmd == "seed":
        from .seed import seed_all

        print(json.dumps(seed_all(), indent=2))
    elif args.cmd == "ingest":
        from .ingestion.pipeline import ingest

        print(json.dumps(ingest(force=args.force), indent=2))
    elif args.cmd == "ask":
        from .agent import ask as run

        r = run(" ".join(args.question))
        if args.json:
            print(json.dumps(r, indent=2, default=str))
        else:
            print("\n".join(f"  . {t}" for t in r["trace"]))
            print(f"\n{r['answer']}\n\n[{r['latency_s']}s | {r['tokens_in']} in / {r['tokens_out']} out tokens | retries={r['retries']}]")
    elif args.cmd == "eval":
        from .evaluation import run_eval

        return run_eval(only=args.ids)
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("agentic_rag.api:app", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
