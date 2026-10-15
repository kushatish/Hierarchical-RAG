"""Run commands from rag_cag_project, e.g. python cli.py ingest data/demo/handbook.md."""
import argparse
import json
from core.engine import Assistant

def main():
    parser = argparse.ArgumentParser(description="Evidence QA")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("paths", nargs="+")
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--semantic-cache", action="store_true")
    commands.add_parser("list")
    delete = commands.add_parser("delete")
    delete.add_argument("id")
    args = parser.parse_args()
    engine = Assistant()
    if args.command == "ingest":
        result = [engine.ingest_file(path) for path in args.paths]
    elif args.command == "ask":
        result = engine.ask(args.question, semantic_cache=args.semantic_cache)
    elif args.command == "list":
        result = engine.list_documents()
    else:
        result = {"deleted": engine.delete_document(args.id)}
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
