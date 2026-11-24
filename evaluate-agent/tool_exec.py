"""Runs one tool call inside the sandbox started by tool_sandbox.py."""

import json
import sys

from sweqapro.tools import execute_readonly_command, semantic_search, view_codebase


def run_tool(call: dict, repo_path: str) -> str:
    name = call.get("name")
    args = call.get("args", {}) or {}
    try:
        if name == "semantic_search":
            result = semantic_search(
                term=args["term"],
                path=args.get("path", repo_path),
                python_only=args.get("python_only", False),
                max_files=args.get("max_files", 100),
                include_lines=args.get("include_lines", False),
            )
            return json.dumps(result)
        if name == "view_codebase":
            return view_codebase(
                path=args["path"],
                view_range=args.get("view_range"),
                concise=args.get("concise", False),
                python_only=args.get("python_only", False),
            )
        if name == "execute_readonly_command":
            return execute_readonly_command(args["command"])
        return f"ERROR: Unknown tool {name}"
    except Exception as e:
        return f"ERROR: Tool failed: {e}"


def main() -> None:
    request = json.loads(sys.argv[1])
    sys.stdout.write(run_tool(request["call"], request["repo_path"]))


if __name__ == "__main__":
    main()
