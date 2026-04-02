from __future__ import annotations

import sys

from kb_qa.service import ask


def _read_question(argv: list[str]) -> str:
    if len(argv) > 1:
        return " ".join(argv[1:]).strip()

    return input("请输入问题: ").strip()


def main() -> int:
    question = _read_question(sys.argv)
    if not question:
        print("问题不能为空。")
        return 1

    result = ask(question)

    print("问题:")
    print(result["question"])
    print("\n回答:")
    print(result["answer"])

    sources = result.get("sources", [])
    if sources:
        print("\n来源:")
        for index, source in enumerate(sources, start=1):
            file_name = source.get("file_name") or "未知文件"
            collection_name = source.get("collection_name") or source.get("collection_key")
            score = source.get("score")
            print(f"[{index}] {collection_name} | {file_name} | score={score}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
