"""
textfmt.py — output sanitizers for Discord.

strip_tables(): Discord doesn't render markdown tables or HTML, so tables show
up as raw pipes and <br> tags show up literally. This rewrites each table into
a bold-title + bullet list and turns <br> tags into real newlines. It's
idempotent and leaves fenced code blocks untouched.
"""
import re

# A separator row looks like |---|:---:|---| (outer pipes optional)
_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_row(line: str) -> bool:
    s = line.strip()
    return s.count("|") >= 2 and (s.startswith("|") or s.endswith("|"))


def _convert_table(rows: list[list[str]]) -> str:
    header, body = rows[0], rows[1:]
    out = []
    for row in body:
        row = row + [""] * (len(header) - len(row))  # pad short rows
        title = row[0] or "(blank)"
        # Keep any bold the model already put on the first cell from doubling up
        title = title.strip("*").strip()
        out.append(f"**{title}**")
        for h, v in zip(header[1:], row[1:]):
            if v:
                out.append(f"- {h}: {v}")
        out.append("")  # blank line between entries
    return "\n".join(out).rstrip()


_BR_RE = re.compile(r"[ \t]*<br\s*/?>[ \t]*", re.IGNORECASE)


def _strip_br(text: str) -> str:
    """<br> -> newline, outside fenced code blocks."""
    if "<" not in text:
        return text
    out, in_code = [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line)
        else:
            out.append(line if in_code else _BR_RE.sub("\n", line))
    return "\n".join(out)


def strip_tables(text: str) -> str:
    if not text:
        return text
    return _strip_br(_strip_tables(text))


def _strip_tables(text: str) -> str:
    if "|" not in text:
        return text

    lines = text.split("\n")
    result: list[str] = []
    in_code = False
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            in_code = not in_code
            result.append(line)
            i += 1
            continue

        # A table = header row, then a separator row, then 0+ body rows
        if (
            not in_code
            and _is_row(line)
            and i + 1 < len(lines)
            and _SEP_RE.match(lines[i + 1])
            and "-" in lines[i + 1]
        ):
            rows = [_split_row(line)]
            j = i + 2
            while j < len(lines) and _is_row(lines[j]) and not _SEP_RE.match(lines[j]):
                rows.append(_split_row(lines[j]))
                j += 1
            result.append(_convert_table(rows))
            i = j
            continue

        result.append(line)
        i += 1

    return "\n".join(result)


if __name__ == "__main__":
    sample = """**Nemotron-3 family – quick comparison**

| Model | Parameter count | Typical context window | Primary strengths |
|-------|----------------|-----------------------|-------------------|
| **nemotron-3-nano:30b** | ~30 billion | 8k tokens | Very fast, low-cost inference |
| **nemotron-3-super** | ~70-80 billion | 16k tokens | Strong reasoning |
| **nemotron-3-ultra** | ~110-130 billion | 32k tokens | State-of-the-art generation |

Pick based on your GPU budget.

```
| keep | this |
|------|------|
| code | block |
```
"""
    sample += "\nA <br>• one <br/>• two<BR />• three\n```\nkeep <br> in code\n```\n"
    out = strip_tables(sample)
    print(out)
    assert "<br>" not in out.split("```")[0]
    assert "keep <br> in code" in out
    assert "|---" not in out.split("```")[0]
    assert strip_tables(out) == out  # idempotent
    print("\nOK")
