"""
project_reader.py — Local Project File Scanner
================================================
Scans a local directory and returns a formatted code context block
for injection into the Council debate as verified project context.

Usage
-----
    from project_reader import scan_project

    context_block, file_list, warnings = scan_project("/path/to/project")

The returned context_block can be passed directly to run_council_debate()
as the ``code_context`` parameter.
"""

import os
from pathlib import Path
from typing import List, Optional, Set, Tuple

from glossary import (
    CODE_REVIEW_CONTEXT_HEADER,
    CODE_REVIEW_EXTENSIONS,
    CODE_REVIEW_MAX_FILE_SIZE,
    CODE_REVIEW_MAX_FILES,
    CODE_REVIEW_MAX_TOTAL_CHARS,
    CODE_REVIEW_SKIP_DIRS,
)


def scan_project(
    folder_path: str,
    extensions: Optional[Set[str]] = None,
    *,
    trusted_source: bool = False,
) -> Tuple[str, List[str], List[str]]:
    """
    Scan a project folder and return a formatted code context block.

    Parameters
    ----------
    folder_path : str
        Absolute or relative path to the project folder.
    extensions : set[str], optional
        File extensions to include (e.g. {".py", ".js"}).
        Defaults to CODE_REVIEW_EXTENSIONS from glossary.
    trusted_source : bool, optional
        True only for server-created sandboxed sources (e.g. the temp directory a
        ZIP upload is extracted into). When False (the default), an arbitrary local
        path is refused unless COUNCIL_ALLOW_LOCAL_SCAN == "1", so a deployed
        instance cannot be tricked into reading arbitrary server files
        (e.g. firebase_key.json).

    Returns
    -------
    Tuple[str, List[str], List[str]]
        (context_block, included_files, warnings)

        - context_block : ready-to-inject string with all file contents
        - included_files : list of relative file paths that were included
        - warnings : list of human-readable skip/truncation notices
    """
    # Security gate — refuse arbitrary local-path scans unless explicitly enabled.
    # Trusted sandboxed callers (the ZIP-extraction temp dir) opt in via
    # trusted_source; everyone else needs COUNCIL_ALLOW_LOCAL_SCAN == "1".
    if not trusted_source and os.environ.get("COUNCIL_ALLOW_LOCAL_SCAN") != "1":
        return (
            "",
            [],
            ["סריקת נתיב מקומי מושבתת בשרת זה — העלה את הפרויקט כקובץ ZIP."],
        )

    extensions = extensions or CODE_REVIEW_EXTENSIONS
    root = Path(folder_path).expanduser().resolve()

    if not root.exists():
        return "", [], [f"תיקייה לא נמצאה: {folder_path}"]
    if not root.is_dir():
        return "", [], [f"הנתיב אינו תיקייה: {folder_path}"]

    included_files: List[str] = []
    warnings: List[str] = []
    file_blocks: List[str] = []
    total_chars = 0
    budget_exhausted = False

    # Walk the tree in a deterministic order (sorted dirnames + filenames).
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place so os.walk skips them entirely.
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in CODE_REVIEW_SKIP_DIRS and not d.startswith(".")
        )

        for filename in sorted(filenames):
            if budget_exhausted:
                break
            if len(included_files) >= CODE_REVIEW_MAX_FILES:
                warnings.append(
                    f"הגעת למגבלת {CODE_REVIEW_MAX_FILES} קבצים — "
                    "חלק מהקבצים לא נכללו."
                )
                budget_exhausted = True
                break

            filepath = Path(dirpath) / filename
            if filepath.suffix.lower() not in extensions:
                continue

            relative = str(filepath.relative_to(root))

            try:
                file_size = filepath.stat().st_size
            except OSError:
                continue

            if file_size > CODE_REVIEW_MAX_FILE_SIZE:
                warnings.append(
                    f"דולג: {relative} "
                    f"({file_size // 1024} KB > מגבלת {CODE_REVIEW_MAX_FILE_SIZE // 1024} KB)"
                )
                continue

            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                warnings.append(f"שגיאת קריאה ב-{relative}: {exc}")
                continue

            # Enforce total character budget
            remaining = CODE_REVIEW_MAX_TOTAL_CHARS - total_chars
            if len(content) > remaining:
                if remaining > 300:
                    content = content[:remaining] + "\n… [קוטע — תקציב תווים מלא]"
                    warnings.append(f"קובץ נקטע: {relative}")
                    budget_exhausted = True
                else:
                    warnings.append(
                        f"דולג: {relative} — תקציב התווים הכולל מוצה."
                    )
                    budget_exhausted = True
                    break

            lang = filepath.suffix.lstrip(".") or "text"
            block = f"### קובץ: {relative}\n```{lang}\n{content}\n```"
            file_blocks.append(block)
            included_files.append(relative)
            total_chars += len(content)

    if not file_blocks:
        return (
            "",
            [],
            warnings + ["לא נמצאו קבצי קוד מתאימים בתיקייה שנבחרה."],
        )

    header = CODE_REVIEW_CONTEXT_HEADER.format(
        folder_name=root.name,
        file_count=len(included_files),
        total_chars=total_chars,
    )
    context_block = header + "\n\n" + "\n\n---\n\n".join(file_blocks)

    return context_block, included_files, warnings
