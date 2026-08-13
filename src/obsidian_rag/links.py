"""Link graph extraction and traversal for the vault.

The vault's links are an explicit knowledge graph: wikilinks and markdown
links between notes. This module extracts them at index time (LinkResolver +
extract_links), and expands vector search hits along them at query time
(expand_neighbors) — graph-aware retrieval without any LLM extraction step.
"""

from __future__ import annotations

import posixpath
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote

# [[Target]], [[Target|alias]], [[Target#heading]], [[Target#heading|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]\|#]+)(?:#[^\]\|]*)?(?:\|[^\]]*)?\]\]")

# [text](target) and [text](<target with spaces>); optional "title" after target
_MDLINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(\s*(?:<([^>]+)>|([^)\s]+))(?:\s+\"[^\"]*\")?\s*\)")

_FENCED_CODE_RE = re.compile(r"```[^\n]*\n.*?\n```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


class LinkResolver:
    """Resolve wikilink names and markdown link targets to vault-relative paths.

    Mirrors Obsidian's resolution order: exact vault-relative path first, then
    unique basename match (case-insensitive fallback).
    """

    def __init__(self, md_paths: Iterable[str]):
        self.paths: Set[str] = {p.replace("\\", "/") for p in md_paths}
        self._by_name: Dict[str, List[str]] = {}
        self._by_name_ci: Dict[str, List[str]] = {}
        for p in self.paths:
            name = posixpath.splitext(posixpath.basename(p))[0]
            self._by_name.setdefault(name, []).append(p)
            self._by_name_ci.setdefault(name.lower(), []).append(p)

    def resolve_name(self, name: str) -> Optional[str]:
        """Resolve a wikilink target (note name or vault-relative path)."""
        name = name.strip().strip("/")
        if not name:
            return None
        for candidate in (name, f"{name}.md"):
            if candidate in self.paths:
                return candidate
        matches = self._by_name.get(posixpath.basename(name)) \
            or self._by_name_ci.get(posixpath.basename(name).lower()) \
            or []
        # Ambiguous basenames resolve to the shortest path, like Obsidian's
        # "shortest path when possible" default.
        return min(matches, key=len) if matches else None

    def resolve_target(self, target: str, source_rel_path: str) -> Optional[str]:
        """Resolve a markdown link target relative to the linking note."""
        target = unquote(target.strip())
        if not target or _SCHEME_RE.match(target) or target.startswith("#"):
            return None
        target = target.split("#", 1)[0]
        if not target.lower().endswith(".md"):
            return None

        if target.startswith("/"):
            resolved = posixpath.normpath(target.lstrip("/"))
        else:
            source_dir = posixpath.dirname(source_rel_path.replace("\\", "/"))
            resolved = posixpath.normpath(posixpath.join(source_dir, target))
        if resolved.startswith(".."):
            return None
        if resolved in self.paths:
            return resolved
        # Fall back to name resolution for bare "Note.md" style links that
        # aren't in the source's own directory.
        return self.resolve_name(posixpath.splitext(resolved)[0])


def extract_links(body: str, source_rel_path: str, resolver: LinkResolver) -> Set[str]:
    """Extract resolved outgoing note links from a markdown body.

    Links inside fenced or inline code are ignored. Only links that resolve
    to an existing markdown file in the vault produce edges; self-links are
    dropped.
    """
    body = _FENCED_CODE_RE.sub("", body)
    body = _INLINE_CODE_RE.sub("", body)

    targets: Set[str] = set()
    for m in _WIKILINK_RE.finditer(body):
        resolved = resolver.resolve_name(m.group(1))
        if resolved:
            targets.add(resolved)
    for m in _MDLINK_RE.finditer(body):
        raw = m.group(1) or m.group(2)
        resolved = resolver.resolve_target(raw, source_rel_path)
        if resolved:
            targets.add(resolved)

    source = source_rel_path.replace("\\", "/")
    targets.discard(source)
    return targets


@dataclass
class Neighbor:
    """A note reached by traversing the link graph from a seed note."""

    path: str
    hop: int
    via: str        # the note that linked here (or was linked from here)
    direction: str  # "link" (via -> path) or "backlink" (path -> via)


def expand_neighbors(
    store,
    seed_paths: Sequence[str],
    hops: int = 1,
    limit: int = 10,
) -> List[Neighbor]:
    """Breadth-first expansion of the link graph around seed notes.

    Traverses outgoing links and backlinks alike, never revisits a note,
    never returns a seed, and stops after `limit` neighbors. Results arrive
    in BFS order: everything at hop 1 before anything at hop 2, so nearer
    context always survives the cap.
    """
    if hops <= 0 or limit <= 0:
        return []

    seeds = [p.replace("\\", "/") for p in seed_paths]
    visited: Set[str] = set(seeds)
    queue: deque[Tuple[str, int]] = deque((p, 0) for p in seeds)
    neighbors: List[Neighbor] = []

    while queue and len(neighbors) < limit:
        path, depth = queue.popleft()
        if depth >= hops:
            continue
        for target in sorted(store.get_links(path)):
            if target not in visited:
                visited.add(target)
                neighbors.append(Neighbor(target, depth + 1, path, "link"))
                queue.append((target, depth + 1))
                if len(neighbors) >= limit:
                    return neighbors
        for source in sorted(store.get_backlinks(path)):
            if source not in visited:
                visited.add(source)
                neighbors.append(Neighbor(source, depth + 1, path, "backlink"))
                queue.append((source, depth + 1))
                if len(neighbors) >= limit:
                    return neighbors
    return neighbors
