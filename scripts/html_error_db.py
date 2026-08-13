"""HTML error database for adaptive auto-correction.

Built from real Gemma 4 E2B outputs on 25 HTML test prompts (247/258 passed).
Each error pattern includes:
  - id: unique error identifier
  - detect: function(html: str) -> bool  — True if error is present
  - fix: function(html: str) -> str      — returns patched HTML
  - description: human-readable explanation
  - affected_prompts: which prompt IDs this error was observed in

Error patterns found (10 real errors, 1 test bug):
  1. Missing CSS color styling (3 prompts) — model produces plain HTML, no <style>
  2. Missing <nav> semantic tag (1 prompt) — uses <div> for nav links
  3. Missing HTML document structure (5 checks, 1 prompt) — returns a fragment
     instead of a full <!DOCTYPE html> document
  4. Missing clearRect() (1 prompt) — canvas game without clearing, causes trails
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class HTMLErrorPattern:
    """A curated HTML error pattern with detection and automatic fix."""
    __test__ = False
    id: str
    description: str
    affected_prompts: list[str]
    detect: Callable[[str], bool]  # Returns True if error is present
    fix: Callable[[str], str]      # Returns patched HTML


# ═══════════════════════════════════════════════════════════════════════
# Error 1: Missing CSS color styling — model produces unstyled HTML
# ═══════════════════════════════════════════════════════════════════════

def _detect_missing_color_styling(html: str) -> bool:
    """Detect HTML that has no color styling at all (no <style>, no CSS colors, no fillStyle)."""
    has_style_tag = bool(re.search(r"<style[^>]*>", html, re.IGNORECASE))
    has_css_link = bool(re.search(r'<link[^>]*stylesheet[^>]*>', html, re.IGNORECASE))
    has_css_color = bool(re.search(r"(?:background|color|fill)[^;{]*:", html, re.IGNORECASE))
    has_fill_style = bool(re.search(r"fillStyle\s*=", html))
    # Only flag if there's body content but zero styling
    has_body = bool(re.search(r"<body[^>]*>.*?</body>", html, re.IGNORECASE | re.DOTALL))
    return has_body and not (has_style_tag or has_css_link or has_css_color or has_fill_style)


def _fix_missing_color_styling(html: str) -> str:
    """Inject a minimal <style> block with basic color styling into <head>."""
    base_style = """<style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; margin: 20px; }
        h1 { color: #2c3e50; }
        a { color: #3498db; }
    </style>"""

    # If there's a <head>, inject before </head>
    if re.search(r"</head\s*>", html, re.IGNORECASE):
        html = re.sub(r"(</head\s*>)", base_style + r"\n\1", html, count=1, flags=re.IGNORECASE)
    # If there's no <head> but there is <html>, add head
    elif re.search(r"<html[^>]*>", html, re.IGNORECASE):
        html = re.sub(
            r"(<html[^>]*>)",
            r"\1\n<head>" + base_style + "\n</head>",
            html, count=1, flags=re.IGNORECASE,
        )
    # If no <html> at all, this will be handled by the document-structure fix
    return html


# ═══════════════════════════════════════════════════════════════════════
# Error 2: Missing <nav> semantic tag — model uses <div> for navigation
# ═══════════════════════════════════════════════════════════════════════

def _detect_missing_nav_tag(html: str) -> bool:
    """Detect navigation links not wrapped in a <nav> tag.
    Heuristic: has multiple <a href> links that look like navigation
    (Home, About, Projects, etc.) but no <nav> tag."""
    has_nav = bool(re.search(r"<nav[^>]*>", html, re.IGNORECASE))
    if has_nav:
        return False
    # Look for a div or section with class/id containing 'nav' that has links
    nav_div = re.search(
        r'<div[^>]*(?:class|id)\s*=\s*["\'][^"\']*nav[^"\']*["\'][^>]*>.*?</div>',
        html, re.IGNORECASE | re.DOTALL,
    )
    # Also check for a cluster of anchor tags (3+ links together)
    links = re.findall(r'<a\s+href\s*=', html, re.IGNORECASE)
    return bool(nav_div) or len(links) >= 3


def _fix_missing_nav_tag(html: str) -> str:
    """Replace navigation divs with <nav> tags.

    Handles two patterns:
    1. <div class="nav"> or <div id="nav"> — explicit nav-named div
    2. <div class="social-links"> or similar — a div containing 3+ anchor tags
    """
    # Pattern 1: div with class/id containing 'nav'
    def _is_nav_div(tag_text: str) -> bool:
        return bool(re.search(r'(?:class|id)\s*=\s*["\'][^"\']*nav[^"\']*["\']', tag_text, re.IGNORECASE))

    # Pattern 2: find all divs and check which contain 3+ <a> tags
    # We'll look for divs that wrap social links, menu items, etc.
    def _is_link_cluster_div(html: str, div_open_tag: str, div_start: int) -> bool:
        # Find the matching </div> for this div (simplified depth counting)
        depth = 1
        pos = div_start
        while depth > 0 and pos < len(html):
            next_open = html.find('<div', pos)
            next_close = html.find('</div', pos)
            if next_close == -1:
                break
            if next_open != -1 and next_open < next_close:
                depth += 1
                pos = next_open + 4
            else:
                depth -= 1
                pos = next_close + 6
        # Count <a> tags between div_start and matching </div>
        div_content = html[div_start:pos]
        links = re.findall(r'<a\s+href', div_content, re.IGNORECASE)
        return len(links) >= 3

    # Find divs to convert
    div_matches = list(re.finditer(r'<div[^>]*>', html, re.IGNORECASE))
    convert_positions: list[tuple[int, int]] = []  # (start, end) of div open tag

    for dm in div_matches:
        div_tag = dm.group(0)
        if _is_nav_div(div_tag):
            convert_positions.append((dm.start(), dm.end()))
        elif _is_link_cluster_div(html, div_tag, dm.end()):
            convert_positions.append((dm.start(), dm.end()))

    # Replace from end to start so positions don't shift
    for start, end in reversed(convert_positions):
        div_tag = html[start:end]
        # Replace <div ...> with <nav ...> (strip class/id that says nav)
        new_tag = re.sub(r'<div', '<nav', div_tag, count=1, flags=re.IGNORECASE)
        html = html[:start] + new_tag + html[end:]

        # Find the matching </div> for this nav and replace it
        depth = 1
        pos = start + len(new_tag)
        while depth > 0 and pos < len(html):
            next_open = html.find('<div', pos)
            next_close = html.find('</div', pos)
            if next_close == -1:
                break
            if next_open != -1 and next_open < next_close:
                depth += 1
                pos = next_open + 4
            else:
                depth -= 1
                if depth == 0:
                    # Replace this </div> with </nav>
                    html = html[:next_close] + '</nav>' + html[next_close + 6:]
                    break
                pos = next_close + 6

    return html


# ═══════════════════════════════════════════════════════════════════════
# Error 3: Missing HTML document structure — model returns a fragment
# instead of a complete HTML document
# ═══════════════════════════════════════════════════════════════════════

def _detect_missing_document_structure(html: str) -> bool:
    """Detect a response that's an HTML fragment rather than a full document.
    The model produced <div> or <style> tags but no <!DOCTYPE>, <html>, <head>, <body>."""
    has_doctype = bool(re.search(r"<!DOCTYPE\s+html>", html, re.IGNORECASE))
    has_html = bool(re.search(r"<html[^>]*>", html, re.IGNORECASE))
    has_body = bool(re.search(r"<body[^>]*>", html, re.IGNORECASE))
    # Has HTML content but missing document wrapper
    has_content = bool(re.search(r"<(?:div|h[1-6]|style|section|article|p|nav|header|footer|form|table|canvas|script)[\s>]", html, re.IGNORECASE))
    return has_content and not (has_doctype and has_html and has_body)


def _fix_missing_document_structure(html: str) -> str:
    """Wrap an HTML fragment in a complete HTML document structure.
    Also promotes the first <h2> to <h1> if no <h1> exists (since the page
    should have exactly one main heading)."""
    # Strip markdown code fences if present
    html = re.sub(r"^```(?:html)?\s*\n", "", html)
    html = re.sub(r"\n```\s*$", "", html)
    html = html.strip()

    # Promote first <h2> to <h1> if no <h1> exists
    if not re.search(r"<h1[^>]*>", html, re.IGNORECASE):
        h2_match = re.search(r"<h2([^>]*)>(.*?)</h2>", html, re.IGNORECASE | re.DOTALL)
        if h2_match:
            html = html[:h2_match.start()] + f"<h1{h2_match.group(1)}>{h2_match.group(2)}</h1>" + html[h2_match.end():]

    # Extract title from first <h1> if available
    title_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else "Page"
    # Strip tags from title
    title = re.sub(r"<[^>]+>", "", title)

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body>
{html}
</body>
</html>"""
    return document


# ═══════════════════════════════════════════════════════════════════════
# Error 4: Missing clearRect() — canvas game doesn't clear between frames
# ═══════════════════════════════════════════════════════════════════════

def _detect_missing_clear_rect(html: str) -> bool:
    """Detect a canvas game that has a game loop but doesn't call clearRect.
    This causes motion trails — the ball/snake leaves a permanent path on canvas.
    Checks both requestAnimationFrame and setInterval based loops."""
    has_raf = bool(re.search(r"requestAnimationFrame", html))
    has_setinterval = bool(re.search(r"setInterval\s*\(", html))
    has_clear = bool(re.search(r"clearRect\s*\(", html))
    has_canvas = bool(re.search(r"<canvas[^>]*>", html, re.IGNORECASE))
    has_get_context = bool(re.search(r"getContext\s*\(", html))
    has_loop = has_raf or has_setinterval
    return has_loop and has_canvas and has_get_context and not has_clear


def _fix_missing_clear_rect(html: str) -> str:
    """Inject a clearRect call at the start of the game loop / draw function.
    Finds the function that contains the game loop call (requestAnimationFrame
    or setInterval) and injects clearRect as the first drawing operation."""
    # Find the canvas id
    canvas_match = re.search(r'<canvas[^>]*\sid\s*=\s*["\']([^"\']+)["\']', html, re.IGNORECASE)
    canvas_id = canvas_match.group(1) if canvas_match else "gameCanvas"

    # Find the context variable name
    ctx_match = re.search(r"(?:var|let|const)\s+(\w+)\s*=\s*\w+\.getContext\s*\(\s*['\"]2d['\"]\s*\)", html)
    ctx_var = ctx_match.group(1) if ctx_match else "ctx"

    # Find canvas variable name
    canvas_var_match = re.search(r"(?:var|let|const)\s+(\w+)\s*=\s*document\.getElementById\s*\(\s*['\"]" + re.escape(canvas_id) + r"['\"]\s*\)", html)
    canvas_var = canvas_var_match.group(1) if canvas_var_match else canvas_id

    # Build the clearRect call
    clear_call = f"{ctx_var}.clearRect(0, 0, {canvas_var}.width, {canvas_var}.height);"

    # Find the function that contains the game loop call and inject clearRect
    func_pattern = r"function\s+(\w+)\s*\([^)]*\)\s*\{"
    funcs = list(re.finditer(func_pattern, html))

    for func_m in funcs:
        func_name = func_m.group(1)
        func_start = func_m.end()
        # Check if this function contains requestAnimationFrame(funcName) or
        # if this function is called by setInterval
        text_ahead = html[func_start:func_start + 3000]
        is_raf_loop = bool(re.search(r"requestAnimationFrame\s*\(\s*" + re.escape(func_name), text_ahead))
        # Check if this function is called by setInterval elsewhere in the code
        is_setinterval_loop = bool(re.search(r"setInterval\s*\(\s*" + re.escape(func_name), html))
        # Also check if function name suggests it's a draw/update/game function
        is_game_func = func_name.lower() in ("draw", "update", "gameloop", "game_loop", "tick", "render", "loop", "step")

        if is_raf_loop or is_setinterval_loop or is_game_func:
            html = html[:func_start] + f"\n    {clear_call}" + html[func_start:]
            break

    return html


# ═══════════════════════════════════════════════════════════════════════
# Error 5: Markdown code fences wrapping HTML output
# ═══════════════════════════════════════════════════════════════════════

def _detect_markdown_fences(html: str) -> bool:
    """Detect HTML wrapped in markdown ```html ... ``` fences.
    Not strictly an error, but fences should be stripped for clean HTML output."""
    return bool(re.match(r"^\s*```(?:html)?\s*\n", html, re.IGNORECASE))


def _fix_markdown_fences(html: str) -> str:
    """Strip markdown code fences from HTML output."""
    html = re.sub(r"^\s*```(?:html)?\s*\n", "", html)
    html = re.sub(r"\n```\s*$", "", html)
    return html.strip()


# ═══════════════════════════════════════════════════════════════════════
# Error 6: Missing viewport meta tag — page won't be responsive
# ═══════════════════════════════════════════════════════════════════════

def _detect_missing_viewport(html: str) -> bool:
    """Detect a page with <head> but no viewport meta tag."""
    has_head = bool(re.search(r"<head[^>]*>", html, re.IGNORECASE))
    has_viewport = bool(re.search(r'<meta\s+[^>]*viewport[^>]*>', html, re.IGNORECASE))
    has_doctype = bool(re.search(r"<!DOCTYPE\s+html>", html, re.IGNORECASE))
    return has_doctype and has_head and not has_viewport


def _fix_missing_viewport(html: str) -> str:
    """Inject viewport meta tag into <head>."""
    viewport_tag = '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
    if re.search(r"</head\s*>", html, re.IGNORECASE):
        html = re.sub(r"(</head\s*>)", viewport_tag + r"\n    \1", html, count=1, flags=re.IGNORECASE)
    return html


# ═══════════════════════════════════════════════════════════════════════
# Error 7: Missing charset meta tag
# ═══════════════════════════════════════════════════════════════════════

def _detect_missing_charset(html: str) -> bool:
    """Detect a page with <head> but no charset meta tag."""
    has_head = bool(re.search(r"<head[^>]*>", html, re.IGNORECASE))
    has_charset = bool(re.search(r'<meta\s+[^>]*charset[^>]*>', html, re.IGNORECASE))
    has_doctype = bool(re.search(r"<!DOCTYPE\s+html>", html, re.IGNORECASE))
    return has_doctype and has_head and not has_charset


def _fix_missing_charset(html: str) -> str:
    """Inject charset meta tag into <head>."""
    charset_tag = '<meta charset="UTF-8">'
    if re.search(r"</head\s*>", html, re.IGNORECASE):
        html = re.sub(r"(</head\s*>)", charset_tag + r"\n    \1", html, count=1, flags=re.IGNORECASE)
    return html


# ═══════════════════════════════════════════════════════════════════════
# Error 8: Missing <title> tag in <head>
# ═══════════════════════════════════════════════════════════════════════

def _detect_missing_title(html: str) -> bool:
    """Detect a page with <head> but no <title>."""
    has_head = bool(re.search(r"<head[^>]*>", html, re.IGNORECASE))
    has_title = bool(re.search(r"<title[^>]*>\s*\S", html, re.IGNORECASE))
    has_doctype = bool(re.search(r"<!DOCTYPE\s+html>", html, re.IGNORECASE))
    return has_doctype and has_head and not has_title


def _fix_missing_title(html: str) -> str:
    """Inject a <title> tag extracted from <h1> or default."""
    title = "Page"
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
    if h1_match:
        title = re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()
    title_tag = f"<title>{title}</title>"
    if re.search(r"</head\s*>", html, re.IGNORECASE):
        html = re.sub(r"(</head\s*>)", title_tag + r"\n    \1", html, count=1, flags=re.IGNORECASE)
    return html


# ═══════════════════════════════════════════════════════════════════════
# Error 9: Missing alt attribute on <img> tags
# ═══════════════════════════════════════════════════════════════════════

def _detect_missing_img_alt(html: str) -> bool:
    """Detect <img> tags without alt attributes."""
    imgs = re.findall(r"<img[^>]*>", html, re.IGNORECASE)
    if not imgs:
        return False
    return any(not re.search(r"\salt\s*=", img, re.IGNORECASE) for img in imgs)


def _fix_missing_img_alt(html: str) -> str:
    """Add alt="" to <img> tags that are missing it."""
    def add_alt(m):
        img_tag = m.group(0)
        if re.search(r"\salt\s*=", img_tag, re.IGNORECASE):
            return img_tag  # Already has alt
        # Try to guess alt from src
        src_match = re.search(r'src\s*=\s*["\']([^"\']+)["\']', img_tag, re.IGNORECASE)
        alt_text = ""
        if src_match:
            # Use filename as alt text (without extension)
            src = src_match.group(1)
            alt_text = re.sub(r"\.(jpg|jpeg|png|gif|svg|webp)$", "", src.split("/")[-1])
            alt_text = alt_text.replace("-", " ").replace("_", " ").capitalize()
        return img_tag.replace(">", f' alt="{alt_text}">', 1)

    return re.sub(r"<img[^>]*>", add_alt, html, flags=re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════
# The curated error database
# ═══════════════════════════════════════════════════════════════════════

HTML_ERROR_DATABASE: list[HTMLErrorPattern] = [
    HTMLErrorPattern(
        id="markdown-fences",
        description="HTML output wrapped in markdown ```html fences — should be stripped",
        affected_prompts=["profile-contact-form", "profile-social-links", "profile-styled-card"],
        detect=_detect_markdown_fences,
        fix=_fix_markdown_fences,
    ),
    HTMLErrorPattern(
        id="missing-document-structure",
        description="HTML fragment instead of full document — missing DOCTYPE, html, head, body",
        affected_prompts=["profile-styled-card"],
        detect=_detect_missing_document_structure,
        fix=_fix_missing_document_structure,
    ),
    HTMLErrorPattern(
        id="missing-color-styling",
        description="No CSS color styling — page renders unstyled (plain browser defaults)",
        affected_prompts=["profile-basic", "profile-contact-form", "profile-skills-list"],
        detect=_detect_missing_color_styling,
        fix=_fix_missing_color_styling,
    ),
    HTMLErrorPattern(
        id="missing-nav-tag",
        description="Navigation links in <div> instead of semantic <nav> tag",
        affected_prompts=["profile-social-links"],
        detect=_detect_missing_nav_tag,
        fix=_fix_missing_nav_tag,
    ),
    HTMLErrorPattern(
        id="missing-clear-rect",
        description="Canvas game uses requestAnimationFrame but never calls clearRect — causes motion trails",
        affected_prompts=["game-snake-clone"],
        detect=_detect_missing_clear_rect,
        fix=_fix_missing_clear_rect,
    ),
    HTMLErrorPattern(
        id="missing-viewport-meta",
        description="Page has <head> but no viewport meta tag — won't be responsive on mobile",
        affected_prompts=[],
        detect=_detect_missing_viewport,
        fix=_fix_missing_viewport,
    ),
    HTMLErrorPattern(
        id="missing-charset-meta",
        description="Page has <head> but no charset meta tag",
        affected_prompts=[],
        detect=_detect_missing_charset,
        fix=_fix_missing_charset,
    ),
    HTMLErrorPattern(
        id="missing-title-tag",
        description="Page has <head> but no <title> tag",
        affected_prompts=["profile-styled-card"],
        detect=_detect_missing_title,
        fix=_fix_missing_title,
    ),
    HTMLErrorPattern(
        id="missing-img-alt",
        description="<img> tags without alt attributes — accessibility issue",
        affected_prompts=[],
        detect=_detect_missing_img_alt,
        fix=_fix_missing_img_alt,
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# API: apply_html_autocorrect
# ═══════════════════════════════════════════════════════════════════════

def apply_html_autocorrect(html: str) -> tuple[str, list[dict]]:
    """Apply all applicable HTML error fixes to an HTML document.

    Returns (fixed_html, list_of_fixes_applied).
    Each fix entry has: {"id": ..., "description": ..., "fixable": bool}

    The order matters: markdown fences stripped first, then document structure
    wrapped, then content-level fixes applied.
    """
    fixes_applied: list[dict] = []
    current = html

    for ep in HTML_ERROR_DATABASE:
        if ep.detect(current):
            before = current
            current = ep.fix(current)
            was_fixed = current != before
            fixes_applied.append({
                "id": ep.id,
                "description": ep.description,
                "fixable": was_fixed,
            })

    return current, fixes_applied