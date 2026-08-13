"""HTML test prompt bank for Gemma 4 E2B code generation screening.

Two categories:
1. Profile pages — personal/portfolio pages with semantic HTML, CSS, structure
2. Game code — HTML5 canvas games with JS, event handlers, game loops

Each prompt has deterministic checks that verify the HTML output:
  - Structural: DOCTYPE, html/head/body tags, closing tags
  - Semantic: proper nesting, required meta tags, alt attributes
  - Game-specific: canvas element, script tags, event listeners, game loop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class HTMLTestPrompt:
    """A single HTML generation prompt with deterministic checks."""
    id: str
    category: str  # "profile" or "game"
    difficulty: str  # "easy", "medium", "hard"
    prompt: str
    # Each check returns (passed: bool, detail: str)
    checks: list[Callable[[str], tuple[bool, str]]]
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════
# Shared HTML check functions
# ═══════════════════════════════════════════════════════════════════════

import re

def _has_doctype(html: str) -> tuple[bool, str]:
    if re.search(r"<!DOCTYPE\s+html>", html, re.IGNORECASE):
        return True, "DOCTYPE present"
    return False, "Missing <!DOCTYPE html>"


def _has_html_tags(html: str) -> tuple[bool, str]:
    has_open = bool(re.search(r"<html[^>]*>", html, re.IGNORECASE))
    has_close = bool(re.search(r"</html\s*>", html, re.IGNORECASE))
    if has_open and has_close:
        return True, "<html> tags present"
    return False, f"Missing <html> tags (open={has_open}, close={has_close})"


def _has_head_body(html: str) -> tuple[bool, str]:
    has_head = bool(re.search(r"<head[^>]*>", html, re.IGNORECASE))
    has_body = bool(re.search(r"<body[^>]*>", html, re.IGNORECASE))
    if has_head and has_body:
        return True, "<head> and <body> present"
    return False, f"Missing head/body (head={has_head}, body={has_body})"


def _has_title(html: str) -> tuple[bool, str]:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m and m.group(1).strip():
        return True, f"Title: '{m.group(1).strip()[:40]}'"
    return False, "Missing or empty <title> tag"


def _has_viewport_meta(html: str) -> tuple[bool, str]:
    if re.search(r'<meta\s+[^>]*viewport[^>]*>', html, re.IGNORECASE):
        return True, "Viewport meta tag present"
    return False, "Missing viewport meta tag (needed for responsive design)"


def _has_charset_meta(html: str) -> tuple[bool, str]:
    if re.search(r'<meta\s+[^>]*charset[^>]*>', html, re.IGNORECASE):
        return True, "Charset meta tag present"
    return False, "Missing charset meta tag"


def _tags_balanced(html: str, tag: str) -> tuple[bool, str]:
    """Check that a specific tag type is balanced (open count == close count)."""
    opens = len(re.findall(rf"<{tag}[\s>]", html, re.IGNORECASE))
    closes = len(re.findall(rf"</{tag}\s*>", html, re.IGNORECASE))
    # Self-closing tags don't count (br, hr, img, input, meta, link)
    self_closing = len(re.findall(rf"<{tag}[^>]*/\s*>", html, re.IGNORECASE))
    opens -= self_closing
    if opens == closes:
        return True, f"<{tag}> balanced ({opens} pairs)"
    return False, f"<{tag}> unbalanced: {opens} open, {closes} close"


def _has_closing_body(html: str) -> tuple[bool, str]:
    if re.search(r"</body\s*>", html, re.IGNORECASE):
        return True, "</body> present"
    return False, "Missing </body> closing tag"


def _has_closing_html(html: str) -> tuple[bool, str]:
    if re.search(r"</html\s*>", html, re.IGNORECASE):
        return True, "</html> present"
    return False, "Missing </html> closing tag"


def _div_balanced(html: str) -> tuple[bool, str]:
    return _tags_balanced(html, "div")


def _section_balanced(html: str) -> tuple[bool, str]:
    return _tags_balanced(html, "section")


def _has_style_or_css_link(html: str) -> tuple[bool, str]:
    has_style = bool(re.search(r"<style[^>]*>", html, re.IGNORECASE))
    has_link = bool(re.search(r'<link[^>]*stylesheet[^>]*>', html, re.IGNORECASE))
    if has_style or has_link:
        return True, "CSS present (style tag or stylesheet link)"
    return False, "No CSS found (<style> or <link rel='stylesheet'>)"


def _has_canvas(html: str) -> tuple[bool, str]:
    m = re.search(r'<canvas[^>]*>', html, re.IGNORECASE)
    if m:
        # Check if it has an id
        if re.search(r'<canvas[^>]*\sid\s*=', html, re.IGNORECASE):
            return True, "<canvas> with id attribute present"
        return False, "<canvas> present but missing id attribute"
    return False, "Missing <canvas> element (needed for HTML5 games)"


def _has_script_tag(html: str) -> tuple[bool, str]:
    if re.search(r"<script[^>]*>", html, re.IGNORECASE):
        return True, "<script> tag present"
    return False, "Missing <script> tag (needed for game logic)"


def _has_script_closing(html: str) -> tuple[bool, str]:
    if re.search(r"</script\s*>", html, re.IGNORECASE):
        return True, "</script> present"
    return False, "Missing </script> closing tag"


def _has_event_listener(html: str) -> tuple[bool, str]:
    if re.search(r"addEventListener\s*\(", html):
        return True, "Event listener(s) present"
    return False, "No addEventListener() found (needed for game input)"


def _has_request_animation_frame(html: str) -> tuple[bool, str]:
    if re.search(r"requestAnimationFrame", html):
        return True, "requestAnimationFrame present (game loop)"
    return False, "No requestAnimationFrame found (needed for game loop)"


def _has_get_context(html: str) -> tuple[bool, str]:
    if re.search(r"getContext\s*\(\s*['\"]2d['\"]\s*\)", html):
        return True, "canvas.getContext('2d') present"
    return False, "Missing getContext('2d') call (needed to draw on canvas)"


def _has_img_alt(html: str) -> tuple[bool, str]:
    """Check that all <img> tags have alt attributes."""
    imgs = re.findall(r"<img[^>]*>", html, re.IGNORECASE)
    if not imgs:
        return True, "No <img> tags (alt check N/A)"
    missing_alt = [img for img in imgs if not re.search(r"\salt\s*=", img, re.IGNORECASE)]
    if missing_alt:
        return False, f"{len(missing_alt)}/{len(imgs)} <img> tags missing alt attribute"
    return True, f"All {len(imgs)} <img> tags have alt attributes"


def _no_broken_br(html: str) -> tuple[bool, str]:
    """Check for common broken br tag patterns."""
    # <br></br> is invalid (br is self-closing)
    broken = re.findall(r"<br\s*>\s*</br\s*>", html, re.IGNORECASE)
    if broken:
        return False, f"{len(broken)} broken <br></br> (should be <br> or <br/>)"
    return True, "No broken <br> tags"


def _has_nav_tag(html: str) -> tuple[bool, str]:
    if re.search(r"<nav[^>]*>", html, re.IGNORECASE):
        return True, "<nav> semantic tag present"
    return False, "Missing <nav> semantic tag (expected in profile pages)"


def _has_header_footer(html: str) -> tuple[bool, str]:
    has_header = bool(re.search(r"<header[^>]*>", html, re.IGNORECASE))
    has_footer = bool(re.search(r"<footer[^>]*>", html, re.IGNORECASE))
    if has_header and has_footer:
        return True, "<header> and <footer> present"
    return False, f"Missing header/footer (header={has_header}, footer={has_footer})"


def _has_main_tag(html: str) -> tuple[bool, str]:
    if re.search(r"<main[^>]*>", html, re.IGNORECASE):
        return True, "<main> semantic tag present"
    return False, "Missing <main> semantic tag"


def _has_h1(html: str) -> tuple[bool, str]:
    h1s = re.findall(r"<h1[^>]*>", html, re.IGNORECASE)
    if len(h1s) == 1:
        return True, "Exactly one <h1> present"
    if len(h1s) == 0:
        return False, "Missing <h1> tag (page should have one main heading)"
    return False, f"{len(h1s)} <h1> tags found (should be exactly 1)"


def _p_balanced(html: str) -> tuple[bool, str]:
    return _tags_balanced(html, "p")


def _ul_li_balanced(html: str) -> tuple[bool, str]:
    ul_open = len(re.findall(r"<ul[\s>]", html, re.IGNORECASE))
    ul_close = len(re.findall(r"</ul\s*>", html, re.IGNORECASE))
    li_open = len(re.findall(r"<li[\s>]", html, re.IGNORECASE))
    li_close = len(re.findall(r"</li\s*>", html, re.IGNORECASE))
    issues = []
    if ul_open != ul_close:
        issues.append(f"<ul>: {ul_open} open, {ul_close} close")
    if li_open != li_close:
        issues.append(f"<li>: {li_open} open, {li_close} close")
    if issues:
        return False, "; ".join(issues)
    return True, f"<ul>/<li> balanced ({ul_open} ul pairs, {li_open} li pairs)"


def _has_keydown_listener(html: str) -> tuple[bool, str]:
    """Check for keyboard event handling (needed for most games)."""
    if re.search(r"addEventListener\s*\(\s*['\"]keydown['\"]", html):
        return True, "keydown event listener present"
    return False, "No keydown listener (games need keyboard input)"


def _has_score_variable(html: str) -> tuple[bool, str]:
    """Check for a score tracking variable in game code."""
    if re.search(r"\bscore\b", html, re.IGNORECASE):
        return True, "Score variable found in game code"
    return False, "No score variable found (games should track score)"


def _has_collision_or_boundary(html: str) -> tuple[bool, str]:
    """Check for collision detection or boundary checking."""
    has_collision = bool(re.search(r"colli", html, re.IGNORECASE))
    has_boundary = bool(re.search(r"boundary|border|wall|edge", html, re.IGNORECASE))
    has_bounds = bool(re.search(r"canvas\.width|canvas\.height", html))
    if has_collision or has_boundary or has_bounds:
        return True, "Collision/boundary handling present"
    return False, "No collision detection or boundary checking found"


def _has_clear_rect(html: str) -> tuple[bool, str]:
    """Check for canvas clearing between frames (needed to avoid trails)."""
    if re.search(r"clearRect\s*\(", html):
        return True, "clearRect() present (canvas cleared each frame)"
    return False, "No clearRect() found (canvas will have motion trails)"


def _has_color_styling(html: str) -> tuple[bool, str]:
    """Check for some form of color styling (background, color, fill)."""
    has_css_color = bool(re.search(r"(?:background|color|fill)[^;{]*:", html, re.IGNORECASE))
    has_fill_style = bool(re.search(r"fillStyle\s*=", html))
    if has_css_color or has_fill_style:
        return True, "Color styling present"
    return False, "No color styling found (page/game will be unstyled)"


def _no_unclosed_quotes(html: str) -> tuple[bool, str]:
    """Detect attributes with unclosed quotes — common small-model error."""
    # Look for: attr="value without closing quote followed by >
    # Pattern: ="text  (no closing " before >)
    issues = []
    for m in re.finditer(r'(\w+)="([^"]*)$', html):
        issues.append(f"Unclosed quote in attribute '{m.group(1)}'")
    for m in re.finditer(r"(\w+)='([^']*)$", html):
        issues.append(f"Unclosed quote in attribute '{m.group(1)}'")
    if issues:
        return False, "; ".join(issues[:3])
    return True, "No unclosed quotes detected"


def _script_has_content(html: str) -> tuple[bool, str]:
    """Check that <script> tags aren't empty."""
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.IGNORECASE | re.DOTALL)
    if not scripts:
        return False, "No <script> tags to check"
    empty = [i for i, s in enumerate(scripts) if not s.strip()]
    if empty:
        return False, f"{len(empty)}/{len(scripts)} <script> tags are empty"
    return True, f"All {len(scripts)} <script> tags have content"


def _has_form_action(html: str) -> tuple[bool, str]:
    """Check that form tags have action attributes."""
    forms = re.findall(r"<form[^>]*>", html, re.IGNORECASE)
    if not forms:
        return True, "No <form> tags (check N/A)"
    missing = [f for f in forms if not re.search(r"\saction\s*=", f, re.IGNORECASE)]
    if missing:
        return False, f"{len(missing)}/{len(forms)} <form> tags missing action attribute"
    return True, f"All {len(forms)} <form> tags have action attributes"


def _has_button_type(html: str) -> tuple[bool, str]:
    """Check that button tags have type attributes."""
    buttons = re.findall(r"<button[^>]*>", html, re.IGNORECASE)
    if not buttons:
        return True, "No <button> tags (check N/A)"
    missing = [b for b in buttons if not re.search(r"\stype\s*=", b, re.IGNORECASE)]
    if missing:
        return False, f"{len(missing)}/{len(buttons)} <button> tags missing type attribute"
    return True, f"All {len(buttons)} <button> tags have type attributes"


# ═══════════════════════════════════════════════════════════════════════
# Profile Page Prompts (12)
# ═══════════════════════════════════════════════════════════════════════

PROFILE_PROMPTS: list[HTMLTestPrompt] = [
    HTMLTestPrompt(
        id="profile-basic",
        category="profile",
        difficulty="easy",
        prompt="Create a simple HTML profile page for a student named Alex. Include their name as a heading, a short bio paragraph, and a photo placeholder. Use semantic HTML tags. Return only the HTML code.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title, _has_h1,
                _p_balanced, _has_img_alt, _has_color_styling],
        description="Basic profile page with name, bio, photo",
    ),
    HTMLTestPrompt(
        id="profile-nav-header",
        category="profile",
        difficulty="medium",
        prompt="Create an HTML profile page with a navigation bar at the top with links to Home, About, Projects sections. Include a header with the person's name 'Jordan Chen'. Use semantic HTML5 tags (nav, header, main, footer). Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_nav_tag, _has_header_footer, _has_main_tag, _has_h1,
                _div_balanced, _has_color_styling],
        description="Profile page with nav, header, footer, semantic tags",
    ),
    HTMLTestPrompt(
        id="profile-portfolio-gallery",
        category="profile",
        difficulty="medium",
        prompt="Create an HTML portfolio page for a photographer. Include a gallery section with 4 image placeholders, each with alt text describing the photo. Add a section heading 'My Work'. Use proper semantic HTML. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_img_alt, _div_balanced, _section_balanced, _has_h1,
                _has_color_styling],
        description="Photographer portfolio with image gallery",
    ),
    HTMLTestPrompt(
        id="profile-contact-form",
        category="profile",
        difficulty="medium",
        prompt="Create an HTML contact page with a form containing name, email, and message fields. Include a submit button. Add labels for each field. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_form_action, _has_button_type, _has_color_styling,
                _div_balanced],
        description="Contact page with form fields and labels",
    ),
    HTMLTestPrompt(
        id="profile-responsive-meta",
        category="profile",
        difficulty="easy",
        prompt="Create a simple HTML profile page for a teacher named Ms. Park. Make it responsive by including the viewport meta tag. Include charset meta. Add a heading and a paragraph about her class. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_viewport_meta, _has_charset_meta, _has_h1, _p_balanced],
        description="Responsive profile with proper meta tags",
    ),
    HTMLTestPrompt(
        id="profile-skills-list",
        category="profile",
        difficulty="easy",
        prompt="Create an HTML profile page for a developer named Sam. Include a section titled 'Skills' with an unordered list of 5 programming languages. Use semantic HTML tags. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_h1, _ul_li_balanced, _section_balanced, _has_color_styling],
        description="Developer profile with skills list",
    ),
    HTMLTestPrompt(
        id="profile-two-column-layout",
        category="profile",
        difficulty="hard",
        prompt="Create an HTML profile page with a two-column layout using CSS. Left column has a sidebar with navigation links. Right column has the main content with a bio and photo. Use div elements with classes and internal CSS. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _div_balanced, _has_style_or_css_link, _has_h1, _has_color_styling,
                _has_img_alt],
        description="Two-column profile layout with CSS",
    ),
    HTMLTestPrompt(
        id="profile-social-links",
        category="profile",
        difficulty="medium",
        prompt="Create an HTML profile page for a content creator. Include clickable social media links (GitHub, YouTube, Twitter) that open in new tabs. Add the creator's name as h1 and a short bio. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_h1, _has_nav_tag, _has_color_styling, _div_balanced],
        description="Profile page with external social media links",
    ),
    HTMLTestPrompt(
        id="profile-full-semantic",
        category="profile",
        difficulty="hard",
        prompt="Create a complete HTML5 profile page using all semantic tags: header, nav, main, section, article, aside, and footer. The page is for a student named Riley who blogs about coding. Include a blog post excerpt in an article tag. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_nav_tag, _has_header_footer, _has_main_tag, _has_h1,
                _section_balanced, _div_balanced, _has_color_styling],
        description="Full semantic HTML5 profile page",
    ),
    HTMLTestPrompt(
        id="profile-styled-card",
        category="profile",
        difficulty="medium",
        prompt="Create an HTML profile card for a team member named Dana. Use a div with class 'card' that contains a photo placeholder, name, job title, and a short description. Style it with CSS inside a style tag — give the card a border, padding, and rounded corners. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _div_balanced, _has_style_or_css_link, _has_img_alt,
                _has_color_styling, _has_h1],
        description="Styled profile card with CSS",
    ),
    HTMLTestPrompt(
        id="profile-table-schedule",
        category="profile",
        difficulty="medium",
        prompt="Create an HTML page showing a student's weekly class schedule using a table. Include a header row with days of the week. Add at least 3 rows of class data. Style the table with CSS borders. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_color_styling, _has_style_or_css_link, _has_h1,
                _div_balanced],
        description="Student schedule table with CSS styling",
    ),
    HTMLTestPrompt(
        id="profile-complete-portfolio",
        category="profile",
        difficulty="hard",
        prompt="Create a complete personal portfolio HTML page for a web developer named Max. Include: a header with name and tagline, navigation links, an about section, a projects section with 3 project cards, a skills section with a list, and a footer with copyright. Use semantic HTML5 and internal CSS. Make it responsive with viewport meta. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_viewport_meta, _has_charset_meta, _has_nav_tag,
                _has_header_footer, _has_main_tag, _has_h1, _section_balanced,
                _div_balanced, _ul_li_balanced, _has_style_or_css_link,
                _has_color_styling],
        description="Complete portfolio page with all sections",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# HTML Game Prompts (12)
# ═══════════════════════════════════════════════════════════════════════

GAME_PROMPTS: list[HTMLTestPrompt] = [
    HTMLTestPrompt(
        id="game-bouncing-ball",
        category="game",
        difficulty="easy",
        prompt="Create an HTML5 canvas game where a ball bounces around the screen. The ball should change direction when it hits the canvas edges. Use JavaScript with requestAnimationFrame for the game loop. Include the canvas and script in a single HTML file. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_canvas, _has_script_tag, _has_script_closing,
                _has_get_context, _has_request_animation_frame,
                _has_clear_rect, _has_collision_or_boundary],
        description="Bouncing ball on canvas",
    ),
    HTMLTestPrompt(
        id="game-click-counter",
        category="game",
        difficulty="easy",
        prompt="Create an HTML5 game where the player clicks a moving target to score points. Use canvas for the target, and display the score on screen. Use addEventListener for click events. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_canvas, _has_script_tag, _has_script_closing,
                _has_get_context, _has_event_listener, _has_score_variable,
                _has_clear_rect],
        description="Click target game with score",
    ),
    HTMLTestPrompt(
        id="game-keyboard-move",
        category="game",
        difficulty="medium",
        prompt="Create an HTML5 canvas game where the player moves a square left and right using arrow keys. Show the square on a canvas. Use keydown event listeners. Use requestAnimationFrame for smooth movement. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_canvas, _has_script_tag, _has_script_closing,
                _has_get_context, _has_keydown_listener,
                _has_request_animation_frame, _has_clear_rect],
        description="Keyboard-controlled square movement",
    ),
    HTMLTestPrompt(
        id="game-simple-platformer",
        category="game",
        difficulty="hard",
        prompt="Create a simple HTML5 platformer game on canvas. The player is a square that can jump with the spacebar and move with arrow keys. Draw a platform the player can stand on. Use gravity in the game logic. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_canvas, _has_script_tag, _has_script_closing,
                _has_get_context, _has_keydown_listener,
                _has_request_animation_frame, _has_clear_rect,
                _has_collision_or_boundary],
        description="Simple platformer with gravity and jumping",
    ),
    HTMLTestPrompt(
        id="game-snake-clone",
        category="game",
        difficulty="hard",
        prompt="Create a simple Snake game using HTML5 canvas. The snake moves automatically and the player steers with arrow keys. Food appears randomly. The score increases when food is eaten. Include the game loop with requestAnimationFrame or setInterval. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_canvas, _has_script_tag, _has_script_closing,
                _has_get_context, _has_keydown_listener, _has_score_variable,
                _has_clear_rect, _has_collision_or_boundary],
        description="Snake game clone",
    ),
    HTMLTestPrompt(
        id="game-color-matcher",
        category="game",
        difficulty="medium",
        prompt="Create an HTML5 game where colored squares appear on canvas and the player must click the one matching a displayed color name. Track the score. Use addEventListener for clicks. Use fillStyle to set colors. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_canvas, _has_script_tag, _has_script_closing,
                _has_get_context, _has_event_listener, _has_score_variable,
                _has_color_styling, _has_clear_rect],
        description="Color matching click game",
    ),
    HTMLTestPrompt(
        id="game-avoid-obstacles",
        category="game",
        difficulty="hard",
        prompt="Create an HTML5 canvas game where the player (a square) moves with arrow keys to avoid falling obstacles. Obstacles spawn at the top and fall down. If the player gets hit, the game ends. Track and display the score. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_canvas, _has_script_tag, _has_script_closing,
                _has_get_context, _has_keydown_listener, _has_score_variable,
                _has_request_animation_frame, _has_clear_rect,
                _has_collision_or_boundary],
        description="Obstacle avoidance game",
    ),
    HTMLTestPrompt(
        id="game-breakout-clone",
        category="game",
        difficulty="hard",
        prompt="Create a simple Breakout game on HTML5 canvas. The player controls a paddle at the bottom with arrow keys. A ball bounces off walls, paddle, and bricks at the top. Destroy all bricks to win. Track the score. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_canvas, _has_script_tag, _has_script_closing,
                _has_get_context, _has_keydown_listener, _has_score_variable,
                _has_request_animation_frame, _has_clear_rect,
                _has_collision_or_boundary],
        description="Breakout/brick breaker clone",
    ),
    HTMLTestPrompt(
        id="game-memory-cards",
        category="game",
        difficulty="medium",
        prompt="Create an HTML5 memory card matching game. Use a grid of cards (div elements, not canvas). When a card is clicked, it flips. If two cards match, they stay face up. Track the number of moves. Use JavaScript with addEventListener. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_script_tag, _has_script_closing, _has_event_listener,
                _has_color_styling, _div_balanced, _has_style_or_css_link],
        description="Memory card matching game",
    ),
    HTMLTestPrompt(
        id="game-tic-tac-toe",
        category="game",
        difficulty="medium",
        prompt="Create an HTML5 Tic-Tac-Toe game. Use a 3x3 grid of buttons or divs. Players alternate clicking cells. JavaScript checks for a winner after each move. Display whose turn it is. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_script_tag, _has_script_closing, _has_event_listener,
                _has_color_styling, _div_balanced],
        description="Tic-Tac-Toe game",
    ),
    HTMLTestPrompt(
        id="game-whack-a-mole",
        category="game",
        difficulty="medium",
        prompt="Create an HTML5 Whack-a-Mole game. Use a grid of holes (div elements). A mole randomly appears in one hole. The player clicks to whack it and score points. Use setInterval to spawn moles. Track the score. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_script_tag, _has_script_closing, _has_event_listener,
                _has_score_variable, _has_color_styling, _div_balanced],
        description="Whack-a-Mole game",
    ),
    HTMLTestPrompt(
        id="game-reaction-time",
        category="game",
        difficulty="easy",
        prompt="Create an HTML5 reaction time game. A colored box appears on screen after a random delay. The player clicks it as fast as possible. Display their reaction time in milliseconds. Use JavaScript and addEventListener. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_script_tag, _has_script_closing, _has_event_listener,
                _has_color_styling, _div_balanced],
        description="Reaction time test game",
    ),
    HTMLTestPrompt(
        id="game-paddle-pong",
        category="game",
        difficulty="hard",
        prompt="Create a simple Pong game on HTML5 canvas. Two paddles (left controlled by W/S keys, right by arrow keys). A ball bounces between them. Score points when the ball passes a paddle. Use requestAnimationFrame. Return only the HTML.",
        checks=[_has_doctype, _has_html_tags, _has_head_body, _has_title,
                _has_canvas, _has_script_tag, _has_script_closing,
                _has_get_context, _has_keydown_listener, _has_score_variable,
                _has_request_animation_frame, _has_clear_rect,
                _has_collision_or_boundary],
        description="Two-player Pong game",
    ),
]


# All prompts combined
ALL_PROMPTS: list[HTMLTestPrompt] = PROFILE_PROMPTS + GAME_PROMPTS


def get_prompts_by_category(category: str) -> list[HTMLTestPrompt]:
    """Filter prompts by category."""
    return [p for p in ALL_PROMPTS if p.category == category]


def list_categories() -> list[str]:
    """Return all categories."""
    return ["profile", "game"]