import re
from html import escape

import pandas as pd

from formatting import configurations_label, display_model_build_name, display_model_family_name


def compact_kpi(label: str, value) -> str:
    return (
        '<div class="compact-kpi">'
        f'<div class="label">{escape(str(label))}</div>'
        f'<div class="value">{escape(str(value))}</div>'
        "</div>"
    )


def champion_summary_item(label: str, value) -> str:
    return (
        "<div>"
        f'<div class="label">{escape(str(label))}</div>'
        f'<div class="value">{escape(str(value))}</div>'
        "</div>"
    )


def summary_panel_from_markdown(markdown_text: str) -> str:
    title = "Summary"
    blocks = []
    current_item = None
    current_note = None

    def render_inline_markdown(text: str) -> str:
        html = escape(text)
        html = re.sub(
            r"\[([^\]]+)\]\((https?://[^)]+)\)",
            r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
            html,
        )
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", html)
        return html

    def flush_item() -> None:
        nonlocal current_item
        if current_item:
            blocks.append(("item", current_item))
            current_item = None

    def flush_note() -> None:
        nonlocal current_note
        if current_note:
            blocks.append(("note", current_note))
            current_note = None

    for raw_line in markdown_text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            flush_item()
            flush_note()
            continue
        if line.startswith("### "):
            flush_item()
            flush_note()
            title = line.replace("### ", "", 1).strip()
        elif line.startswith("- "):
            flush_item()
            flush_note()
            current_item = line.replace("- ", "", 1).strip()
        elif current_item:
            current_item += " " + line
        else:
            current_note = f"{current_note} {line}" if current_note else line
    flush_item()
    flush_note()

    body_parts = []
    in_list = False
    for block_type, text in blocks:
        if block_type == "item":
            if not in_list:
                body_parts.append("<ul>")
                in_list = True
            body_parts.append(f"<li>{render_inline_markdown(text)}</li>")
        else:
            if in_list:
                body_parts.append("</ul>")
                in_list = False
            body_parts.append(f'<p class="summary-note">{render_inline_markdown(text)}</p>')
    if in_list:
        body_parts.append("</ul>")

    if " | " in title:
        title_label, title_context = title.split(" | ", 1)
        title_html = (
            '<div class="summary-title-inline">'
            f'<span class="summary-title-label">{escape(title_label)}</span>'
            '<span class="summary-title-divider">|</span>'
            f'<span class="summary-title-context">{escape(title_context)}</span>'
            "</div>"
        )
    else:
        title_html = f'<div class="summary-title">{escape(title)}</div>'

    return (
        '<div class="champion-summary">'
        f"{title_html}"
        f"{''.join(body_parts)}"
        "</div>"
    )


def model_scope_summary_html(model_scope: pd.DataFrame) -> str:
    rows = []
    for family, group in model_scope.groupby("model_family", sort=False):
        build_parts = [
            f"{display_model_build_name(row.model_build)} ({configurations_label(row.configurations)})"
            for row in group.itertuples(index=False)
        ]
        rows.append(
            "<p>"
            f"<strong>{escape(display_model_family_name(family))}:</strong> "
            f"{escape(', '.join(build_parts))}"
            "</p>"
        )
    return "".join(rows)
