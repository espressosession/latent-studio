"""One monochrome stroke-icon set for the whole UI, replacing emoji. Button icons
are baked as a `data:` URI (no file written to disk); the state panel uses inline
`<svg>` instead, where `currentColor` works."""

from urllib.parse import quote

# Icon color for the two button kinds. The secondary tone is a mid warm grey that
# stays legible on both the light and the dark button fill.
ICON_ON_PRIMARY = "#ffffff"
ICON_ON_SECONDARY = "#8a817c"

# 24x24 viewBox, stroke-drawn. {c} is the stroke/fill color, substituted per use.
_MARKUP: dict[str, str] = {
    "sparkles": (
        '<path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9-1.9 5.1-1.9-5.1L5 10.5l5.1-1.9z"/>'
        '<path d="M18.5 15.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/>'
    ),
    "download": '<path d="M12 3v12"/><path d="M7 11l5 5 5-5"/><path d="M4 20h16"/>',
    "upload": '<path d="M12 15V3"/><path d="M7 8l5-5 5 5"/><path d="M4 20h16"/>',
    "reuse": '<path d="M3 3v6h6"/><path d="M21 12A9 9 0 0 0 6 5.3L3 9"/>',
    "copy": (
        '<rect x="9" y="9" width="12" height="12" rx="2"/>'
        '<path d="M5 15a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2"/>'
    ),
    "dice": (
        '<rect x="3" y="3" width="18" height="18" rx="3"/>'
        '<circle cx="8.5" cy="8.5" r="1.3" fill="{c}" stroke="none"/>'
        '<circle cx="12" cy="12" r="1.3" fill="{c}" stroke="none"/>'
        '<circle cx="15.5" cy="15.5" r="1.3" fill="{c}" stroke="none"/>'
    ),
    "check":'<circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/>',
    "alert": (
        '<path d="M10.3 4.3 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0z"/>'
        '<path d="M12 9.5v4"/><path d="M12 17h.01"/>'
    ),
}


def svg(name: str, size: int = 18, color: str = "currentColor") -> str:
    """Inline <svg> markup — for gr.HTML, where currentColor works."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.75" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="flex:none;vertical-align:middle">{_MARKUP[name].format(c=color)}</svg>'
    )


def button_icon(name: str, color: str = ICON_ON_SECONDARY) -> dict:
    """Icon value for gr.Button/gr.UploadButton/gr.DownloadButton(icon=...) — a data:
    URI in FileData clothing. The dict form matters: Block.serve_static_file() passes a
    dict straight through untouched, but treats a bare string as a literal filesystem
    path and tries to cache it as a file — which silently breaks for a data: URI. (Every
    button's icon= type hint says `str`, but the dict is what actually renders.)"""
    uri = "data:image/svg+xml;utf8," + quote(svg(name, size=20, color=color))
    return {"path": uri, "url": uri}
