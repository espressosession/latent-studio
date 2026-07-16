"""Gradio app layout: build_app() assembles the gr.Blocks UI and wires every
control to its handler in callbacks.py. One Generate button drives both modes
(single image / comparison grid) and both pipelines.

Layout rule: nothing containing a slider is ever hidden — Gradio mounts a
slider inside a zero-width display:none container and it stays invisible
until touched — so controls that don't apply right now are greyed out
instead, with the reason in their info line (see callbacks.gating_updates)."""

import random

import gradio as gr

from .callbacks import (
    dim_updates,
    gating_updates,
    generate_button_label,
    model_status_html,
    on_advanced_toggle,
    on_button_label_change,
    on_controlnet_change,
    on_gallery_select,
    on_generate,
    on_import_settings,
    on_preload,
    on_preprocess,
    on_randomize_seed,
)
from .design_tokens import (
    CONTROLNET_CHOICES,
    DESC_ADVANCED,
    DESC_AVOID,
    DESC_CFG,
    DESC_COMPARE,
    DESC_MODEL,
    DESC_PRELOAD,
    DESC_PROMPT,
    DESC_REFERENCE,
    DESC_REFERENCE_SCALE,
    DESC_SEED,
    DESC_STEPS,
    DESC_STYLE,
    DESC_STYLE_STRENGTH,
    EXAMPLE_PROMPTS,
    STATUS_REFERENCE_OFF,
    STATUS_SEED_RANDOM,
    STATUS_STYLE_OFF,
    SWEEP_CHOICES,
    THEME,
)
from .icons import ICON_ON_PRIMARY, button_icon
from .metadata_view import metadata_to_control_values, prompt_used_html, settings_html
from .progress import status_html
from .registry import CHECKPOINTS, DEFAULT_CHECKPOINT_ID, DEFAULT_LORA_ID, LORAS


def _title(text: str) -> gr.Markdown:
    """A control's title, as an h3 — container=False drops the normal component label."""
    return gr.Markdown(f"### {text}")


def _description(text: str) -> gr.Markdown:
    return gr.Markdown(text)


def _sublabel(text: str) -> gr.Markdown:
    """A small bold label for a sub-field (Compare tab's Start/End/Steps)."""
    return gr.Markdown(f"**{text}**")


def _heading(title: str, description: str) -> None:
    """Title + description pair shown above a control. Deliberately not wrapped in a
    gr.Group() — Group's own background/border turned out to come from a Gradio CSS
    quirk (--border-color-primary, shared with unrelated elements) with no clean theme
    lever to switch off, so plain spacing wins over a custom CSS chase."""
    _title(title)
    _description(description)


def _compare_column(heading: str, description: str, choices: list, value: str, spec, interactive: bool = True):
    """One column of the Compare tab. The first column's field radio is its own on/off
    switch ("Nothing" = off) and stays clickable; the second column's only makes sense
    once the first is active, so it starts (and re-)greys via `interactive`/
    dim_updates()'s gate_field. Start/End/Steps grey out until a real setting is picked,
    and dim_updates() re-ranges them for whichever setting that is."""
    _heading(heading, description)
    field = gr.Radio(choices=choices, value=value, container=False, info="", interactive=interactive)
    low, high, step = (spec.minimum, spec.maximum, spec.step) if spec else (1, 100, 1)
    start, end = (spec.start, spec.end) if spec else (10, 50)
    with gr.Row():
        with gr.Column(min_width=140):
            _sublabel("Start")
            from_slider = gr.Slider(low, high, value=start, step=step, container=False, interactive=False)
        with gr.Column(min_width=140):
            _sublabel("End")
            to_slider = gr.Slider(low, high, value=end, step=step, container=False, interactive=False)
        with gr.Column(min_width=100):
            _sublabel("Steps")
            count = gr.Number(
                value=spec.count if spec else 5, precision=0, minimum=2,
                maximum=spec.max_count if spec else 10, container=False, interactive=False,
            )
    return field, from_slider, to_slider, count


def build_app() -> gr.Blocks:
    checkpoint_choices = [(cp.label, cp.id) for cp in CHECKPOINTS]
    lora_choices = [(lora.label, lora.id) for lora in LORAS]
    compare_choices = [("Nothing", "off")] + SWEEP_CHOICES

    with gr.Blocks(title="Latent Studio") as demo:
        history_state = gr.State([])
        metadata_state = gr.State(None)  # the selected image's metadata; drives Reuse
        selected_index_state = gr.State(0)
        # Set True by Reuse/Import alongside a restored field1/field2 to tell that
        # column's own dim_updates() cascade to skip its usual value-reset just once —
        # see dim_updates()'s docstring.
        suppress1_state = gr.State(False)
        suppress2_state = gr.State(False)

        gr.Markdown(
            "# Latent Studio\n"
            "_Describe a scene, then let an artist paint it._ Six hand-trained styles — "
            "from Hokusai's waves to Rembrandt's candlelight — are ready to try alongside "
            "your own prompts, with every setting behind them yours to tune and reproduce. "
            "Press **Generate** to start; the first run downloads a model (a few minutes), "
            "every one after takes seconds."
        )

        # ---------------- main area ----------------
        with gr.Row():
            # LEFT — prompt, button, live state. Each control is a title + description +
            # field in sequence; Gradio's own layout gap gives them breathing room.
            with gr.Column(scale=3, variant="panel"):
                _heading("Prompt", DESC_PROMPT)
                prompt_input = gr.Textbox(
                    container=False,
                    value=random.choice(EXAMPLE_PROMPTS),
                    placeholder="Describe the image you want…",
                    lines=3,
                )
                _heading("Avoid", DESC_AVOID)
                negative_prompt_input = gr.Textbox(
                    container=False,
                    placeholder="e.g. blurry, low quality, text…",
                    lines=2,
                )
                generate_button = gr.Button(
                    generate_button_label(DEFAULT_CHECKPOINT_ID, DEFAULT_LORA_ID),
                    variant="primary",
                    size="lg",
                    icon=button_icon("sparkles", ICON_ON_PRIMARY),
                )
                state_info = gr.HTML(
                    status_html("check", "Ready."), container=False, padding=False,
                    apply_default_css=False, elem_id="state_info",
                )

            # CENTER — the picture and the prompt that made it.
            with gr.Column(scale=6):
                # Gradio's preview image is exactly gallery height minus a fixed 60px
                # thumbnail strip (measured in the shipped component CSS) — 572 renders
                # the 512px image at its true resolution instead of shrunk to ~500px.
                history_gallery = gr.Gallery(
                    show_label=False, container=False, preview=True, selected_index=0,
                    columns=8, height=572, object_fit="contain", buttons=[],
                )
                prompt_used = gr.HTML(
                    prompt_used_html(None), container=False, padding=False, apply_default_css=False
                )

            # RIGHT — Advanced only: exact settings, the way back to them, and the way to
            # keep the image itself.
            with gr.Column(scale=3, visible=False) as advanced_column:
                _title("Settings")
                settings_view = gr.HTML(
                    settings_html(None), container=False, padding=False, apply_default_css=False
                )
                apply_settings_button = gr.Button(
                    "Reuse these settings", size="sm", icon=button_icon("reuse")
                )
                with gr.Row():
                    # Uploads a settings JSON or a PNG this app produced (its embedded
                    # metadata is read back out) and applies it like "Reuse" does —
                    # copying the box above by hand already covers the old Copy button.
                    import_button = gr.UploadButton(
                        "Import", size="sm", icon=button_icon("upload"), file_types=[".json", ".png"],
                    )
                    export_button = gr.DownloadButton("Export", size="sm", icon=button_icon("download"))
                download_image_button = gr.DownloadButton(
                    "Download image", size="sm", icon=button_icon("download"), visible=False
                )

        # ---------------- settings ----------------
        with gr.Tabs():
            with gr.Tab("Model & style"):
                with gr.Row():
                    with gr.Column():
                        _heading("Model", DESC_MODEL)
                        checkpoint_radio = gr.Radio(
                            choices=checkpoint_choices, value=DEFAULT_CHECKPOINT_ID,
                            container=False, info="",
                        )
                        model_status = gr.HTML(
                            model_status_html(), container=False, padding=False, apply_default_css=False
                        )
                    with gr.Column():
                        _heading("Style", DESC_STYLE)
                        lora_radio = gr.Radio(
                            choices=lora_choices, value=DEFAULT_LORA_ID, container=False, info="",
                        )
                        _heading("Style strength", DESC_STYLE_STRENGTH)
                        lora_weight_slider = gr.Slider(
                            minimum=0.0, maximum=1.5, step=0.05, value=1.0, container=False,
                            info=STATUS_STYLE_OFF, interactive=False,
                        )

            with gr.Tab("Tuning"):
                with gr.Row():
                    with gr.Column():
                        _heading("Prompt strength", DESC_CFG)
                        cfg_slider = gr.Slider(
                            minimum=1.0, maximum=20.0, step=0.05, value=7.5, container=False, info="",
                        )
                        _heading("Detail", DESC_STEPS)
                        steps_slider = gr.Slider(
                            minimum=1, maximum=100, step=1, value=30, container=False, info="",
                        )
                    with gr.Column():
                        _heading("Seed", DESC_SEED)
                        seed_mode = gr.Radio(
                            choices=[("A new one every time", "random"), ("Always the same", "fixed")],
                            value="random", container=False, info="",
                        )
                        seed_input = gr.Number(
                            value=1312, precision=0, container=False,
                            info=STATUS_SEED_RANDOM, interactive=False,
                        )
                        randomize_button = gr.Button(
                            "Roll a new seed", size="sm", icon=button_icon("dice"), interactive=False
                        )

            # ControlNet (ADV) is disabled for the submission — unstable on the Colab
            # demo runtime; the core generator is complete without it. Components stay
            # defined so the wiring below still resolves; on_generate hardwires the
            # dispatch to the plain pipeline regardless of these controls.
            with gr.Tab("Reference image", visible=False):
                with gr.Row():
                    with gr.Column():
                        _heading("Copy a shape from an image", DESC_REFERENCE)
                        controlnet_select = gr.Radio(
                            choices=CONTROLNET_CHOICES, value="off", container=False, info="",
                        )
                    with gr.Column():
                        _heading("Reference strength", DESC_REFERENCE_SCALE)
                        controlnet_scale = gr.Slider(
                            0.0, 2.0, value=1.0, step=0.05, container=False,
                            info=STATUS_REFERENCE_OFF, interactive=False,
                        )
                with gr.Row(visible=False) as controlnet_group:
                    with gr.Column():
                        controlnet_input_image = gr.Image(
                            label="Your reference", type="pil", sources=["upload"], height=300
                        )
                        controlnet_preprocess_button = gr.Button("Process", size="sm")
                    with gr.Column():
                        controlnet_preview = gr.Image(
                            label="What the model will actually follow", type="pil",
                            interactive=False, height=300,
                        )
                gr.Markdown(
                    "_Your chosen style still applies — the reference fixes the composition, "
                    "the style paints it._"
                )

            with gr.Tab("Compare"):
                _heading("Compare settings", DESC_COMPARE)
                with gr.Row():
                    with gr.Column():
                        field1, from1, to1, count1 = _compare_column(
                            "Compare this",
                            'The setting to walk across the grid, left to right. Leave it on '
                            '"Nothing" for a single image.',
                            compare_choices, "off", None,
                        )
                    with gr.Column():
                        field2, from2, to2, count2 = _compare_column(
                            "And this",
                            "Optional — add a second setting to lay it out top to bottom as well, "
                            "turning the row into a full table.",
                            compare_choices, "off", None, interactive=False,
                        )

            with gr.Tab("Setup"):
                with gr.Row():
                    with gr.Column():
                        _heading("Advanced view", DESC_ADVANCED)
                        advanced_toggle = gr.Checkbox(
                            label="Show each image's exact settings",
                            value=False, container=False, info="",
                        )
                    with gr.Column():
                        _heading("Preload models", DESC_PRELOAD)
                        preload_button = gr.Button(
                            "Preload all models", size="sm", icon=button_icon("download")
                        )

        # ---------------- wiring ----------------
        gating_inputs = [field1, field2, lora_radio, seed_mode, controlnet_select]
        gating_outputs = [
            cfg_slider, steps_slider, lora_weight_slider,
            seed_mode, seed_input, randomize_button, controlnet_scale,
        ]
        label_inputs = [checkpoint_radio, lora_radio, field1, count1, field2, count2]
        dim1_outputs = [field1, from1, to1, count1, suppress1_state]
        dim2_outputs = [field2, from2, to2, count2, suppress2_state]
        # Reuse/Import restore field1/field2/Start/End/Steps from the grid's own stored
        # sweep config (or reset Compare to off for a single image / an older export
        # with no stored config) — see metadata_view.metadata_to_control_values. The
        # suppress flags ride along so the field-radio's own re-ranging cascade doesn't
        # immediately clobber the values just restored in this same update.
        apply_outputs = [
            checkpoint_radio, lora_radio, lora_weight_slider, prompt_input, negative_prompt_input,
            cfg_slider, steps_slider, seed_mode, seed_input, controlnet_select, controlnet_scale,
            field1, from1, to1, count1, field2, from2, to2, count2,
            suppress1_state, suppress2_state,
        ]

        # Every control that can grey another one out recomputes the whole gating set.
        # show_progress="hidden" throughout this block: these are instant UI-state
        # recomputations, not loading work, so the components they touch shouldn't
        # flash into Gradio's default pending overlay.
        for trigger in (field1, field2, lora_radio, seed_mode, controlnet_select):
            trigger.change(gating_updates, inputs=gating_inputs, outputs=gating_outputs, show_progress="hidden")

        # The button's label carries both what a click will do (load model/style?) and
        # how many images it produces — recomputed by anything that affects either.
        for trigger in (checkpoint_radio, lora_radio, field1, field2, count1, count2):
            trigger.change(
                on_button_label_change, inputs=label_inputs, outputs=generate_button, show_progress="hidden"
            )

        # field1 is Compare's own on/off switch now — picking a setting re-ranges its
        # own Start/End/Steps. field2 only re-ranges on its own change; a field1 change
        # just re-evaluates whether field2's column is usable, without resetting it.
        field1.change(
            lambda f, s: dim_updates(f, f, reset=True, suppress=s),
            inputs=[field1, suppress1_state], outputs=dim1_outputs,
            show_progress="hidden",
        )
        field1.change(
            lambda f1, f2, s: dim_updates(f1, f2, reset=False, gate_field=True, suppress=s),
            inputs=[field1, field2, suppress2_state], outputs=dim2_outputs,
            show_progress="hidden",
        )
        field2.change(
            lambda f1, f2, s: dim_updates(f1, f2, reset=True, gate_field=True, suppress=s),
            inputs=[field1, field2, suppress2_state], outputs=dim2_outputs,
            show_progress="hidden",
        )

        advanced_toggle.change(
            on_advanced_toggle, inputs=advanced_toggle, outputs=advanced_column, show_progress="hidden"
        )
        preload_button.click(
            on_preload, outputs=[preload_button, state_info], show_progress="hidden"
        )  # the button + state panel draw their own progress
        randomize_button.click(on_randomize_seed, outputs=seed_input, show_progress="hidden")
        controlnet_select.change(
            on_controlnet_change, inputs=controlnet_select,
            outputs=[controlnet_group, controlnet_preview],
            show_progress="hidden",
        )
        controlnet_preprocess_button.click(
            on_preprocess, inputs=[controlnet_input_image, controlnet_select], outputs=controlnet_preview,
            show_progress="hidden",
        )

        generate_button.click(
            on_generate,
            inputs=[
                checkpoint_radio, lora_radio, prompt_input, negative_prompt_input,
                cfg_slider, steps_slider, seed_mode, seed_input, lora_weight_slider,
                controlnet_select, controlnet_preview, controlnet_scale,
                field1, from1, to1, count1,
                field2, from2, to2, count2,
                history_state,
            ],
            outputs=[
                metadata_state, history_state, history_gallery, selected_index_state,
                download_image_button, export_button,
                generate_button, model_status, state_info,
                prompt_used, settings_view, seed_input,
            ],
            show_progress="hidden",  # the state panel draws its own bars
        )
        history_gallery.select(
            on_gallery_select, inputs=history_state,
            outputs=[
                metadata_state, selected_index_state, download_image_button, export_button,
                prompt_used, settings_view,
            ],
            show_progress="hidden",
        )
        apply_settings_button.click(
            metadata_to_control_values, inputs=metadata_state, outputs=apply_outputs, show_progress="hidden"
        )
        import_button.upload(
            on_import_settings, inputs=import_button, outputs=apply_outputs + [state_info, generate_button],
            show_progress="hidden",
        )

    return demo


if __name__ == "__main__":
    # Gradio 6 takes theme on launch(), not on the Blocks constructor. inline=False +
    # inbrowser=True: a real browser tab, not an inline iframe (the default inside a
    # notebook); footer_links=[] drops Gradio's own API/Settings footer.
    build_app().launch(theme=THEME, inline=False, inbrowser=True, footer_links=[])
