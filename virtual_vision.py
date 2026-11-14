"""MiniMax H3 Image Conditioning Plus.

No reference: reuse the proven Qwen3-VL virtual vision patch route without
constructing an RGB image. At least one actual reference: delegate to ComfyUI's
own MiniMaxH3ImageConditioning.execute, verbatim.

The public node schema is ComfyUI V3, matching the upstream dynamic Autogrow
image inputs (image_0 through image_99). This differs intentionally from the
old fixed 4-input V1 schema in the previous developer ZIP.
"""

import torch
from comfy_api.latest import io
from comfy.text_encoders.minimax import MiniMaxQwen3VL
from comfy_extras import nodes_minimax_h3 as official
import comfy.model_management

VIRTUAL_SIZE = 1024
PATCH_SIZE = 16
TEMPORAL_PATCH = 2
GRID = VIRTUAL_SIZE // PATCH_SIZE
PATCH_ROWS = GRID * GRID
PATCH_WIDTH = 3 * TEMPORAL_PATCH * PATCH_SIZE * PATCH_SIZE


class _VirtualVision:
    """Marker consumed by Qwen's image preprocessing hook, not an IMAGE."""
    __slots__ = ()


VIRTUAL_VISION = _VirtualVision()


def _virtual_patchify(device):
    """Equivalent to the preprocessed black 1024² image; no RGB raster."""
    patches = torch.full((PATCH_ROWS, PATCH_WIDTH), -1.0,
                         dtype=torch.float32, device=device)
    grid = torch.tensor([[1, GRID, GRID]], dtype=torch.long)
    return patches, grid


def install_virtual_preprocessor():
    """Patch only the sentinel case. Other Qwen inputs use the original code."""
    method = MiniMaxQwen3VL.preprocess_embed
    if getattr(method, '_h3_virtual_vision_v1', False):
        return

    def preprocess(self, embed, device):
        if isinstance(embed, dict) and embed.get('data') is VIRTUAL_VISION:
            if embed.get('type') != 'image':
                raise RuntimeError('H3 Conditioning Plus: virtual marker has no image modality')
            if getattr(self, 'model_type', None) != 'qwen3vl_32b':
                raise RuntimeError('H3 Conditioning Plus: MiniMax H3 32B text encoder required')
            patches, grid = _virtual_patchify(device)
            merged, deepstack = self.visual(patches, grid)
            return merged, {'grid': grid, 'deepstack': deepstack}
        return method(self, embed, device)

    preprocess._h3_virtual_vision_v1 = True
    preprocess._h3_virtual_vision_original = method
    MiniMaxQwen3VL.preprocess_embed = preprocess


class H3ImageVirtualVisionConditioning(io.ComfyNode):
    """Legacy node ID kept to avoid breaking existing T2I workflows."""

    @classmethod
    def define_schema(cls):
        # Match official MiniMaxH3ImageConditioning V3 inputs, including the
        # Autogrow.TemplatePrefix sequence image_0 ... image_99.
        return io.Schema(
            node_id='H3ImageVirtualVisionConditioning',
            display_name='MiniMax H3 Image Conditioning Plus',
            category='model/conditioning/minimax',
            description=(
                'No image: neutral Qwen vision patches without constructing an RGB image. '
                'With image references: executes the official MiniMax H3 Image Conditioning '
                'node unchanged, including all reference/VAE behavior.'
            ),
            inputs=[
                io.Clip.Input('clip'),
                io.Vae.Input('vae', optional=True,
                    tooltip='Official reference VAE input; unchanged with real pictures.'),
                io.String.Input('prompt', multiline=True, dynamic_prompts=True),
                io.Int.Input('width', default=0, min=0, max=official.nodes.MAX_RESOLUTION,
                    step=official.CANVAS_MULTIPLE,
                    tooltip='0 follows the official rule: first image size or 2048 without references.'),
                io.Int.Input('height', default=0, min=0, max=official.nodes.MAX_RESOLUTION,
                    step=official.CANVAS_MULTIPLE,
                    tooltip='0 follows the official rule: first image size or 2048 without references.'),
                io.Autogrow.Input(
                    'images', optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input('image', tooltip='Image reference; the first connected image is <Picture 1>.'),
                        prefix='image_', min=0, max=100,
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name='positive'),
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, width, height, vae=None, images=None) -> io.NodeOutput:
        # Critical compatibility rule: do not reconstruct or transform real
        # references. Route the entire call to the official implementation.
        # It handles sorted slots, tokenization, VAE reference latents, and
        # original output size, including partial width/height overrides.
        if images and any(image is not None for image in images.values()):
            return official.MiniMaxH3ImageConditioning.execute(
                clip, prompt, width, height, vae=vae, images=images,
            )

        # Only the empty-reference branch deviates from upstream.
        install_virtual_preprocessor()
        tokens = clip.tokenize(prompt, images=[VIRTUAL_VISION], picture_label='Picture %d: ')
        cond = clip.encode_from_tokens_scheduled(tokens)

        # Preserve verification from the user-tested version; a changed Qwen
        # implementation should fail clearly instead of silently doing T2I.
        tags = cond[0][1].get('minimax_token_tags') if cond else None
        if tags is None or int((tags == 0).sum()) < 1000:
            raise RuntimeError('H3 Conditioning Plus: Qwen visual tokens missing; check ComfyUI version.')

        if not (width and height):
            width = width or official.IMAGE_SIZE
            height = height or official.IMAGE_SIZE

        latent = torch.zeros(
            [1, 24, 1, height // 16, width // 16],
            device=comfy.model_management.intermediate_device(),
        )
        print(f'[H3 Conditioning Plus] virtual vision (no RGB); '
              f'Qwen visual tokens={int((tags == 0).sum())}; '
              f'output={width}x{height}')
        return io.NodeOutput(cond, {'samples': latent})
