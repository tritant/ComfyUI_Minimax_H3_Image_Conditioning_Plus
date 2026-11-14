"""MiniMax H3 Image Conditioning Plus — ComfyUI V3 extension."""
from comfy_api.latest import ComfyExtension, io
from .virtual_vision import H3ImageVirtualVisionConditioning


class MiniMaxH3ImageConditioningPlusExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [H3ImageVirtualVisionConditioning]


async def comfy_entrypoint() -> ComfyExtension:
    return MiniMaxH3ImageConditioningPlusExtension()
