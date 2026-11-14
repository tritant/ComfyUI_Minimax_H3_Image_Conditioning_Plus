# ComfyUI MiniMax H3 Image Conditioning Plus

An enhanced replacement for the official **MiniMax H3 Image Conditioning** node in ComfyUI, designed to improve **texture quality, fine details, and overall rendering** in Text-to-Image (T2I) generation.

## Why Conditioning Plus?

When generating images without a reference, the original conditioning node can produce less convincing textures and fine details.

Using a neutral visual reference can significantly improve the rendering. **Conditioning Plus brings this benefit to T2I without requiring a reference image.**

It provides neutral vision patches directly to Qwen3-VL, activating the visual conditioning path without creating an RGB image.

## Installation

1. Download or clone this repository.
2. Place the repository folder in `ComfyUI/custom_nodes/`.
3. Restart ComfyUI.
4. In your workflow, replace **MiniMax H3 Image Conditioning** with **MiniMax H3 Image Conditioning Plus**.

**MiniMax H3 Image Conditioning Plus — Better textures, richer details, no reference image required.**
