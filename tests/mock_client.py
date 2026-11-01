"""Scripted stand-in for VLMClient: no server, no GPU. Returns queued answers in order and
records every request, so search loops can be tested exactly."""

from __future__ import annotations

from foveated_search.search.geometry import GRID
from foveated_search.serving.client import CallResult, SentImage
from foveated_search.serving.images import EncodedImage, encode_png


class MockClient:
    def __init__(self, answers: list[str] | None = None, default: str = '{"absent":true}'):
        self.answers = list(answers or [])
        self.default = default
        self.requests: list[dict] = []

    def encode(self, img) -> EncodedImage:
        return encode_png(img)

    async def chat(self, images, text, *, max_tokens=None, extra_body=None, **kw) -> CallResult:
        enc = [im if isinstance(im, EncodedImage) else self.encode(im) for im in images]
        self.requests.append({"images": enc, "text": text, "extra_body": extra_body})
        answer = self.answers.pop(0) if self.answers else self.default
        sent = []
        for e in enc:
            rw, rh = max(GRID, round(e.width / GRID) * GRID), max(GRID, round(e.height / GRID) * GRID)
            sent.append(SentImage(e.width, e.height, e.pixels, e.num_bytes, rw, rh, rw * rh // (GRID * GRID)))
        n_out = max(1, len(answer) // 2)
        return CallResult(answer, 20 + sum(s.vision_tokens for s in sent), n_out,
                          sum(s.vision_tokens for s in sent), 0.01, 0.01 * n_out, "stop", sent)
