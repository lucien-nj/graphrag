# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLMEmbedding based on Hugging Face Transformers."""

from typing import TYPE_CHECKING, Any, Unpack

from graphrag_llm.embedding.embedding import LLMEmbedding
from graphrag_llm.middleware import with_middleware_pipeline
from graphrag_llm.types import LLMEmbedding as LLMEmbeddingData
from graphrag_llm.types import LLMEmbeddingResponse, LLMEmbeddingUsage

if TYPE_CHECKING:
    from graphrag_cache import Cache, CacheKeyCreator

    from graphrag_llm.config import ModelConfig
    from graphrag_llm.metrics import MetricsProcessor, MetricsStore
    from graphrag_llm.rate_limit import RateLimiter
    from graphrag_llm.retry import Retry
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import (
        LLMEmbeddingArgs,
        Metrics,
    )


class TransformersLLMEmbedding(LLMEmbedding):
    """LLMEmbedding based on local Hugging Face Transformers models."""

    _model_config: "ModelConfig"
    _model_id: str
    _track_metrics: bool = False
    _metrics_store: "MetricsStore"
    _metrics_processor: "MetricsProcessor | None"
    _cache: "Cache | None"
    _cache_key_creator: "CacheKeyCreator"
    _tokenizer: "Tokenizer"
    _rate_limiter: "RateLimiter | None"
    _retrier: "Retry | None"

    def __init__(
        self,
        *,
        model_id: str,
        model_config: "ModelConfig",
        tokenizer: "Tokenizer",
        metrics_store: "MetricsStore",
        metrics_processor: "MetricsProcessor | None" = None,
        rate_limiter: "RateLimiter | None" = None,
        retrier: "Retry | None" = None,
        cache: "Cache | None" = None,
        cache_key_creator: "CacheKeyCreator",
        device: str | None = None,
        normalize_embeddings: bool = False,
        trust_remote_code: bool = False,
        use_fast: bool = True,
        **kwargs: Any,
    ):
        """Initialize TransformersLLMEmbedding.

        Args
        ----
            model_id: str
                The model ID, e.g., "local/bge-small-en"
            model_config: ModelConfig
                The configuration for the model.
            tokenizer: Tokenizer
                The tokenizer to use.
            metrics_store: MetricsStore | None (default: None)
                The metrics store to use.
            metrics_processor: MetricsProcessor | None (default: None)
                The metrics processor to use.
            cache: Cache | None (default: None)
                An optional cache instance.
            cache_key_creator: CacheKeyCreator
                The cache key creator function.
            rate_limiter: RateLimiter | None (default: None)
                The rate limiter to use.
            retrier: Retry | None (default: None)
                The retry strategy to use.
            device: str | None (default: None)
                The torch device to run inference on. If None, auto-selects cuda if available.
            normalize_embeddings: bool (default: False)
                Whether to L2-normalize the produced embeddings.
            trust_remote_code: bool (default: False)
                Whether to trust remote code when loading the model.
            use_fast: bool (default: True)
                Whether to use the fast Rust-based tokenizer. If loading fails,
                automatically falls back to the slow Python tokenizer.
        """
        self._model_id = model_id
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store
        self._metrics_processor = metrics_processor
        self._track_metrics = metrics_processor is not None
        self._cache = cache
        self._cache_key_creator = cache_key_creator
        self._rate_limiter = rate_limiter
        self._retrier = retrier

        self._embedding, self._embedding_async = _create_base_embeddings(
            model_config=model_config,
            device=device,
            normalize_embeddings=normalize_embeddings,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )

        self._embedding, self._embedding_async = with_middleware_pipeline(
            model_config=self._model_config,
            model_fn=self._embedding,
            async_model_fn=self._embedding_async,
            request_type="embedding",
            cache=self._cache,
            cache_key_creator=self._cache_key_creator,
            tokenizer=self._tokenizer,
            metrics_processor=self._metrics_processor,
            rate_limiter=self._rate_limiter,
            retrier=self._retrier,
        )

    def embedding(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> "LLMEmbeddingResponse":
        """Sync embedding method."""
        request_metrics: Metrics | None = kwargs.pop("metrics", None) or {}
        if not self._track_metrics:
            request_metrics = None

        try:
            return self._embedding(metrics=request_metrics, **kwargs)
        finally:
            if request_metrics:
                self._metrics_store.update_metrics(metrics=request_metrics)

    async def embedding_async(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> "LLMEmbeddingResponse":
        """Async embedding method."""
        request_metrics: Metrics | None = kwargs.pop("metrics", None) or {}
        if not self._track_metrics:
            request_metrics = None

        try:
            return await self._embedding_async(metrics=request_metrics, **kwargs)
        finally:
            if request_metrics:
                self._metrics_store.update_metrics(metrics=request_metrics)

    @property
    def metrics_store(self) -> "MetricsStore":
        """Get metrics store."""
        return self._metrics_store

    @property
    def tokenizer(self) -> "Tokenizer":
        """Get tokenizer."""
        return self._tokenizer


def _create_base_embeddings(
    *,
    model_config: "ModelConfig",
    device: str | None,
    normalize_embeddings: bool,
    trust_remote_code: bool,
    use_fast: bool,
) -> tuple:
    """Create base embedding functions using Hugging Face Transformers."""
    import asyncio

    import torch
    from torch.nn import functional
    from transformers import AutoModel, AutoTokenizer

    model_name = model_config.model
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
    except (ValueError, ImportError) as fast_err:
        if use_fast:
            try:
                hf_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code,
                    use_fast=False,
                )
            except Exception as slow_err:
                msg = (
                    f"Failed to load tokenizer for '{model_name}'. "
                    f"Fast tokenizer failed: {fast_err}. "
                    f"Slow tokenizer also failed: {slow_err}. "
                    "Please check that the model path is correct and that the required "
                    "dependencies (e.g., sentencepiece, tiktoken, tokenizers) are installed."
                )
                raise RuntimeError(msg) from slow_err
        else:
            raise

    hf_model = (
        AutoModel
        .from_pretrained(model_name, trust_remote_code=trust_remote_code)
        .to(device)
        .eval()
    )

    def _base_embedding(**kwargs: Any) -> LLMEmbeddingResponse:
        kwargs.pop("metrics", None)
        texts = kwargs.get("input")
        if isinstance(texts, str):
            texts = [texts]

        encoded = hf_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = hf_model(**encoded)

        attention_mask = encoded["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        if normalize_embeddings:
            embeddings = functional.normalize(embeddings, p=2, dim=1)

        total_tokens = int(attention_mask.sum().item())

        data = [
            LLMEmbeddingData(object="embedding", embedding=vec.tolist(), index=i)
            for i, vec in enumerate(embeddings)
        ]

        return LLMEmbeddingResponse(
            object="list",
            data=data,
            model=model_name,
            usage=LLMEmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
        )

    async def _base_embedding_async(**kwargs: Any) -> LLMEmbeddingResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: _base_embedding(**kwargs))

    return _base_embedding, _base_embedding_async
