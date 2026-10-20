## Concepts

Internal fragmentation: Memory reserved for a request but will never be used. e.g. you reserve 1024 slots and the request only take up 300 tokens.

External Fragmentation: smaller chunks between reserved memories remain unused because allocation must be contiguous. e.g. a slot in the middle is held up since step 0.

Reservation waste: memory reserved and will be used but later on during the inference. e.g. You reserved slot 120 but it stays empty till the decode step reaches token 120.

Block table: A table that maps logical blocks to physical blocks and their addresses in the memory. It's like the one the OS uses to map virtual pages to physical pages. LLM requests are just like OS processes. each entry records how many slots are occupied and how many requests reference it. 

- Paging needs a custom attention kernel because it needs to identify and fetch different KV blocks separately. Standard kernels expect  KV in contiguous blocks.

- What paging gives that static per-sequence buffer cant:
    - sharing blocks across sequences (system prompot).
    - no memory waste from fragmentation and reservation.
    - no need to know max_len up front.

Most of the gains that come with pagedAttention can be benefited from during many concurrent requests. But concurrent requests doesn't mean many concurrent users, it could be one user running parallel sampling, beam search or even more than one model running on their local GPU and in these cases the fact that a request only takes up as much slots as it needs makes memory waste minimal. Since pagedAttention allows sharing and growth on demand, it could still be useful for single users (batch size 1).
## Transformers internals

Inside the transformers library, there is a Cache Interface class and two "manager" classes inherit it. 
The update function in Cache:
```
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
```
Two classes inherit from cache: DynamicCache & StaticCache. These are not more than containers for their respective CacheLayer classes. So for example in the dynamic caching (which is the default for transformers), the DynamicLayer class is the class with the update function and it updates per layer.

DynamicLayer:
It stores the key and value states as tensors of shape [batch_size, num_heads, seq_len, head_dim] when I update all it does is torch.cat the seq_length part with the new tokens. The way it starts off is waiting for the first key value pairs (of the prefill) to arrive and then create tensors for each one, these tensors are the ones that get updated.
Other methods in DynamicLayer:
get_seq_length: how many tokens are cached

get_mask_sizes: what the attention mask must cover

get_max_length: capacity. It returns -1, meaning unbounded, because the layer can grow indefinitely. A static cache returns its preallocated size, so generation knows when it’s full.

crop: roll back tokens. The main user is assisted/speculative decoding. A small draft model proposes, say, 5 tokens, the big model verifies them in one pass, and only some are accepted. The rejected tokens’ K and V were already added to the cache, so crop cuts them off.

Cropping is also useful for reusing a cached shared prefix, such as a system prompt, then trimming back to it between requests.

StaticLayer:

It preallocate the full buffers:

self.keys   = torch.zeros((B, num_heads, max_cache_len, k_head_dim), ...)
self.values = torch.zeros((B, num_heads, max_cache_len, v_head_dim), ...)

The full memory cost is paid up front. For Qwen2-7B with B = 1 and max_cache_len = 4096:

Per layer: it is 2 × 2 × 4096 × 128 × 2 B = 4 MB per layer, × 28 = 112 MB, reserved immediately even if you generate only 10 tokens.
Now going to models/qwen2/modeling_qwen2.py, where the reference implementation of the forward pass of qwen2 is implemented (which is usually substituted with flashattention because its faster), here's the order of events in the forward pass function:

hidden_states
  → q_proj / k_proj / v_proj      # compute Q, K, V for the NEW tokens only
  → apply RoPE to Q and K
  → past_key_values.update(...)   # ← here: K, V become past + new
  → attention(Q, K_full, V_full)
  → o_proj

So, to plug in my cache I must implement: a pagedCache class that inherits the cache class and the PagedCacheLayer should have the following functions:
    - update
    - get_mask_size
    - crop

## Design decisions
## Local use cases

Paging: Sessions grab KV memory only as they grow, so the VRAM they don't use stays free for other apps or models, instead of being reserved per session.
Sharing: different LLM sessions have the same computer-level system prompt and skills.
Swapping: offloading some kv cache from idle sessions to RAM to process other requests.
Copy-on-write: All sessions share the global-context prefix blocks read-only. A block is copied only when a session first writes into a shared, partly filled block (the last prefix block), so creating a session is free.

## Open questions
## Surprises