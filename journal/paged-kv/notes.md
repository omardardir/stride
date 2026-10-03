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
## Design decisions
## Local use cases

Paging: Sessions grab KV memory only as they grow, so the VRAM they don't use stays free for other apps or models, instead of being reserved per session.
Sharing: different LLM sessions have the same computer-level system prompt and skills.
Swapping: offloading some kv cache from idle sessions to RAM to process other requests.
Copy-on-write: All sessions share the global-context prefix blocks read-only. A block is copied only when a session first writes into a shared, partly filled block (the last prefix block), so creating a session is free.

## Open questions
## Surprises