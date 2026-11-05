# Compatibility routes: `/v1/responses` and `/v1/embeddings`

Both routes reuse the existing private-alias and ledger machinery. Neither opens
a general pass-through: a request can only reach a channel that startup audit
already verified.

## `POST /v1/responses`

OpenAI's Responses API is translated to a chat completion before routing, so the
same quota reservation, fallback ordering and telemetry apply.

Supported:

- `input` as a string, or a list of `message` items with `role` in
  `user`, `assistant`, `system`, `developer`.
- `instructions` (prepended as a `system` message).
- `tools` in the flat Responses shape (`{"type": "function", "name": ...}`),
  translated to the nested chat shape. Tool calls are returned as `function_call`
  output items.
- `tool_choice` as `auto`, `none` or `required`.
- `text.format` with `text`, `json_object` or `json_schema`.
- `max_output_tokens`, `temperature`, `top_p`, `stream`.
- `model` must be one of the 22 chat aliases (profiles, compatibility names, or
  legacy gateway names).

Returned object: `id`, `object: "response"`, `created_at`, `status`, `model`,
`output`, `output_text` and `usage` in Responses shape. `output_text` is the
SDK convenience field; it is empty when the upstream returned only tool calls.

Streaming is supported for text output only. It emits
`response.created`, `response.output_item.added`, `response.content_part.added`,
`response.output_text.delta`, `response.output_text.done`,
`response.content_part.done`, `response.output_item.done` and
`response.completed`, each with a `sequence_number`.

Rejected with HTTP 400, never silently degraded:

- `previous_response_id` (the shim is stateless),
- `function_call_output` and image/audio input items,
- non-function tools and structured `tool_choice` objects,
- `stream: true` combined with `tools` (streaming tool-call deltas are not
  translated).

## `POST /v1/embeddings`

Requires the operator to name the accepted aliases:

```sh
export SIDECAR_EMBEDDING_ALIASES=sc.embed
```

Each name must already exist as an audited route alias, and must not collide
with a chat alias; otherwise startup fails with a named error. The alias is sent
to New-API `/embeddings` with the relay token, so New-API still owns the
alias-to-model mapping.

Supported: `input` as a string or a non-empty list of strings (up to 2048 items,
32 KiB each), and `model` set to a configured embedding alias. The response is
validated (bounded size, finite numeric vectors, integer indexes) before it is
returned; the requested alias is echoed back.

Not supported: token arrays, file inputs, `encoding_format: "base64"`.

Accounting: each batch reserves an estimated prompt charge (3 characters per
token, at least 1), then settles on the upstream `usage.prompt_tokens` when
reported. Output tokens are zero. A failed batch is accounted and cooled down
exactly like a failed chat attempt.

## Limits

- `/v1/embeddings` is not advertised in `/v1/models`; chat aliases and embedding
  aliases are deliberately separate namespaces.
- The Responses shim keeps no response objects, so clients cannot continue a
  response by id.
- Neither route has been exercised against a live provider from VPS-40. The
  remaining gate is the same as for chat: install and validate the sidecar
  against the audited New-API channels.
