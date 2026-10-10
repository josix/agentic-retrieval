# `retrieval.po_loader`

Per-entry loader for gettext `.po` files, used to turn a PO repository into a
translation memory. Stdlib-only (`re`, `dataclasses`, `pathlib`); no `polib`.

Each live entry becomes one chunk-granularity `Document`:

- `docid` is `"{path}:{start}-{end}"`; the span covers the whole entry block
  (translator comments included), so it is a precise `Read` target.
- `text` is `msgctxt`, `msgid`, `msgid_plural` and `msgstr` joined by
  newlines (an untranslated entry contributes only its English text).
- `metadata` carries `kind` (`"po_entry"`), `msgid`, `msgstr`, `fuzzy`,
  `translated`, `line` (the `msgid` line) and `references`, plus `msgctxt`,
  `msgid_plural` and `msgstr_plural` when present.

The header entry (`msgid ""`) and obsolete `#~` entries are never indexed.
Fuzzy and untranslated entries are indexed and flagged so callers can filter.
`.pot` templates are not discovered.

The module registers itself as the `.po` record loader (see
[record_loaders](record_loaders.md)). `text` combines msgid and msgstr by
design, so entries match English and Chinese queries alike (cross-lingual).

::: retrieval.po_loader

## Next steps

- [Customize indexing: PO files](../../how-to/customize-indexing.md#po-files-translation-memory)
- [Reference: record_loaders API](record_loaders.md)
- [Reference: project_loader API](project_loader.md)
- [Reference: CLI](../cli.md)
