/**
 * Zhihu-style callout blocks written as container directives:
 *
 * ```md
 * :::tip{title="Heads up"}
 * Body text, with **full Markdown** inside.
 * :::
 * ```
 *
 * Supported names: note, tip, info, warning, danger, quote.
 * The directive is rewritten to `<aside class="callout callout-tip">` with an
 * optional label element, which `src/styles/prose.css` styles.
 */

const KNOWN = new Map([
  ['note', 'Note'],
  ['tip', 'Tip'],
  ['info', 'Info'],
  ['warning', 'Warning'],
  ['danger', 'Caution'],
  ['quote', 'Quote'],
]);

export const calloutPlugin = {
  name: 'callout',

  containerDirective(node, ctx) {
    const kind = KNOWN.has(node.name) ? node.name : 'note';
    const title = node.attributes?.title ?? KNOWN.get(kind);

    ctx.setProperty(node, 'data', {
      hName: 'aside',
      hProperties: {
        class: `callout callout-${kind}`,
        'data-callout': kind,
        role: kind === 'warning' || kind === 'danger' ? 'note' : undefined,
      },
    });

    ctx.prependChild(node, {
      type: 'paragraph',
      data: { hName: 'p', hProperties: { class: 'callout-title' } },
      children: [{ type: 'text', value: title }],
    });
  },
};
