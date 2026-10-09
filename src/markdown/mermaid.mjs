/**
 * Renders ```mermaid code blocks via the mermaid library in the browser.
 *
 * The Markdown pipeline does not run a headless browser at build time, so the
 * block is emitted as a `<div class="mermaid-diagram">` holding the source in
 * `data-mermaid`. `PostLayout` loads mermaid on pages that have such a block
 * and swaps each div for the generated SVG. Keeping the source in an attribute
 * (rather than as text content) avoids HTML-escaping issues with quotes and
 * keeps the diagram editable if the page is saved as text.
 */

function escapeAttribute(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/**
 * @param {string} html
 * @param {{ sourceFormat: string }} ctx
 * @returns {{ raw: string, mdxExpressions: false } | { type: 'html', value: string }}
 */
function wrap(html, ctx) {
  if (ctx.sourceFormat === 'mdx') {
    return { raw: html, mdxExpressions: false };
  }
  return { type: 'html', value: html };
}

export const mermaidPlugin = {
  name: 'mermaid',

  code(node, ctx) {
    if (node.lang !== 'mermaid') return;

    const source = String(node.value ?? '').trim();
    if (!source) return;

    const html =
      `<div class="mermaid-diagram" data-mermaid="${escapeAttribute(source)}" role="img"></div>`;
    return wrap(html, ctx);
  },
};
