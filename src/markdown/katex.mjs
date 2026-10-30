import katex from 'katex';

/**
 * Renders `$inline$` and `$$display$$` math to static KaTeX HTML at build time.
 *
 * Sätteri parses math into `math` / `inlineMath` nodes once `features.math` is
 * on; this plugin swaps each one for the rendered markup.
 *
 * The two return shapes are not interchangeable:
 *
 * - `{ type: 'html', value }` splices the HTML in verbatim, which is what inline
 *   math needs — it stays inside the surrounding paragraph.
 * - `{ raw, mdxExpressions: false }` re-parses the string as Markdown. For an
 *   inline formula that wraps the result in a *new* paragraph, producing a `<p>`
 *   nested inside a `<p>`. Browsers cannot nest paragraphs, so they close the
 *   outer one early and the rest of the sentence is pushed onto its own line.
 *
 * MDX cannot represent an `html` node as JSX, so it has to take the `raw` path —
 * but there the MDX compiler splices the markup into the JSX children of the
 * enclosing paragraph, so no stray paragraph appears. Hence the branch on
 * `ctx.sourceFormat`: `html` for `.md`, `raw` for `.mdx`.
 */
export const katexPlugin = {
  name: 'katex',

  math(node, ctx) {
    return wrap(render(node.value, true), ctx);
  },

  inlineMath(node, ctx) {
    return wrap(render(node.value, false), ctx);
  },
};

/**
 * Picks the return shape the current pipeline can represent.
 *
 * @param {string} html
 * @param {{ sourceFormat: string }} ctx
 * @returns {{ raw: string, mdxExpressions: false } | { type: 'html', value: string }}
 */
function wrap(html, ctx) {
  return ctx.sourceFormat === 'mdx'
    ? { raw: escapeForMdx(html), mdxExpressions: false }
    : { type: 'html', value: html };
}

/**
 * Makes KaTeX markup safe to hand to the MDX compiler as `raw` Markdown.
 *
 * The `raw` path re-parses the string as Markdown, and KaTeX output carries the
 * TeX source verbatim in its MathML `<annotation>`. Two things go wrong there:
 *
 * - Braces are read as JSX expressions despite `mdxExpressions: false`, so
 *   `\left\{x: y_{b}\right\}` fails the build with "Unexpected content after
 *   expression".
 * - Markdown syntax inside the text is live: `\{` is a backslash escape, so the
 *   backslash disappears from the annotation, and `^`, `$` or `*` could turn into
 *   superscript, math or emphasis.
 *
 * Encoding those characters as numeric character references sidesteps both. A
 * reference is decoded back to the same character in text and attribute values,
 * so the rendered page and the copy-pasteable TeX are unchanged.
 *
 * @param {string} html
 */
function escapeForMdx(html) {
  return html.replace(/[{}\\*_^~`$\[\]]/g, (ch) => `&#${ch.charCodeAt(0)};`);
}

function render(source, displayMode) {
  const wrapper = displayMode ? 'div' : 'span';

  let html;
  try {
    html = katex.renderToString(source, {
      displayMode,
      // Render broken formulas in red instead of failing the whole build.
      throwOnError: false,
      strict: 'ignore',
      output: 'htmlAndMathml',
      trust: false,
    });
  } catch (error) {
    // renderToString can still throw on malformed input despite throwOnError.
    const message = error instanceof Error ? error.message : String(error);
    return `<${wrapper} class="math math-error" title="${escapeAttribute(message)}">${escapeText(source)}</${wrapper}>`;
  }

  const className = displayMode ? 'math math-display' : 'math math-inline';
  return `<${wrapper} class="${className}">${html}</${wrapper}>`;
}

function escapeText(value) {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function escapeAttribute(value) {
  return escapeText(value).replace(/"/g, '&quot;');
}
