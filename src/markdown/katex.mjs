import katex from 'katex';

/**
 * Renders `$inline$` and `$$display$$` math to static KaTeX HTML at build time.
 *
 * Sätteri parses math into `math` / `inlineMath` nodes once `features.math` is
 * on; this plugin swaps each one for the rendered markup. `mdxExpressions:
 * false` is required because KaTeX output contains `{` characters that MDX
 * would otherwise try to evaluate as JavaScript expressions.
 */
export const katexPlugin = {
  name: 'katex',

  math(node) {
    return { raw: render(node.value, true), mdxExpressions: false };
  },

  inlineMath(node) {
    return { raw: render(node.value, false), mdxExpressions: false };
  },
};

function render(source, displayMode) {
  const wrapper = displayMode ? 'div' : 'span';
  const className = displayMode ? 'math math-display' : 'math math-inline';

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

  return `<${wrapper} class="${className}">${html}</${wrapper}>`;
}

function escapeText(value) {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function escapeAttribute(value) {
  return escapeText(value).replace(/"/g, '&quot;');
}
