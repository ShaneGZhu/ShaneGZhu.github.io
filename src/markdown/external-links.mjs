/**
 * Adds `target="_blank"` and `rel="noopener noreferrer"` to off-site links.
 *
 * Runs on the hast side so it sees the final `href`, after Markdown link
 * references have been resolved. Anchors, root-relative and relative links are
 * left alone.
 */
export const externalLinksPlugin = {
  name: 'external-links',

  element: {
    filter: ['a'],
    visit(node, ctx) {
      const href = node.properties?.href;
      if (typeof href !== 'string') return;
      if (!/^https?:\/\//i.test(href)) return;

      ctx.setProperty(node, 'target', '_blank');
      ctx.setProperty(node, 'rel', 'noopener noreferrer');
      ctx.setProperty(node, 'class', 'external-link');
    },
  },
};
