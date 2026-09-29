import Slugger from 'github-slugger';

/**
 * Two HTML refinements applied while Sätteri walks the HAST tree.
 *
 * 1. `headingAnchorsPlugin` assigns the heading `id` and appends a link to it,
 *    so every section is directly linkable. It runs before Astro's own
 *    heading-ids plugin, which then keeps the id we set — meaning the ids in the
 *    table of contents and in the HTML always agree. The anchor element is left
 *    empty on purpose and its `#` glyph comes from CSS, so the injected text
 *    never leaks into the heading text Astro records for the TOC.
 * 2. `tableWrapperPlugin` wraps tables in a scroll container so wide tables
 *    don't stretch the page.
 */
export function headingAnchorsPlugin() {
  // One slugger per document keeps duplicate headings unique (`intro`, `intro-1`).
  const slugger = new Slugger();

  return {
    name: 'heading-anchors',

    element: {
      filter: ['h2', 'h3', 'h4'],
      visit(node, ctx) {
        const existing = node.properties?.id;
        const id =
          typeof existing === 'string' && existing.length > 0
            ? existing
            : slugger.slug(ctx.textContent(node));

        if (id.length === 0) return;
        if (typeof existing !== 'string') ctx.setProperty(node, 'id', id);

        ctx.appendChild(node, {
          type: 'element',
          tagName: 'a',
          properties: {
            href: `#${id}`,
            class: 'heading-anchor',
            'aria-hidden': 'true',
            tabindex: '-1',
          },
          children: [],
        });
      },
    },
  };
}

export const tableWrapperPlugin = {
  name: 'table-wrapper',

  element: {
    filter: ['table'],
    visit(node, ctx) {
      const parent = ctx.parent(node);
      // Guard against double-wrapping.
      if (parent?.type === 'element' && parent.tagName === 'div') {
        const cls = parent.properties?.class ?? parent.properties?.className;
        if (typeof cls === 'string' && cls.includes('table-wrapper')) return;
      }

      ctx.wrapNode(node, {
        type: 'element',
        tagName: 'div',
        properties: { class: 'table-wrapper' },
        children: [],
      });
    },
  },
};
