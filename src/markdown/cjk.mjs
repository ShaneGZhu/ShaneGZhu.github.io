const CJK =
  '\\u2e80-\\u303f' + // CJK radicals, punctuation
  '\\u3040-\\u30ff' + // Hiragana, Katakana
  '\\u3400-\\u4dbf' + // CJK extension A
  '\\u4e00-\\u9fff' + // CJK unified ideographs
  '\\uf900-\\ufaff' + // compatibility ideographs
  '\\ufe30-\\ufe4f' + // CJK compatibility forms
  '\\uff00-\\uffef'; //  fullwidth forms

const BETWEEN_CJK = new RegExp(`([${CJK}])[ \\t]*\\n[ \\t]*([${CJK}])`, 'g');

/** Whitespace *within a line* between two CJK characters, as folded YAML produces. */
const SPACE_BETWEEN_CJK = new RegExp(`([${CJK}])[ \\t]+([${CJK}])`, 'g');

/**
 * Removes the stray space a source line break leaves between two CJK characters.
 *
 * Markdown folds a single newline inside a paragraph into a space, which is
 * correct for languages that separate words with spaces and wrong for Chinese and
 * Japanese: hard-wrapping a paragraph in the editor would inject a visible ~5px
 * gap mid-sentence at every wrap point.
 *
 * Only breaks with CJK on *both* sides are joined, so mixed-language text keeps
 * the space it needs (`中文\nEnglish` stays `中文 English`), as does ordinary
 * English prose.
 */
export const cjkLineBreakPlugin = {
  name: 'cjk-line-break',

  text(node) {
    const value = node.value;
    if (!value.includes('\n')) return;

    let next = value;
    // Adjacent matches share a character (`一\n二\n三`), so a single pass can miss
    // every other break. Loop until the string settles.
    for (let i = 0; i < 8; i += 1) {
      const joined = next.replace(BETWEEN_CJK, '$1$2');
      if (joined === next) break;
      next = joined;
    }

    if (next === value) return;
    return { ...node, value: next };
  },
};

/**
 * Removes whitespace that sits between two CJK characters.
 *
 * Frontmatter is not Markdown: a folded YAML scalar (`>-`) turns every wrap point
 * into a single space, so a Chinese `description` written across several lines
 * picks up gaps mid-sentence — "数值 一致性" — that show up in listings and link
 * previews. This is the frontmatter counterpart of the plugin above, applied
 * through the content schema.
 *
 * Spaces between CJK and Latin are kept, since a mixed-language description needs
 * them.
 *
 * @param {string} value
 * @returns {string}
 */
export function collapseCjkSpaces(value) {
  let next = value;
  // Overlapping matches (`一 二 三`) need more than one pass.
  for (let i = 0; i < 8; i += 1) {
    const collapsed = next.replace(SPACE_BETWEEN_CJK, '$1$2');
    if (collapsed === next) break;
    next = collapsed;
  }
  return next;
}
