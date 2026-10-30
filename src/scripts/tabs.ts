/**
 * Progressive tabs for figures that offer several views of the same data.
 *
 * The server renders every view as a plain, visible block with its own title,
 * so the page is complete without JavaScript. This script then turns the block
 * into a tab set: it reveals the tab list, gives the views their roles, and
 * shows one at a time. Keyboard handling follows the WAI-ARIA tabs pattern with
 * automatic activation (arrow keys, Home, End).
 *
 * Markup contract:
 *
 *   <div data-tabs>
 *     <div role="tablist" hidden>
 *       <button role="tab" aria-controls="panel-a">…</button>
 *     </div>
 *     <section id="panel-a" data-tab-panel>…</section>
 *   </div>
 */
export function initTabs(root: HTMLElement): void {
  const list = root.querySelector<HTMLElement>('[role="tablist"]');
  const tabs = Array.from(root.querySelectorAll<HTMLButtonElement>('[role="tab"]'));
  if (!list || tabs.length === 0) return;

  const panelFor = (tab: HTMLButtonElement) =>
    document.getElementById(tab.getAttribute('aria-controls') ?? '');

  for (const tab of tabs) {
    const panel = panelFor(tab);
    if (!panel) return;
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', tab.id);
    panel.tabIndex = 0;
  }

  const select = (next: HTMLButtonElement, focus: boolean) => {
    for (const tab of tabs) {
      const active = tab === next;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      panelFor(tab)?.toggleAttribute('hidden', !active);
    }
    if (focus) next.focus();
  };

  list.addEventListener('click', (event) => {
    const tab = (event.target as Element).closest<HTMLButtonElement>('[role="tab"]');
    if (tab) select(tab, false);
  });

  list.addEventListener('keydown', (event) => {
    const current = tabs.indexOf(document.activeElement as HTMLButtonElement);
    if (current === -1) return;
    const last = tabs.length - 1;
    const target =
      event.key === 'ArrowRight' ? (current === last ? 0 : current + 1)
      : event.key === 'ArrowLeft' ? (current === 0 ? last : current - 1)
      : event.key === 'Home' ? 0
      : event.key === 'End' ? last
      : -1;
    if (target === -1) return;
    event.preventDefault();
    select(tabs[target]!, true);
  });

  list.hidden = false;
  select(tabs[0]!, false);
}

export function initAllTabs(): void {
  for (const root of document.querySelectorAll<HTMLElement>('[data-tabs]')) initTabs(root);
}
