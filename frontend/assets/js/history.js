const KEY = "voicerag.history.v1";
const MAX = 20;

export function loadHistory() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const items = JSON.parse(raw);
    return Array.isArray(items) ? items : [];
  } catch {
    return [];
  }
}

export function addHistory(entry) {
  const items = loadHistory();
  items.unshift(entry);
  while (items.length > MAX) items.pop();
  try {
    localStorage.setItem(KEY, JSON.stringify(items));
  } catch {}
  return items;
}

export function clearHistory() {
  try {
    localStorage.removeItem(KEY);
  } catch {}
}

export function timeAgo(ts) {
  const s = Math.max(1, Math.floor((Date.now() - ts) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}
