const timestampPattern = /\[(\d{1,2}:\d{2}(?::\d{2})?)\]/g;

export function linkifyTimestamps(content: string): string {
  return content.replace(timestampPattern, (match, time) => {
    const seconds = parseTimestampSeconds(time);
    if (seconds == null) return match;
    return `[${time}](timestamp:${seconds})`;
  });
}

export function parseTimestampSeconds(value: string): number | null {
  const parts = value.split(":").map((part) => part.trim());
  if (parts.length < 2 || parts.length > 3) return null;
  const secondsPart = parts.pop();
  if (!secondsPart) return null;
  const seconds = Number(secondsPart);
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  const minutesPart = parts.pop();
  if (minutesPart == null) return null;
  const minutes = Number(minutesPart);
  if (!Number.isFinite(minutes) || minutes < 0) return null;
  const hoursPart = parts.pop();
  const hours = hoursPart != null ? Number(hoursPart) : 0;
  if (!Number.isFinite(hours) || hours < 0) return null;
  return Math.floor(hours * 3600 + minutes * 60 + seconds);
}

export function parseTimestampHref(href: string): number | null {
  const raw = href.slice("timestamp:".length).trim();
  const seconds = Number(raw);
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  return Math.floor(seconds);
}
