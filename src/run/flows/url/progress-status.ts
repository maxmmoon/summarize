import type { OscProgressController } from "../../../tty/osc-progress.js";

export function createUrlProgressStatus({
  enabled,
  spinner,
  oscProgress,
  now = () => Date.now(),
}: {
  enabled: boolean;
  spinner: { setText: (text: string) => void };
  oscProgress: OscProgressController;
  now?: () => number;
}) {
  let summaryText: string | null = null;
  let slidesActive = false;
  let slidesText: string | null = null;
  let lastSlideRenderAt = 0;

  const render = (text: string | null) => {
    if (!enabled || !text) return;
    spinner.setText(text);
  };

  return {
    setSummary(text: string, oscLabel?: string | null) {
      summaryText = text;
      if (slidesActive) return;
      render(text);
      if (oscLabel) {
        oscProgress.setIndeterminate(oscLabel);
      }
    },
    setSlides(text: string, percent?: number | null) {
      slidesActive = true;
      const previousSlidesText = slidesText;
      slidesText = text;
      const nowMs = now();
      if (previousSlidesText == null || nowMs - lastSlideRenderAt >= 100) {
        lastSlideRenderAt = nowMs;
        render(text);
      }
      if (typeof percent === "number" && Number.isFinite(percent)) {
        oscProgress.setPercent("Slides", Math.max(0, Math.min(100, percent)));
      } else {
        oscProgress.setIndeterminate("Slides");
      }
    },
    clearSlides() {
      slidesActive = false;
      slidesText = null;
      if (summaryText) {
        render(summaryText);
        oscProgress.setIndeterminate("Summarizing");
      }
    },
    isSlidesActive() {
      return slidesActive;
    },
    getSummaryText() {
      return summaryText;
    },
    getSlidesText() {
      return slidesText;
    },
  };
}
