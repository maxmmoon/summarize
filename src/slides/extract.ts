import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { ExtractedLinkContent, MediaCache } from "../content/index.js";
import { extractYouTubeVideoId, isDirectMediaUrl, isYouTubeUrl } from "../content/index.js";
import { resolveExecutableInPath } from "../run/env.js";
import { prepareSlidesInput } from "./ingest.js";
import { runOcrOnSlides } from "./ocr.js";
import { runProcess, runProcessCapture, runWithConcurrency } from "./process.js";
import {
  adjustTimestampWithinSegment,
  applyMaxSlidesFilter,
  applyMinDurationFilter,
  buildIntervalTimestamps,
  buildSceneSegments,
  buildSegments,
  calibrateSceneThreshold,
  clamp,
  detectSceneTimestamps,
  filterTimestampsByMinDuration,
  findSceneSegment,
  mergeTimestamps,
  parseShowinfoTimestamp,
  probeVideoInfo,
  resolveExtractedTimestamp,
  roundThreshold,
  selectTimestampTargets,
} from "./scene-detection.js";
import type { SlideSettings } from "./settings.js";
import { buildDirectSourceId, buildYoutubeSourceId } from "./source-id.js";
import {
  buildSlidesDirId,
  readSlidesCacheIfValid,
  resolveSlidesDir,
  serializeSlideImagePath,
} from "./store.js";
import type {
  SlideAutoTune,
  SlideExtractionResult,
  SlideImage,
  SlideSource,
  SlideSourceKind,
} from "./types.js";

const slidesLocks = new Map<string, Promise<void>>();
const YT_DLP_TIMEOUT_MS = 300_000;
const DEFAULT_SLIDES_WORKERS = 8;
const DEFAULT_SLIDES_SAMPLE_COUNT = 8;
// Prefer broadly-decodable H.264/MP4 for ffmpeg stability.
// (Some "bestvideo" picks AV1 which can fail on certain ffmpeg builds / hwaccel setups.)
const DEFAULT_YT_DLP_FORMAT_EXTRACT =
  "bestvideo[height<=720][vcodec^=avc1][ext=mp4]/best[height<=720][vcodec^=avc1][ext=mp4]/bestvideo[height<=720][ext=mp4]/best[height<=720]";

type SlidesLogger = ((message: string) => void) | null;

export { parseShowinfoTimestamp, resolveExtractedTimestamp };

function createSlidesLogger(logger: SlidesLogger) {
  const logSlides = (message: string) => {
    if (!logger) return;
    logger(message);
  };
  const logSlidesTiming = (label: string, startedAt: number) => {
    const elapsedMs = Date.now() - startedAt;
    logSlides(`${label} elapsedMs=${elapsedMs}`);
    return elapsedMs;
  };
  return { logSlides, logSlidesTiming };
}

function resolveSlidesWorkers(env: Record<string, string | undefined>): number {
  const raw = env.SUMMARIZE_SLIDES_WORKERS ?? env.SLIDES_WORKERS;
  if (!raw) return DEFAULT_SLIDES_WORKERS;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_SLIDES_WORKERS;
  return Math.max(1, Math.min(16, Math.round(parsed)));
}

function resolveSlidesSampleCount(env: Record<string, string | undefined>): number {
  const raw = env.SUMMARIZE_SLIDES_SAMPLES ?? env.SLIDES_SAMPLES;
  if (!raw) return DEFAULT_SLIDES_SAMPLE_COUNT;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_SLIDES_SAMPLE_COUNT;
  return Math.max(3, Math.min(12, Math.round(parsed)));
}

function resolveSlidesYtDlpExtractFormat(env: Record<string, string | undefined>): string {
  return (
    env.SUMMARIZE_SLIDES_YTDLP_FORMAT_EXTRACT ??
    env.SLIDES_YTDLP_FORMAT_EXTRACT ??
    DEFAULT_YT_DLP_FORMAT_EXTRACT
  ).trim();
}

function resolveSlidesStreamFallback(env: Record<string, string | undefined>): boolean {
  const raw = env.SLIDES_EXTRACT_STREAM?.trim().toLowerCase();
  return raw === "1" || raw === "true" || raw === "yes";
}

function buildYtDlpCookiesArgs(cookiesFromBrowser?: string | null): string[] {
  const value = typeof cookiesFromBrowser === "string" ? cookiesFromBrowser.trim() : "";
  return value.length > 0 ? ["--cookies-from-browser", value] : [];
}

function buildSlidesMediaCacheKey(url: string): string {
  return `${url}#summarize-slides`;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = units[0] ?? "B";
  for (let i = 1; i < units.length && value >= 1024; i += 1) {
    value /= 1024;
    unit = units[i] ?? unit;
  }
  const rounded = value >= 100 ? Math.round(value) : Math.round(value * 10) / 10;
  return `${rounded}${unit}`;
}

function resolveToolPath(
  binary: string,
  env: Record<string, string | undefined>,
  explicitEnvKey?: string,
): string | null {
  const explicit =
    explicitEnvKey && typeof env[explicitEnvKey] === "string" ? env[explicitEnvKey]?.trim() : "";
  if (explicit) return resolveExecutableInPath(explicit, env);
  return resolveExecutableInPath(binary, env);
}

type ExtractSlidesArgs = {
  source: SlideSource;
  settings: SlideSettings;
  noCache?: boolean;
  mediaCache?: MediaCache | null;
  env: Record<string, string | undefined>;
  timeoutMs: number;
  ytDlpPath: string | null;
  ytDlpCookiesFromBrowser?: string | null;
  ffmpegPath: string | null;
  tesseractPath: string | null;
  hooks?: {
    onSlideChunk?: (chunk: {
      slide: SlideImage;
      meta: {
        slidesDir: string;
        sourceUrl: string;
        sourceId: string;
        sourceKind: SlideSourceKind;
        ocrAvailable: boolean;
      };
    }) => void;
    onSlidesTimeline?: ((slides: SlideExtractionResult) => void) | null;
    onSlidesProgress?: ((text: string) => void) | null;
    onSlidesLog?: ((message: string) => void) | null;
  } | null;
};

export function resolveSlideSource({
  url,
  extracted,
}: {
  url: string;
  extracted: ExtractedLinkContent;
}): SlideSource | null {
  const directUrl = extracted.video?.url ?? extracted.url;
  const youtubeCandidate =
    extractYouTubeVideoId(extracted.video?.url ?? "") ??
    extractYouTubeVideoId(extracted.url) ??
    extractYouTubeVideoId(url);
  if (youtubeCandidate) {
    return {
      url: `https://www.youtube.com/watch?v=${youtubeCandidate}`,
      kind: "youtube",
      sourceId: buildYoutubeSourceId(youtubeCandidate),
    };
  }

  if (extracted.video?.kind === "direct" || isDirectMediaUrl(directUrl) || isDirectMediaUrl(url)) {
    const normalized = directUrl || url;
    return {
      url: normalized,
      kind: "direct",
      sourceId: buildDirectSourceId(normalized),
    };
  }

  if (isYouTubeUrl(url)) {
    const fallbackId = extractYouTubeVideoId(url);
    if (fallbackId) {
      return {
        url: `https://www.youtube.com/watch?v=${fallbackId}`,
        kind: "youtube",
        sourceId: buildYoutubeSourceId(fallbackId),
      };
    }
  }

  return null;
}

export function resolveSlideSourceFromUrl(url: string): SlideSource | null {
  const youtubeCandidate = extractYouTubeVideoId(url);
  if (youtubeCandidate) {
    return {
      url: `https://www.youtube.com/watch?v=${youtubeCandidate}`,
      kind: "youtube",
      sourceId: buildYoutubeSourceId(youtubeCandidate),
    };
  }

  if (isDirectMediaUrl(url)) {
    return {
      url,
      kind: "direct",
      sourceId: buildDirectSourceId(url),
    };
  }

  if (isYouTubeUrl(url)) {
    const fallbackId = extractYouTubeVideoId(url);
    if (fallbackId) {
      return {
        url: `https://www.youtube.com/watch?v=${fallbackId}`,
        kind: "youtube",
        sourceId: buildYoutubeSourceId(fallbackId),
      };
    }
  }

  return null;
}

export async function extractSlidesForSource({
  source,
  settings,
  noCache = false,
  mediaCache = null,
  env,
  timeoutMs,
  ytDlpPath,
  ytDlpCookiesFromBrowser,
  ffmpegPath,
  tesseractPath,
  hooks,
}: ExtractSlidesArgs): Promise<SlideExtractionResult> {
  const slidesDir = resolveSlidesDir(settings.outputDir, source.sourceId);
  return withSlidesLock(
    slidesDir,
    async () => {
      const { logSlides, logSlidesTiming } = createSlidesLogger(hooks?.onSlidesLog ?? null);
      if (!noCache) {
        const cached = await readSlidesCacheIfValid({ source, settings });
        if (cached) {
          hooks?.onSlidesTimeline?.(cached);
          return cached;
        }
      }

      const reportSlidesProgress = (() => {
        const onSlidesProgress = hooks?.onSlidesProgress;
        if (!onSlidesProgress) return null;
        let lastText = "";
        let lastPercent = 0;
        return (label: string, percent: number, detail?: string) => {
          const clamped = clamp(Math.round(percent), 0, 100);
          const nextPercent = Math.max(lastPercent, clamped);
          const suffix = detail ? ` ${detail}` : "";
          const text = `Slides: ${label}${suffix} ${nextPercent}%`;
          if (text === lastText) return;
          lastText = text;
          lastPercent = nextPercent;
          onSlidesProgress(text);
        };
      })();

      const warnings: string[] = [];
      const workers = resolveSlidesWorkers(env);
      const totalStartedAt = Date.now();
      logSlides(
        `pipeline=ingest(sequential)->scene-detect(parallel:${workers})->extract-frames(parallel:${workers})->ocr(parallel:${workers})`,
      );

      const ffmpegBinary = ffmpegPath ?? resolveToolPath("ffmpeg", env, "FFMPEG_PATH");
      if (!ffmpegBinary) {
        throw new Error("Missing ffmpeg (install ffmpeg or add it to PATH).");
      }
      const ffprobeBinary = resolveToolPath("ffprobe", env, "FFPROBE_PATH");

      if (settings.ocr && !tesseractPath) {
        const resolved = resolveToolPath("tesseract", env, "TESSERACT_PATH");
        if (!resolved) {
          throw new Error("Missing tesseract OCR (install tesseract or skip --slides-ocr).");
        }
        tesseractPath = resolved;
      }
      const ocrEnabled = Boolean(settings.ocr && tesseractPath);
      const ocrAvailable = Boolean(
        tesseractPath ?? resolveToolPath("tesseract", env, "TESSERACT_PATH"),
      );

      const P_PREPARE = 2;
      const P_FETCH_VIDEO = 6;
      const P_DOWNLOAD_VIDEO = 35;
      const P_DETECT_SCENES = 60;
      const P_EXTRACT_FRAMES = 90;
      const P_OCR = 99;
      const P_FINAL = 100;

      {
        const prepareStartedAt = Date.now();
        await prepareSlidesDir(slidesDir);
        logSlidesTiming("prepare output dir", prepareStartedAt);
      }
      reportSlidesProgress?.("preparing source", P_PREPARE);

      const {
        inputPath,
        inputCleanup,
        warnings: ingestWarnings,
      } = await prepareSlidesInput({
        source,
        mediaCache,
        timeoutMs,
        ytDlpPath,
        ytDlpCookiesFromBrowser,
        resolveSlidesYtDlpExtractFormat: () => resolveSlidesYtDlpExtractFormat(env),
        resolveSlidesStreamFallback: () => resolveSlidesStreamFallback(env),
        buildSlidesMediaCacheKey,
        formatBytes,
        reportSlidesProgress,
        logSlidesTiming,
        downloadYoutubeVideo,
        downloadRemoteVideo,
        resolveYoutubeStreamUrl,
      });
      warnings.push(...ingestWarnings);

      try {
        const ffmpegStartedAt = Date.now();
        reportSlidesProgress?.("detecting scenes", P_FETCH_VIDEO + 2);
        const detection = await detectSlideTimestamps({
          ffmpegPath: ffmpegBinary,
          ffprobePath: ffprobeBinary,
          inputPath,
          sceneThreshold: settings.sceneThreshold,
          autoTuneThreshold: settings.autoTuneThreshold,
          env,
          timeoutMs,
          warnings,
          workers,
          sampleCount: resolveSlidesSampleCount(env),
          onSegmentProgress: (completed, total) => {
            const ratio = total > 0 ? completed / total : 0;
            const mapped = P_FETCH_VIDEO + 2 + ratio * (P_DETECT_SCENES - (P_FETCH_VIDEO + 2));
            reportSlidesProgress?.(
              "detecting scenes",
              mapped,
              total > 0 ? `(${completed}/${total})` : undefined,
            );
          },
          logSlides,
          logSlidesTiming,
        });
        reportSlidesProgress?.("detecting scenes", P_DETECT_SCENES);
        logSlidesTiming("ffmpeg scene-detect", ffmpegStartedAt);

        const interval = buildIntervalTimestamps({
          durationSeconds: detection.durationSeconds,
          minDurationSeconds: settings.minDurationSeconds,
          maxSlides: settings.maxSlides,
        });
        const combined = mergeTimestamps(
          detection.timestamps,
          interval?.timestamps ?? [],
          settings.minDurationSeconds,
        );
        if (combined.length === 0) {
          throw new Error("No slides detected; try adjusting slide extraction settings.");
        }
        const sceneSegments = buildSceneSegments(detection.timestamps, detection.durationSeconds);
        const selected = interval?.timestamps.length
          ? selectTimestampTargets({
              targets: interval.timestamps,
              sceneTimestamps: detection.timestamps,
              minDurationSeconds: settings.minDurationSeconds,
              intervalSeconds: interval.intervalSeconds,
            })
          : combined;
        const spaced = filterTimestampsByMinDuration(selected, settings.minDurationSeconds);
        const trimmed = applyMaxSlidesFilter(
          spaced.map((timestamp, index) => {
            const segment = findSceneSegment(sceneSegments, timestamp);
            const adjusted = adjustTimestampWithinSegment(timestamp, segment);
            return { index: index + 1, timestamp: adjusted, imagePath: "", segment };
          }),
          settings.maxSlides,
          warnings,
          (imagePath) => {
            void fs.rm(imagePath, { force: true }).catch(() => {});
          },
        );

        const timelineSlides: SlideExtractionResult = {
          sourceUrl: source.url,
          sourceKind: source.kind,
          sourceId: source.sourceId,
          slidesDir,
          slidesDirId: buildSlidesDirId(slidesDir),
          sceneThreshold: settings.sceneThreshold,
          autoTuneThreshold: settings.autoTuneThreshold,
          autoTune: detection.autoTune,
          maxSlides: settings.maxSlides,
          minSlideDuration: settings.minDurationSeconds,
          ocrRequested: settings.ocr,
          ocrAvailable,
          slides: trimmed.map(({ segment: _segment, ...slide }) => slide),
          warnings,
        };
        hooks?.onSlidesTimeline?.(timelineSlides);

        // Emit placeholders immediately so the UI can render the slide list while frames are still extracting.
        if (hooks?.onSlideChunk) {
          const meta = {
            slidesDir,
            sourceUrl: source.url,
            sourceId: source.sourceId,
            sourceKind: source.kind,
            ocrAvailable,
          };
          for (const slide of trimmed) {
            const { segment: _segment, ...payload } = slide;
            hooks.onSlideChunk({ slide: { ...payload, imagePath: "" }, meta });
          }
        }

        const formatProgressCount = (completed: number, total: number) =>
          total > 0 ? `(${completed}/${total})` : "";
        const reportFrameProgress = (completed: number, total: number) => {
          const ratio = total > 0 ? completed / total : 0;
          reportSlidesProgress?.(
            "extracting frames",
            P_DETECT_SCENES + ratio * (P_EXTRACT_FRAMES - P_DETECT_SCENES),
            formatProgressCount(completed, total),
          );
        };
        reportFrameProgress(0, trimmed.length);

        const onSlideChunk = hooks?.onSlideChunk;
        const extractFrames = async () =>
          extractFramesAtTimestamps({
            ffmpegPath: ffmpegBinary,
            inputPath,
            outputDir: slidesDir,
            timestamps: trimmed.map((slide) => slide.timestamp),
            segments: trimmed.map((slide) => slide.segment ?? null),
            durationSeconds: detection.durationSeconds,
            timeoutMs,
            workers,
            onProgress: reportFrameProgress,
            onStatus: hooks?.onSlidesProgress ?? null,
            onSlide: onSlideChunk
              ? (slide) =>
                  onSlideChunk({
                    slide,
                    meta: {
                      slidesDir,
                      sourceUrl: source.url,
                      sourceId: source.sourceId,
                      sourceKind: source.kind,
                      ocrAvailable,
                    },
                  })
              : null,
            logSlides,
            logSlidesTiming,
          });
        const extractFramesStartedAt = Date.now();
        const extractedSlides: SlideImage[] = await extractFrames();
        const extractElapsedMs = logSlidesTiming?.(
          `extract frames (count=${trimmed.length}, parallel=${workers})`,
          extractFramesStartedAt,
        );
        if (trimmed.length > 0 && typeof extractElapsedMs === "number") {
          logSlides?.(
            `extract frames avgMsPerFrame=${Math.round(extractElapsedMs / trimmed.length)}`,
          );
        }

        const rawSlides = applyMinDurationFilter(
          extractedSlides,
          settings.minDurationSeconds,
          warnings,
          (imagePath) => {
            void fs.rm(imagePath, { force: true }).catch(() => {});
          },
        );

        const renameStartedAt = Date.now();
        const renamedSlides = await renameSlidesWithTimestamps(rawSlides, slidesDir);
        logSlidesTiming?.("rename slides", renameStartedAt);
        if (renamedSlides.length === 0) {
          throw new Error("No slides extracted; try lowering --slides-scene-threshold.");
        }

        let slidesWithOcr = renamedSlides;
        if (ocrEnabled && tesseractPath) {
          const ocrStartedAt = Date.now();
          logSlides?.(`ocr start count=${renamedSlides.length} mode=parallel workers=${workers}`);
          const ocrStartPercent = P_OCR - 3;
          const reportOcrProgress = (completed: number, total: number) => {
            const ratio = total > 0 ? completed / total : 0;
            reportSlidesProgress?.(
              "running OCR",
              ocrStartPercent + ratio * (P_OCR - ocrStartPercent),
              formatProgressCount(completed, total),
            );
          };
          reportOcrProgress(0, renamedSlides.length);
          slidesWithOcr = await runOcrOnSlides(
            renamedSlides,
            tesseractPath,
            workers,
            reportOcrProgress,
          );
          const elapsedMs = logSlidesTiming?.("ocr done", ocrStartedAt);
          if (renamedSlides.length > 0 && typeof elapsedMs === "number") {
            logSlides?.(`ocr avgMsPerSlide=${Math.round(elapsedMs / renamedSlides.length)}`);
          }
        }

        reportSlidesProgress?.("finalizing", P_FINAL - 1);

        if (hooks?.onSlideChunk) {
          for (const slide of slidesWithOcr) {
            hooks.onSlideChunk({
              slide,
              meta: {
                slidesDir,
                sourceUrl: source.url,
                sourceId: source.sourceId,
                sourceKind: source.kind,
                ocrAvailable,
              },
            });
          }
        }

        const result: SlideExtractionResult = {
          sourceUrl: source.url,
          sourceKind: source.kind,
          sourceId: source.sourceId,
          slidesDir,
          slidesDirId: buildSlidesDirId(slidesDir),
          sceneThreshold: settings.sceneThreshold,
          autoTuneThreshold: settings.autoTuneThreshold,
          autoTune: detection.autoTune,
          maxSlides: settings.maxSlides,
          minSlideDuration: settings.minDurationSeconds,
          ocrRequested: settings.ocr,
          ocrAvailable,
          slides: slidesWithOcr,
          warnings,
        };

        await writeSlidesJson(result, slidesDir);
        reportSlidesProgress?.("finalizing", P_FINAL);
        logSlidesTiming("slides total", totalStartedAt);
        return result;
      } finally {
        if (inputCleanup) {
          await inputCleanup();
        }
      }
    },
    () => {
      hooks?.onSlidesProgress?.("Slides: queued");
    },
  );
}

async function prepareSlidesDir(slidesDir: string): Promise<void> {
  await fs.mkdir(slidesDir, { recursive: true });
  const entries = await fs.readdir(slidesDir);
  await Promise.all(
    entries.map(async (entry) => {
      if (entry.startsWith("slide_") && entry.endsWith(".png")) {
        await fs.rm(path.join(slidesDir, entry), { force: true });
      }
      if (entry === "slides.json") {
        await fs.rm(path.join(slidesDir, entry), { force: true });
      }
    }),
  );
}

async function downloadYoutubeVideo({
  ytDlpPath,
  url,
  timeoutMs,
  format,
  cookiesFromBrowser,
  onProgress,
}: {
  ytDlpPath: string;
  url: string;
  timeoutMs: number;
  format: string;
  cookiesFromBrowser?: string | null;
  onProgress?: ((percent: number, detail?: string) => void) | null;
}): Promise<{ filePath: string; cleanup: () => Promise<void> }> {
  const dir = await fs.mkdtemp(path.join(tmpdir(), `summarize-slides-${randomUUID()}-`));
  const outputTemplate = path.join(dir, "video.%(ext)s");
  const progressTemplate =
    "progress:%(progress.downloaded_bytes)s|%(progress.total_bytes)s|%(progress.total_bytes_estimate)s";
  const args = [
    "-f",
    format,
    "--no-playlist",
    "--no-warnings",
    "--concurrent-fragments",
    "4",
    ...buildYtDlpCookiesArgs(cookiesFromBrowser),
    ...(onProgress ? ["--progress", "--newline", "--progress-template", progressTemplate] : []),
    "-o",
    outputTemplate,
    url,
  ];
  await runProcess({
    command: ytDlpPath,
    args,
    timeoutMs: Math.max(timeoutMs, YT_DLP_TIMEOUT_MS),
    errorLabel: "yt-dlp",
    onStderrLine: (line, handle) => {
      if (!onProgress) return;
      const trimmed = line.trim();
      if (trimmed.startsWith("progress:")) {
        const payload = trimmed.slice("progress:".length);
        const [downloadedRaw, totalRaw, estimateRaw] = payload.split("|");
        const downloaded = Number.parseFloat(downloadedRaw);
        if (!Number.isFinite(downloaded) || downloaded < 0) return;
        const totalCandidate = Number.parseFloat(totalRaw);
        const estimateCandidate = Number.parseFloat(estimateRaw);
        const totalBytes =
          Number.isFinite(totalCandidate) && totalCandidate > 0
            ? totalCandidate
            : Number.isFinite(estimateCandidate) && estimateCandidate > 0
              ? estimateCandidate
              : null;
        if (!totalBytes || totalBytes <= 0) return;
        const percent = Math.max(0, Math.min(100, Math.round((downloaded / totalBytes) * 100)));
        const detail = `(${formatBytes(downloaded)}/${formatBytes(totalBytes)})`;
        onProgress(percent, detail);
        handle?.setProgress(percent, detail);
        return;
      }
      if (!trimmed.startsWith("[download]")) return;
      const percentMatch = trimmed.match(/\b(\d{1,3}(?:\.\d+)?)%\b/);
      if (!percentMatch) return;
      const percent = Number(percentMatch[1]);
      if (!Number.isFinite(percent) || percent < 0 || percent > 100) return;
      const etaMatch = trimmed.match(/\bETA\s+(\S+)\b/);
      const speedMatch = trimmed.match(/\bat\s+(\S+)\b/);
      const detailParts = [
        speedMatch?.[1] ? `at ${speedMatch[1]}` : null,
        etaMatch?.[1] ? `ETA ${etaMatch[1]}` : null,
      ].filter(Boolean);
      const detail = detailParts.length ? detailParts.join(" ") : undefined;
      onProgress(percent, detail);
      handle?.setProgress(percent, detail ?? null);
    },
    onStdoutLine: onProgress
      ? (line, handle) => {
          if (!line.trim().startsWith("progress:")) return;
          const payload = line.trim().slice("progress:".length);
          const [downloadedRaw, totalRaw, estimateRaw] = payload.split("|");
          const downloaded = Number.parseFloat(downloadedRaw);
          if (!Number.isFinite(downloaded) || downloaded < 0) return;
          const totalCandidate = Number.parseFloat(totalRaw);
          const estimateCandidate = Number.parseFloat(estimateRaw);
          const totalBytes =
            Number.isFinite(totalCandidate) && totalCandidate > 0
              ? totalCandidate
              : Number.isFinite(estimateCandidate) && estimateCandidate > 0
                ? estimateCandidate
                : null;
          if (!totalBytes || totalBytes <= 0) return;
          const percent = Math.max(0, Math.min(100, Math.round((downloaded / totalBytes) * 100)));
          const detail = `(${formatBytes(downloaded)}/${formatBytes(totalBytes)})`;
          onProgress(percent, detail);
          handle?.setProgress(percent, detail);
        }
      : undefined,
  });

  const files = await fs.readdir(dir);
  const candidates = [];
  for (const entry of files) {
    if (entry.endsWith(".part") || entry.endsWith(".ytdl")) continue;
    const filePath = path.join(dir, entry);
    const stat = await fs.stat(filePath).catch(() => null);
    if (stat?.isFile()) {
      candidates.push({ filePath, size: stat.size });
    }
  }
  if (candidates.length === 0) {
    await fs.rm(dir, { recursive: true, force: true });
    throw new Error("yt-dlp completed but no video file was downloaded.");
  }
  candidates.sort((a, b) => b.size - a.size);
  const filePath = candidates[0].filePath;
  return {
    filePath,
    cleanup: async () => {
      await fs.rm(dir, { recursive: true, force: true });
    },
  };
}

async function downloadRemoteVideo({
  url,
  timeoutMs,
  onProgress,
}: {
  url: string;
  timeoutMs: number;
  onProgress?: ((percent: number, detail?: string) => void) | null;
}): Promise<{ filePath: string; cleanup: () => Promise<void> }> {
  const dir = await fs.mkdtemp(path.join(tmpdir(), `summarize-slides-${randomUUID()}-`));
  let suffix = ".bin";
  try {
    const parsed = new URL(url);
    const ext = path.extname(parsed.pathname);
    if (ext) suffix = ext;
  } catch {
    // ignore
  }
  const filePath = path.join(dir, `video${suffix}`);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: controller.signal });
    if (!res.ok) {
      throw new Error(`Download failed: ${res.status} ${res.statusText}`);
    }
    const totalRaw = res.headers.get("content-length");
    const total = totalRaw ? Number(totalRaw) : 0;
    const hasTotal = Number.isFinite(total) && total > 0;
    const reader = res.body?.getReader();
    if (!reader) {
      throw new Error("Download failed: missing response body");
    }
    const handle = await fs.open(filePath, "w");
    let downloaded = 0;
    let lastPercent = -1;
    let lastReportedBytes = 0;
    const reportProgress = () => {
      if (!onProgress) return;
      if (hasTotal) {
        const percent = Math.max(0, Math.min(100, Math.round((downloaded / total) * 100)));
        if (percent === lastPercent) return;
        lastPercent = percent;
        const detail = `(${formatBytes(downloaded)}/${formatBytes(total)})`;
        onProgress(percent, detail);
        return;
      }
      if (downloaded - lastReportedBytes < 2 * 1024 * 1024) return;
      lastReportedBytes = downloaded;
      onProgress(0, `(${formatBytes(downloaded)})`);
    };
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value) continue;
        await handle.write(value);
        downloaded += value.byteLength;
        reportProgress();
      }
    } finally {
      await handle.close();
    }
    if (hasTotal) {
      onProgress?.(100, `(${formatBytes(downloaded)}/${formatBytes(total)})`);
    }
    return {
      filePath,
      cleanup: async () => {
        await fs.rm(dir, { recursive: true, force: true });
      },
    };
  } catch (error) {
    await fs.rm(dir, { recursive: true, force: true }).catch(() => null);
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function resolveYoutubeStreamUrl({
  ytDlpPath,
  url,
  timeoutMs,
  format,
  cookiesFromBrowser,
}: {
  ytDlpPath: string;
  url: string;
  timeoutMs: number;
  format: string;
  cookiesFromBrowser?: string | null;
}): Promise<string> {
  const args = ["-f", format, ...buildYtDlpCookiesArgs(cookiesFromBrowser), "-g", url];
  const output = await runProcessCapture({
    command: ytDlpPath,
    args,
    timeoutMs: Math.max(timeoutMs, YT_DLP_TIMEOUT_MS),
    errorLabel: "yt-dlp",
  });
  const lines = output
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    throw new Error("yt-dlp did not return a stream URL.");
  }
  return lines[0];
}

async function detectSlideTimestamps({
  ffmpegPath,
  ffprobePath,
  inputPath,
  sceneThreshold,
  autoTuneThreshold,
  env,
  timeoutMs,
  warnings,
  workers,
  sampleCount,
  onSegmentProgress,
  logSlides,
  logSlidesTiming,
}: {
  ffmpegPath: string;
  ffprobePath: string | null;
  inputPath: string;
  sceneThreshold: number;
  autoTuneThreshold: boolean;
  env: Record<string, string | undefined>;
  timeoutMs: number;
  warnings: string[];
  workers: number;
  sampleCount: number;
  onSegmentProgress?: ((completed: number, total: number) => void) | null;
  logSlides?: ((message: string) => void) | null;
  logSlidesTiming?: ((label: string, startedAt: number) => number) | null;
}): Promise<{ timestamps: number[]; autoTune: SlideAutoTune; durationSeconds: number | null }> {
  const probeStartedAt = Date.now();
  const resolvedFfprobePath = ffprobePath ?? resolveExecutableInPath("ffprobe", env);
  const videoInfo = resolvedFfprobePath
    ? await probeVideoInfo({
        ffprobePath: resolvedFfprobePath,
        inputPath,
        timeoutMs,
      })
    : { durationSeconds: null, width: null, height: null };
  logSlidesTiming?.("ffprobe video info", probeStartedAt);

  const calibration = await calibrateSceneThreshold({
    ffmpegPath,
    inputPath,
    durationSeconds: videoInfo.durationSeconds,
    sampleCount,
    timeoutMs,
    logSlides,
  });

  const baseThreshold = sceneThreshold;
  const calibratedThreshold = calibration.threshold;
  const chosenThreshold = autoTuneThreshold ? calibratedThreshold : baseThreshold;
  if (autoTuneThreshold && chosenThreshold !== baseThreshold) {
    warnings.push(`Auto-tuned scene threshold from ${baseThreshold} to ${chosenThreshold}`);
  }

  const segments = buildSegments(videoInfo.durationSeconds, workers);
  const detectStartedAt = Date.now();
  let effectiveThreshold = chosenThreshold;
  let timestamps = await detectSceneTimestamps({
    ffmpegPath,
    inputPath,
    threshold: effectiveThreshold,
    timeoutMs,
    segments,
    workers,
    onSegmentProgress,
    runWithConcurrency,
  });
  logSlidesTiming?.(
    `scene detection base (threshold=${effectiveThreshold}, segments=${segments.length})`,
    detectStartedAt,
  );

  if (timestamps.length === 0) {
    const fallbackThreshold = Math.max(0.05, roundThreshold(effectiveThreshold * 0.5));
    if (fallbackThreshold !== effectiveThreshold) {
      const retryStartedAt = Date.now();
      timestamps = await detectSceneTimestamps({
        ffmpegPath,
        inputPath,
        threshold: fallbackThreshold,
        timeoutMs,
        segments,
        workers,
        onSegmentProgress,
        runWithConcurrency,
      });
      logSlidesTiming?.(
        `scene detection retry (threshold=${fallbackThreshold}, segments=${segments.length})`,
        retryStartedAt,
      );
      warnings.push(
        `Scene detection retry used lower threshold ${fallbackThreshold} after zero detections`,
      );
      if (timestamps.length > 0) {
        effectiveThreshold = fallbackThreshold;
      }
    }
  }

  const autoTune: SlideAutoTune = autoTuneThreshold
    ? {
        enabled: true,
        chosenThreshold: timestamps.length > 0 ? effectiveThreshold : baseThreshold,
        confidence: calibration.confidence,
        strategy: "hash",
      }
    : {
        enabled: false,
        chosenThreshold: baseThreshold,
        confidence: 0,
        strategy: "none",
      };

  return { timestamps, autoTune, durationSeconds: videoInfo.durationSeconds };
}

async function extractFramesAtTimestamps({
  ffmpegPath,
  inputPath,
  outputDir,
  timestamps,
  segments,
  durationSeconds,
  timeoutMs,
  workers,
  onProgress,
  onStatus,
  onSlide,
  logSlides,
  logSlidesTiming,
}: {
  ffmpegPath: string;
  inputPath: string;
  outputDir: string;
  timestamps: number[];
  segments?: Array<{ start: number; end: number | null } | null>;
  durationSeconds?: number | null;
  timeoutMs: number;
  workers: number;
  onProgress?: ((completed: number, total: number) => void) | null;
  onStatus?: ((text: string) => void) | null;
  onSlide?: ((slide: SlideImage) => void) | null;
  logSlides?: ((message: string) => void) | null;
  logSlidesTiming?: ((label: string, startedAt: number) => number) | null;
}): Promise<SlideImage[]> {
  type FrameStats = { ymin: number | null; ymax: number | null; yavg: number | null };
  type FrameQuality = { brightness: number; contrast: number };

  const FRAME_ADJUST_RANGE_SECONDS = 10;
  const FRAME_ADJUST_STEP_SECONDS = 2;
  const FRAME_MIN_BRIGHTNESS = 0.24;
  const FRAME_MIN_CONTRAST = 0.16;
  const SEEK_PAD_SECONDS = 8;

  const clampTimestamp = (value: number) => {
    const upper =
      typeof durationSeconds === "number" && Number.isFinite(durationSeconds) && durationSeconds > 0
        ? Math.max(0, durationSeconds - 0.1)
        : Number.POSITIVE_INFINITY;
    return clamp(value, 0, upper);
  };

  const resolveSegmentBounds = (segment: { start: number; end: number | null } | null) => {
    if (!segment) return null;
    const start = Math.max(0, segment.start);
    const end =
      typeof segment.end === "number" && Number.isFinite(segment.end) ? segment.end : null;
    if (end != null && end <= start) return null;
    return { start, end };
  };

  const resolveSegmentPadding = (segment: { start: number; end: number | null } | null) => {
    if (!segment || segment.end == null) return 0;
    const duration = Math.max(0, segment.end - segment.start);
    if (duration <= 0) return 0;
    return Math.min(1.5, Math.max(0.2, duration * 0.08));
  };

  const parseSignalstats = (line: string, stats: FrameStats): void => {
    if (!line.includes("lavfi.signalstats.")) return;
    const match = line.match(/lavfi\.signalstats\.(YMIN|YMAX|YAVG)=(\d+(?:\.\d+)?)/);
    if (!match) return;
    const value = Number(match[2]);
    if (!Number.isFinite(value)) return;
    if (match[1] === "YMIN") stats.ymin = value;
    if (match[1] === "YMAX") stats.ymax = value;
    if (match[1] === "YAVG") stats.yavg = value;
  };

  const toQuality = (stats: FrameStats): FrameQuality | null => {
    if (stats.ymin == null || stats.ymax == null || stats.yavg == null) return null;
    const brightness = clamp(stats.yavg / 255, 0, 1);
    const contrast = clamp((stats.ymax - stats.ymin) / 255, 0, 1);
    return { brightness, contrast };
  };

  const scoreQuality = (quality: FrameQuality, deltaSeconds: number) => {
    const penalty = Math.min(1, Math.abs(deltaSeconds) / FRAME_ADJUST_RANGE_SECONDS) * 0.05;
    // Prefer brighter frames (dark-but-contrasty thumbnails are still unpleasant).
    return quality.brightness * 0.55 + quality.contrast * 0.45 - penalty;
  };

  const extractFrame = async (
    timestamp: number,
    outputPath: string,
    opts?: { timeoutMs?: number },
  ): Promise<{
    slide: SlideImage;
    quality: FrameQuality | null;
    actualTimestamp: number | null;
    seekBase: number;
  }> => {
    const stats: FrameStats = { ymin: null, ymax: null, yavg: null };
    let actualTimestamp: number | null = null;
    const effectiveTimeoutMs =
      typeof opts?.timeoutMs === "number" && Number.isFinite(opts.timeoutMs) && opts.timeoutMs > 0
        ? opts.timeoutMs
        : timeoutMs;
    const seekBase = Math.max(0, timestamp - SEEK_PAD_SECONDS);
    const seekOffset = Math.max(0, timestamp - seekBase);
    const args = [
      "-hide_banner",
      ...(seekBase > 0 ? ["-ss", String(seekBase)] : []),
      "-i",
      inputPath,
      ...(seekOffset > 0 ? ["-ss", String(seekOffset)] : []),
      "-vf",
      "signalstats,showinfo,metadata=print",
      "-vframes",
      "1",
      "-q:v",
      "2",
      "-an",
      "-sn",
      "-update",
      "1",
      outputPath,
    ];
    await runProcess({
      command: ffmpegPath,
      args,
      timeoutMs: effectiveTimeoutMs,
      errorLabel: "ffmpeg",
      onStderrLine: (line) => {
        if (actualTimestamp == null) {
          const parsed = parseShowinfoTimestamp(line);
          if (parsed != null) actualTimestamp = parsed;
        }
        parseSignalstats(line, stats);
      },
    });
    const stat = await fs.stat(outputPath).catch(() => null);
    if (!stat?.isFile() || stat.size === 0) {
      throw new Error(`ffmpeg produced no output frame at ${outputPath}`);
    }
    const quality = toQuality(stats);
    return {
      slide: { index: 0, timestamp, imagePath: outputPath },
      quality,
      actualTimestamp,
      seekBase,
    };
  };

  const slides: SlideImage[] = [];
  const startedAt = Date.now();
  const tasks = timestamps.map((timestamp, index) => async () => {
    const segment = segments?.[index] ?? null;
    const bounds = resolveSegmentBounds(segment);
    const padding = resolveSegmentPadding(segment);
    const clampedTimestamp = clampTimestamp(timestamp);
    const safeTimestamp =
      bounds && bounds.end != null
        ? bounds.end - padding <= bounds.start + padding
          ? clampTimestamp(bounds.start + (bounds.end - bounds.start) * 0.5)
          : clamp(clampedTimestamp, bounds.start + padding, bounds.end - padding)
        : bounds
          ? Math.max(bounds.start + padding, clampedTimestamp)
          : clampedTimestamp;
    const outputPath = path.join(outputDir, `slide_${String(index + 1).padStart(4, "0")}.png`);
    const extracted = await extractFrame(safeTimestamp, outputPath);
    const resolvedTimestamp = resolveExtractedTimestamp({
      requested: safeTimestamp,
      actual: extracted.actualTimestamp,
      seekBase: extracted.seekBase,
    });
    const delta = resolvedTimestamp - safeTimestamp;
    if (Math.abs(delta) >= 0.25) {
      const actualLabel =
        extracted.actualTimestamp != null && Number.isFinite(extracted.actualTimestamp)
          ? extracted.actualTimestamp.toFixed(2)
          : "n/a";
      logSlides?.(
        `frame pts slide=${index + 1} req=${safeTimestamp.toFixed(2)}s actual=${actualLabel}s base=${extracted.seekBase.toFixed(2)}s -> ${resolvedTimestamp.toFixed(2)}s delta=${delta.toFixed(2)}s`,
      );
    }
    const imageVersion = Date.now();
    onSlide?.({
      index: index + 1,
      timestamp: resolvedTimestamp,
      imagePath: outputPath,
      imageVersion,
    });
    return {
      index: index + 1,
      timestamp: resolvedTimestamp,
      requestedTimestamp: safeTimestamp,
      imagePath: outputPath,
      quality: extracted.quality,
      imageVersion,
      segment: bounds,
    };
  });
  const results = await runWithConcurrency(tasks, workers, onProgress ?? undefined);
  const ordered = results.filter(Boolean).sort((a, b) => a.index - b.index);

  const fixTasks: Array<() => Promise<void>> = [];
  for (const frame of ordered) {
    slides.push({
      index: frame.index,
      timestamp: frame.timestamp,
      imagePath: frame.imagePath,
      imageVersion: frame.imageVersion,
    });
    const quality = frame.quality;
    if (!quality) continue;
    const shouldPreferBrighterFirstSlide = frame.index === 1 && frame.timestamp < 8;
    const needsAdjust =
      quality.brightness < FRAME_MIN_BRIGHTNESS ||
      quality.contrast < FRAME_MIN_CONTRAST ||
      (shouldPreferBrighterFirstSlide && (quality.brightness < 0.58 || quality.contrast < 0.2));
    if (!needsAdjust) continue;
    fixTasks.push(async () => {
      const bounds = resolveSegmentBounds(frame.segment ?? null);
      const padding = resolveSegmentPadding(frame.segment ?? null);
      const minTs = bounds
        ? clampTimestamp(bounds.start + padding)
        : clampTimestamp(frame.timestamp - FRAME_ADJUST_RANGE_SECONDS);
      const maxTs =
        bounds && bounds.end != null
          ? clampTimestamp(bounds.end - padding)
          : clampTimestamp(frame.timestamp + FRAME_ADJUST_RANGE_SECONDS);
      if (maxTs <= minTs) return;
      const baseTimestamp = clamp(frame.timestamp, minTs, maxTs);
      const maxRange = Math.min(FRAME_ADJUST_RANGE_SECONDS, maxTs - minTs);
      if (!Number.isFinite(maxRange) || maxRange < FRAME_ADJUST_STEP_SECONDS) return;
      const candidateOffsets: number[] = [];
      for (
        let offset = FRAME_ADJUST_STEP_SECONDS;
        offset <= maxRange;
        offset += FRAME_ADJUST_STEP_SECONDS
      ) {
        candidateOffsets.push(offset, -offset);
      }
      let best = {
        timestamp: baseTimestamp,
        offsetSeconds: 0,
        quality,
        score: scoreQuality(quality, 0),
      };
      let selectedTimestamp = baseTimestamp;
      let didReplace = false;
      const minImproveDelta = shouldPreferBrighterFirstSlide ? 0.015 : 0.03;
      for (const offsetSeconds of candidateOffsets) {
        if (offsetSeconds === 0) continue;
        const candidateTimestamp = clamp(baseTimestamp + offsetSeconds, minTs, maxTs);
        if (Math.abs(candidateTimestamp - baseTimestamp) < 0.01) continue;
        const tempPath = path.join(
          outputDir,
          `slide_${String(frame.index).padStart(4, "0")}_alt.png`,
        );
        try {
          const candidate = await extractFrame(candidateTimestamp, tempPath, {
            timeoutMs: Math.min(timeoutMs, 12_000),
          });
          if (!candidate.quality) continue;
          const resolvedCandidateTimestamp = resolveExtractedTimestamp({
            requested: candidateTimestamp,
            actual: candidate.actualTimestamp,
            seekBase: candidate.seekBase,
          });
          const score = scoreQuality(candidate.quality, offsetSeconds);
          if (score > best.score + minImproveDelta) {
            best = {
              timestamp: resolvedCandidateTimestamp,
              offsetSeconds,
              quality: candidate.quality,
              score,
            };
            try {
              await fs.rename(tempPath, frame.imagePath);
            } catch (err) {
              const code = err && typeof err === "object" && "code" in err ? String(err.code) : "";
              if (code === "EEXIST") {
                await fs.rm(frame.imagePath, { force: true }).catch(() => null);
                await fs.rename(tempPath, frame.imagePath);
              } else {
                throw err;
              }
            }
            didReplace = true;
            selectedTimestamp = resolvedCandidateTimestamp;
          } else {
            await fs.rm(tempPath, { force: true }).catch(() => null);
          }
        } catch {
          await fs.rm(tempPath, { force: true }).catch(() => null);
        }
      }
      if (!didReplace) return;
      const updatedVersion = Date.now();
      const slide = slides[frame.index - 1];
      if (slide) {
        slide.imageVersion = updatedVersion;
        slide.timestamp = selectedTimestamp;
      }
      if (selectedTimestamp !== frame.timestamp) {
        const offsetSeconds = (selectedTimestamp - frame.timestamp).toFixed(2);
        const baseBrightness = quality.brightness.toFixed(2);
        const baseContrast = quality.contrast.toFixed(2);
        const bestBrightness = best.quality?.brightness?.toFixed(2) ?? baseBrightness;
        const bestContrast = best.quality?.contrast?.toFixed(2) ?? baseContrast;
        logSlides?.(
          `thumbnail adjust slide=${frame.index} ts=${frame.timestamp.toFixed(2)}s -> ${selectedTimestamp.toFixed(2)}s offset=${offsetSeconds}s base=${baseBrightness}/${baseContrast} best=${bestBrightness}/${bestContrast}`,
        );
      }
      onSlide?.({
        index: frame.index,
        timestamp: selectedTimestamp,
        imagePath: frame.imagePath,
        imageVersion: updatedVersion,
      });
    });
  }
  if (fixTasks.length > 0) {
    const fixStartedAt = Date.now();
    const THUMB_START = 90;
    const THUMB_END = 96;
    // Avoid UI "stuck" at a static percent while we do expensive refinement passes.
    onStatus?.(`Slides: improving thumbnails ${THUMB_START}%`);
    logSlides?.(
      `thumbnail adjust start count=${fixTasks.length} range=±${FRAME_ADJUST_RANGE_SECONDS}s step=${FRAME_ADJUST_STEP_SECONDS}s`,
    );
    await runWithConcurrency(fixTasks, Math.min(4, workers), (completed, total) => {
      const ratio = total > 0 ? completed / total : 0;
      const percent = Math.round(THUMB_START + ratio * (THUMB_END - THUMB_START));
      onStatus?.(`Slides: improving thumbnails ${percent}%`);
    });
    onStatus?.(`Slides: improving thumbnails ${THUMB_END}%`);
    logSlidesTiming?.("thumbnail adjust done", fixStartedAt);
  }
  logSlidesTiming?.(
    `extract frame loop (count=${timestamps.length}, workers=${workers})`,
    startedAt,
  );
  return slides;
}

async function renameSlidesWithTimestamps(
  slides: SlideImage[],
  slidesDir: string,
): Promise<SlideImage[]> {
  const renamed: SlideImage[] = [];
  for (const slide of slides) {
    const timestampLabel = slide.timestamp.toFixed(2);
    const filename = `slide_${slide.index.toString().padStart(4, "0")}_${timestampLabel}s.png`;
    const nextPath = path.join(slidesDir, filename);
    if (slide.imagePath !== nextPath) {
      await fs.rename(slide.imagePath, nextPath).catch(async () => {
        await fs.copyFile(slide.imagePath, nextPath);
        await fs.rm(slide.imagePath, { force: true });
      });
    }
    renamed.push({ ...slide, imagePath: nextPath });
  }
  return renamed;
}

async function withSlidesLock<T>(
  key: string,
  fn: () => Promise<T>,
  onWait?: (() => void) | null,
): Promise<T> {
  const previous = slidesLocks.get(key) ?? null;
  if (previous && onWait) onWait();
  let release = () => {};
  const current = new Promise<void>((resolve) => {
    release = resolve;
  });
  slidesLocks.set(key, current);
  await (previous ?? Promise.resolve());
  try {
    return await fn();
  } finally {
    release();
    if (slidesLocks.get(key) === current) {
      slidesLocks.delete(key);
    }
  }
}

async function writeSlidesJson(result: SlideExtractionResult, slidesDir: string): Promise<void> {
  const slidesDirId = result.slidesDirId ?? buildSlidesDirId(slidesDir);
  const payload = {
    sourceUrl: result.sourceUrl,
    sourceKind: result.sourceKind,
    sourceId: result.sourceId,
    slidesDir,
    slidesDirId,
    sceneThreshold: result.sceneThreshold,
    autoTuneThreshold: result.autoTuneThreshold,
    autoTune: result.autoTune,
    maxSlides: result.maxSlides,
    minSlideDuration: result.minSlideDuration,
    ocrRequested: result.ocrRequested,
    ocrAvailable: result.ocrAvailable,
    slideCount: result.slides.length,
    warnings: result.warnings,
    slides: result.slides.map((slide) => ({
      ...slide,
      imagePath: serializeSlideImagePath(slidesDir, slide.imagePath),
    })),
  };
  await fs.writeFile(path.join(slidesDir, "slides.json"), JSON.stringify(payload, null, 2), "utf8");
}
