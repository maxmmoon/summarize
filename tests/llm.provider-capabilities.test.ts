import { describe, expect, it } from "vitest";
import {
  DEFAULT_AUTO_CLI_ORDER,
  DEFAULT_CLI_MODELS,
  envHasRequiredKey,
  parseCliProviderName,
  requiredEnvForCliProvider,
  requiredEnvForGatewayProvider,
  supportsDocumentAttachments,
  supportsStreaming,
} from "../src/llm/provider-capabilities.js";

describe("llm provider capabilities", () => {
  it("exposes stable CLI defaults and parsing", () => {
    expect(DEFAULT_AUTO_CLI_ORDER).toEqual(["claude", "gemini", "codex", "agent"]);
    expect(DEFAULT_CLI_MODELS.gemini).toBe("gemini-3-flash");
    expect(parseCliProviderName(" GeMiNi ")).toBe("gemini");
    expect(requiredEnvForCliProvider("agent")).toBe("CLI_AGENT");
  });

  it("tracks native provider capabilities centrally", () => {
    expect(requiredEnvForGatewayProvider("google")).toBe("GEMINI_API_KEY");
    expect(supportsDocumentAttachments("google")).toBe(true);
    expect(supportsDocumentAttachments("xai")).toBe(false);
    expect(supportsStreaming("anthropic")).toBe(true);
  });

  it("handles provider env aliases", () => {
    expect(
      envHasRequiredKey(
        {
          GOOGLE_GENERATIVE_AI_API_KEY: "gemini",
        },
        "GEMINI_API_KEY",
      ),
    ).toBe(true);
    expect(envHasRequiredKey({ ZAI_API_KEY: "z" }, "Z_AI_API_KEY")).toBe(true);
    expect(envHasRequiredKey({}, "OPENAI_API_KEY")).toBe(false);
  });
});
