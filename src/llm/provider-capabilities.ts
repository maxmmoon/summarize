import type { CliProvider } from "../config.js";

export type GatewayProvider = "xai" | "openai" | "google" | "anthropic" | "zai" | "nvidia";

export type RequiredModelEnv =
  | "XAI_API_KEY"
  | "OPENAI_API_KEY"
  | "NVIDIA_API_KEY"
  | "GEMINI_API_KEY"
  | "ANTHROPIC_API_KEY"
  | "OPENROUTER_API_KEY"
  | "Z_AI_API_KEY"
  | "CLI_CLAUDE"
  | "CLI_CODEX"
  | "CLI_GEMINI"
  | "CLI_AGENT";

type ProviderCapability = {
  requiredEnv: RequiredModelEnv;
  supportsDocuments: boolean;
  supportsStreaming: boolean;
};

const PROVIDER_CAPABILITIES: Record<GatewayProvider, ProviderCapability> = {
  xai: { requiredEnv: "XAI_API_KEY", supportsDocuments: false, supportsStreaming: true },
  openai: { requiredEnv: "OPENAI_API_KEY", supportsDocuments: true, supportsStreaming: true },
  google: { requiredEnv: "GEMINI_API_KEY", supportsDocuments: true, supportsStreaming: true },
  anthropic: {
    requiredEnv: "ANTHROPIC_API_KEY",
    supportsDocuments: true,
    supportsStreaming: true,
  },
  zai: { requiredEnv: "Z_AI_API_KEY", supportsDocuments: false, supportsStreaming: true },
  nvidia: { requiredEnv: "NVIDIA_API_KEY", supportsDocuments: false, supportsStreaming: true },
};

export const DEFAULT_CLI_MODELS: Record<CliProvider, string> = {
  claude: "sonnet",
  codex: "gpt-5.2",
  gemini: "gemini-3-flash",
  agent: "gpt-5.2",
};

export const DEFAULT_AUTO_CLI_ORDER: CliProvider[] = ["claude", "gemini", "codex", "agent"];

export function parseCliProviderName(raw: string): CliProvider | null {
  const normalized = raw.trim().toLowerCase();
  if (normalized === "claude") return "claude";
  if (normalized === "codex") return "codex";
  if (normalized === "gemini") return "gemini";
  if (normalized === "agent") return "agent";
  return null;
}

export function requiredEnvForCliProvider(provider: CliProvider): RequiredModelEnv {
  return provider === "codex"
    ? "CLI_CODEX"
    : provider === "gemini"
      ? "CLI_GEMINI"
      : provider === "agent"
        ? "CLI_AGENT"
        : "CLI_CLAUDE";
}

export function requiredEnvForGatewayProvider(provider: GatewayProvider): RequiredModelEnv {
  return PROVIDER_CAPABILITIES[provider].requiredEnv;
}

export function supportsDocumentAttachments(provider: GatewayProvider): boolean {
  return PROVIDER_CAPABILITIES[provider].supportsDocuments;
}

export function supportsStreaming(provider: GatewayProvider): boolean {
  return PROVIDER_CAPABILITIES[provider].supportsStreaming;
}

export function envHasRequiredKey(
  env: Record<string, string | undefined>,
  requiredEnv: RequiredModelEnv,
): boolean {
  if (requiredEnv === "GEMINI_API_KEY") {
    return Boolean(
      env.GEMINI_API_KEY?.trim() ||
        env.GOOGLE_GENERATIVE_AI_API_KEY?.trim() ||
        env.GOOGLE_API_KEY?.trim(),
    );
  }
  if (requiredEnv === "Z_AI_API_KEY") {
    return Boolean(env.Z_AI_API_KEY?.trim() || env.ZAI_API_KEY?.trim());
  }
  return Boolean(env[requiredEnv]?.trim());
}

