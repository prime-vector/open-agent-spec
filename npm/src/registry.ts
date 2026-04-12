import { parseSpec } from "./loader.js";
import type { OASpec } from "./types.js";
import { OAError } from "./loader.js";

const REGISTRY_BASE = "https://openagentspec.dev/registry";
const OA_SCHEME = "oa://";

/**
 * Expand an `oa://namespace/name[@version]` shorthand to a full HTTPS URL.
 *
 *   oa://prime-vector/summariser        → .../prime-vector/summariser/latest/spec.yaml
 *   oa://prime-vector/summariser@1.0.0  → .../prime-vector/summariser/1.0.0/spec.yaml
 */
export function resolveSpecUrl(ref: string): string {
  if (!ref.startsWith(OA_SCHEME)) return ref; // already a plain HTTP(S) URL

  let rest = ref.slice(OA_SCHEME.length);
  let version = "latest";

  if (rest.includes("@")) {
    const at = rest.lastIndexOf("@");
    version = rest.slice(at + 1);
    rest = rest.slice(0, at);
  }

  const parts = rest.replace(/^\/|\/$/g, "").split("/");
  if (parts.length !== 2) {
    throw new OAError(
      `Invalid oa:// reference '${ref}'. Expected: oa://namespace/name or oa://namespace/name@version`,
      "SPEC_LOAD_ERROR",
      "delegation",
    );
  }

  const [namespace, name] = parts;
  return `${REGISTRY_BASE}/${namespace}/${name}/${version}/spec.yaml`;
}

export function isRemoteRef(ref: string): boolean {
  return (
    ref.startsWith(OA_SCHEME) ||
    ref.startsWith("https://") ||
    ref.startsWith("http://")
  );
}

export async function fetchRemoteSpec(url: string): Promise<OASpec> {
  let res: Response;
  try {
    res = await fetch(url, {
      headers: { Accept: "application/yaml, text/yaml, text/plain, */*" },
    });
  } catch (err) {
    throw new OAError(
      `Registry fetch failed for '${url}': ${String(err)}`,
      "SPEC_LOAD_ERROR",
      "delegation",
    );
  }

  if (!res.ok) {
    throw new OAError(
      `Registry fetch failed for '${url}': HTTP ${res.status} ${res.statusText}`,
      "SPEC_LOAD_ERROR",
      "delegation",
    );
  }

  const raw = await res.text();
  return parseSpec(raw, url);
}
