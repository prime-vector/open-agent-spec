import Link from "next/link";
import registryData from "../../public/registry/index.json";

const TAG_COLOURS: Record<string, string> = {
  text: "bg-blue-50 text-blue-700 border-blue-200",
  nlp: "bg-violet-50 text-violet-700 border-violet-200",
  summarisation: "bg-amber-50 text-amber-700 border-amber-200",
  sentiment: "bg-pink-50 text-pink-700 border-pink-200",
  classification: "bg-green-50 text-green-700 border-green-200",
  keywords: "bg-teal-50 text-teal-700 border-teal-200",
  extraction: "bg-cyan-50 text-cyan-700 border-cyan-200",
  code: "bg-orange-50 text-orange-700 border-orange-200",
  review: "bg-red-50 text-red-700 border-red-200",
  security: "bg-rose-50 text-rose-700 border-rose-200",
  quality: "bg-lime-50 text-lime-700 border-lime-200",
};

const DEFAULT_TAG = "bg-stone-100 text-stone-600 border-stone-300";

function tagClass(tag: string) {
  return TAG_COLOURS[tag] ?? DEFAULT_TAG;
}

type RegistrySpec = (typeof registryData.specs)[number];

function UsageSnippet({ spec }: { spec: RegistrySpec }) {
  const lines = [
    `tasks:`,
    `  my_task:`,
    `    description: Delegate to ${spec.name}`,
    `    spec: oa://${spec.id}`,
    `    task: ${spec.task}`,
  ].join("\n");

  return (
    <div className="overflow-hidden rounded-lg border border-stone-700 bg-stone-900">
      <div className="border-b border-stone-700/60 px-3 py-1.5">
        <span className="text-[10px] font-medium text-stone-400">
          usage — paste into any spec
        </span>
      </div>
      <pre className="overflow-x-auto px-3 py-2.5 font-mono text-[10px] leading-relaxed text-stone-300">
        {lines}
      </pre>
    </div>
  );
}

function SpecCard({ spec }: { spec: RegistrySpec }) {
  return (
    <div className="flex flex-col rounded-xl border border-stone-300/60 bg-white/70 p-4 shadow-sm backdrop-blur-sm">
      {/* Header */}
      <div className="mb-3 flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold text-stone-900">{spec.name}</h2>
            <span className="rounded-full border border-stone-300/60 bg-stone-100 px-2 py-0.5 text-[10px] font-medium text-stone-500">
              v{spec.latest}
            </span>
          </div>
          <p className="mt-0.5 font-mono text-[10px] text-stone-500">
            oa://{spec.id}
          </p>
        </div>
        <a
          href={spec.url}
          target="_blank"
          rel="noopener noreferrer"
          className="shrink-0 rounded-lg border border-stone-300/60 bg-stone-50 px-2.5 py-1 text-[10px] font-medium text-stone-600 transition hover:bg-stone-100"
        >
          View YAML ↗
        </a>
      </div>

      {/* Description */}
      <p className="mb-3 text-xs leading-relaxed text-stone-600">
        {spec.description}
      </p>

      {/* Tags */}
      <div className="mb-4 flex flex-wrap gap-1">
        {spec.tags.map((tag) => (
          <span
            key={tag}
            className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${tagClass(tag)}`}
          >
            {tag}
          </span>
        ))}
      </div>

      {/* Usage snippet */}
      <div className="mt-auto">
        <UsageSnippet spec={spec} />
      </div>
    </div>
  );
}

export default function RegistryPage() {
  const specs = registryData.specs;
  const namespaces = Array.from(new Set(specs.map((s) => s.namespace)));

  return (
    <main className="bg-marketing min-h-screen text-stone-900">
      <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 lg:px-10">

        {/* Back nav */}
        <div className="mb-8">
          <Link
            href="/"
            className="text-xs text-stone-500 transition hover:text-stone-800"
          >
            ← Open Agent Spec
          </Link>
        </div>

        {/* Hero */}
        <div className="mb-10">
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <h1 className="font-serif text-3xl font-semibold tracking-tight text-stone-900 sm:text-4xl">
              OA Spec Registry
            </h1>
            <span className="rounded-full border border-stone-400/50 bg-white/70 px-3 py-1 text-[10px] font-medium text-stone-600 backdrop-blur-sm">
              {specs.length} specs · {namespaces.length} namespace
            </span>
          </div>
          <p className="max-w-2xl text-sm leading-relaxed text-stone-600">
            Shared, versioned, reusable agent specs. Reference any spec from your
            own YAML using the{" "}
            <code className="rounded bg-stone-200/80 px-1 text-stone-800">
              oa://
            </code>{" "}
            shorthand — no copy-paste, no duplication.
          </p>
        </div>

        {/* How it works — compact explainer */}
        <div className="mb-10 overflow-hidden rounded-xl border border-stone-300/50 bg-white/50 p-4 shadow-sm backdrop-blur-sm sm:p-6">
          <h2 className="mb-4 text-xs font-semibold uppercase tracking-wide text-stone-900">
            How registry specs work
          </h2>
          <div className="grid gap-6 sm:grid-cols-2">
            <div>
              <p className="mb-3 text-xs leading-relaxed text-stone-600">
                Any task in your spec can delegate its implementation to a
                registry spec. The{" "}
                <code className="rounded bg-stone-200 px-1 text-stone-800">
                  oa run
                </code>{" "}
                CLI fetches it, resolves the task, and returns the result
                transparently.
              </p>
              <div className="grid gap-2 text-xs text-stone-600 sm:grid-cols-1">
                {[
                  ["oa://namespace/name", "Always pulls the latest version"],
                  ["oa://namespace/name@1.0.0", "Pinned to a specific version"],
                  ["https://…/spec.yaml", "Any third-party spec URL"],
                ].map(([code, desc]) => (
                  <div key={code} className="flex items-start gap-2">
                    <code className="shrink-0 rounded bg-stone-200/80 px-1.5 py-0.5 text-[10px] text-stone-800">
                      {code}
                    </code>
                    <span className="text-stone-500">{desc}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="overflow-hidden rounded-lg border border-stone-700 bg-stone-900">
              <div className="border-b border-stone-700/60 px-3 py-1.5">
                <span className="text-[10px] font-medium text-stone-400">
                  my-agent.yaml
                </span>
              </div>
              <pre className="overflow-x-auto px-3 py-3 font-mono text-[10px] leading-relaxed text-stone-300">
                {`open_agent_spec: "1.5.0"

agent:
  name: my-pipeline
  description: Research pipeline

tasks:
  summarise:
    description: Delegate to shared summariser
    spec: oa://prime-vector/summariser
    task: summarise

  sentiment:
    description: Analyse summary tone
    spec: oa://prime-vector/sentiment@1.0.0
    task: analyse_sentiment
    depends_on: [summarise]`}
              </pre>
            </div>
          </div>
        </div>

        {/* Spec grid */}
        <div className="mb-4 text-xs font-semibold uppercase tracking-wide text-stone-500">
          prime-vector · official specs
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {specs.map((spec) => (
            <SpecCard key={spec.id} spec={spec} />
          ))}
        </div>

        {/* Publish your own — CTA */}
        <div className="mt-12 rounded-xl border border-stone-300/50 bg-white/50 p-5 text-center shadow-sm backdrop-blur-sm">
          <p className="mb-1 text-sm font-semibold text-stone-900">
            Publish your own spec
          </p>
          <p className="mb-4 text-xs text-stone-600">
            Anyone can contribute a spec. Open a PR to the{" "}
            <code className="rounded bg-stone-200 px-1 text-stone-800">
              Website/public/registry/
            </code>{" "}
            directory and add your namespace.
          </p>
          <a
            href="https://github.com/prime-vector/open-agent-spec"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center rounded-lg bg-stone-900 px-5 py-2 text-xs font-medium text-stone-50 shadow transition hover:bg-stone-800"
          >
            Contribute on GitHub ↗
          </a>
        </div>

        {/* Footer breadcrumb */}
        <div className="mt-10 text-center text-[10px] text-stone-400">
          Served at{" "}
          <code className="rounded bg-stone-200/60 px-1 text-stone-600">
            openagentspec.dev/registry/index.json
          </code>
        </div>
      </div>
    </main>
  );
}
