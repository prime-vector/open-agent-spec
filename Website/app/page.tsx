import Image from "next/image";
import Link from "next/link";
import Logo from "../OAS-Logo.webp";
import CliGif from "../oacli.gif";
import AacDemoGif from "../oaaacdemog.gif";
import DiagramOa from "../diagramoa.png";
import InstallCard from "../components/InstallCard";

export default function HomePage() {
  return (
    <main className="bg-marketing relative min-h-screen overflow-x-hidden text-stone-900">
      {/* Large OA watermark — behind content */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed left-0 top-0 z-0 select-none overflow-hidden"
      >
        <span className="block -translate-x-4 -translate-y-8 font-serif text-[min(42vw,420px)] font-semibold leading-none tracking-[0.08em] text-stone-300/40 sm:translate-x-0 sm:translate-y-0 sm:text-[min(38vw,480px)] md:text-[min(36vw,520px)]">
          OA
        </span>
      </div>

      <div className="relative z-10 flex min-h-screen">
        {/* Sidebar — collapses on mobile via hidden sm:block */}
        <aside className="hidden w-48 shrink-0 border-r border-stone-300/50 bg-stone-200/30 px-3 py-6 text-xs text-stone-600 backdrop-blur-sm sm:block lg:w-52">
          <div className="mb-4 text-[10px] font-semibold uppercase tracking-wide text-stone-800">
            Open Agent Spec
          </div>
          <nav className="space-y-1">
            {[
              ["#overview", "Overview"],
              ["#install", "Install"],
              ["#whats-new", "What's New"],
              ["#features", "Features"],
              ["#registry", "Registry"],
              ["#problem", "The Problem"],
              ["#how-it-works", "How It Works"],
              ["#modes", "Two Ways"],
              ["#repo-native", "Repo-native"],
              ["#ci", "CI & Sub-agents"],
              ["#not", "What OA Is Not"],
              ["#why", "Why OA"],
            ].map(([href, label]) => (
              <a
                key={href}
                href={href}
                className="block rounded px-2 py-1 hover:bg-stone-300/40"
              >
                {label}
              </a>
            ))}
          </nav>
        </aside>

        <div className="flex-1 px-4 py-6 sm:px-6 sm:py-10 lg:px-10">
          {/* INSTALL — top card */}
          <section
            id="install"
            className="mx-auto mb-10 max-w-5xl scroll-mt-6"
          >
            <InstallCard />
          </section>

          {/* Hero — premium serif title */}
          <section
            id="overview"
            className="mx-auto mb-12 max-w-5xl scroll-mt-6"
          >
            <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div className="flex items-start gap-3">
                <div className="mt-1 h-11 w-11 shrink-0 overflow-hidden rounded-lg border border-stone-300/60 bg-white p-1 shadow-sm">
                  <Image
                    src={Logo}
                    alt="Open Agent Spec logo"
                    className="h-full w-full object-contain"
                    priority
                  />
                </div>
                <div>
                  <h1 className="font-serif text-4xl font-semibold leading-tight tracking-tight text-stone-900 sm:text-5xl md:text-6xl">
                    Open Agent Spec
                  </h1>
                  <p className="mt-3 max-w-xl font-serif text-lg text-stone-600 sm:text-xl md:text-2xl">
                    AI agents as code | define once in YAML, run anywhere.
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <span className="rounded-full border border-stone-400/50 bg-white/60 px-3 py-1 text-[10px] font-medium text-stone-700 backdrop-blur-sm">
                  Open source · MIT
                </span>
                <span className="rounded-full border border-stone-400/50 bg-white/60 px-3 py-1 text-[10px] font-medium text-stone-700 backdrop-blur-sm">
                  Spec v1.4.0
                </span>
              </div>
            </div>
            <p className="max-w-2xl text-sm leading-relaxed text-stone-700 sm:text-base">
              AI agents are fragmented across frameworks and runtimes. Open Agent
              Spec defines them once, declaratively, and runs them anywhere via the{" "}
              <code className="rounded bg-stone-200/80 px-1 py-0.5 text-stone-800">
                oa
              </code>{" "}
              CLI.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link
                href="/examples"
                className="rounded-lg bg-stone-900 px-5 py-2.5 text-sm font-medium text-stone-50 shadow-md transition hover:bg-stone-800"
              >
                Run a demo
              </Link>
              <Link
                href="https://github.com/prime-vector/open-agent-spec"
                className="rounded-lg border border-stone-400/60 bg-white/70 px-5 py-2.5 text-sm font-medium text-stone-800 backdrop-blur-sm transition hover:bg-white"
              >
                View on GitHub
              </Link>
              <Link
                href="https://github.com/prime-vector/open-agent-spec/blob/main/Website/content/defaultSpec.yaml"
                className="rounded-lg border border-stone-400/60 bg-white/70 px-5 py-2.5 text-sm font-medium text-stone-800 backdrop-blur-sm transition hover:bg-white"
              >
                Read the spec
              </Link>
            </div>

            {/* What's new in 1.4.0 */}
            <div id="whats-new" className="mt-8 scroll-mt-6">
              <div className="mb-4 flex flex-wrap items-baseline gap-2">
                <span className="inline-block rounded bg-stone-900 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-stone-100">
                  New in 1.4.0
                </span>
                <p className="text-sm font-semibold text-stone-900">
                  Tool use, spec composition, and a test harness | all from YAML.
                </p>
              </div>
              <div className="grid gap-4 sm:grid-cols-3">
                {/* Tool use */}
                <div className="overflow-hidden rounded-xl border border-stone-700 bg-stone-900">
                  <div className="border-b border-stone-700 px-4 py-2">
                    <span className="text-[11px] font-semibold text-stone-300">Tool use</span>
                    <p className="mt-0.5 text-[10px] text-stone-500">Native tools, MCP servers, or custom Python, declared in the spec.</p>
                  </div>
                  <div className="overflow-x-auto p-4 font-mono text-[10px] leading-relaxed text-stone-300">
                    <pre>{`tools:
  reader:
    type: native
    native: file.read

tasks:
  summarise:
    tools: [reader]
    prompts:
      system: "Summarise the file."
      user: "Read {path} and summarise."`}</pre>
                  </div>
                </div>

                {/* Spec composition */}
                <div className="overflow-hidden rounded-xl border border-stone-700 bg-stone-900">
                  <div className="border-b border-stone-700 px-4 py-2">
                    <span className="text-[11px] font-semibold text-stone-300">Spec composition</span>
                    <p className="mt-0.5 text-[10px] text-stone-500">Delegate tasks to shared specialist specs. Reuse without duplication.</p>
                  </div>
                  <div className="overflow-x-auto p-4 font-mono text-[10px] leading-relaxed text-stone-300">
                    <pre>{`tasks:
  summarise:
    description: Delegated summariser
    spec: ./shared/summariser.yaml
    task: summarise

  sentiment:
    description: Delegated sentiment
    spec: ./shared/sentiment.yaml
    task: analyse_sentiment`}</pre>
                  </div>
                </div>

                {/* Multi-step + test harness */}
                <div className="overflow-hidden rounded-xl border border-stone-700 bg-stone-900">
                  <div className="border-b border-stone-700 px-4 py-2">
                    <span className="text-[11px] font-semibold text-stone-300">Chaining + test harness</span>
                    <p className="mt-0.5 text-[10px] text-stone-500">Data dependencies with <code className="text-stone-400">depends_on</code>. Eval cases with <code className="text-stone-400">oa test</code>.</p>
                  </div>
                  <div className="overflow-x-auto p-4 font-mono text-[10px] leading-relaxed text-stone-300">
                    <pre>{`tasks:
  extract:
    description: Pull key facts
    output: { facts: string }

  summarise:
    depends_on: [extract]
    description: Summarise facts
    output: { summary: string }

# oa test --spec agent.yaml`}</pre>
                  </div>
                </div>
              </div>
              <div className="mt-2 text-[11px] text-stone-500">
                No orchestration engine. No framework. No SDK dependencies. Just YAML and the <code className="rounded bg-stone-200/60 px-1 text-stone-700">oa</code> CLI.
              </div>
            </div>

            {/* CLI demo — under hero CTAs */}
            <div className="mt-8 overflow-hidden rounded-xl border border-stone-300/50 bg-stone-900/5 shadow-sm">
              <Image
                src={CliGif}
                alt="Open Agent Spec CLI in action"
                unoptimized
                className="h-auto w-full object-cover object-top"
                sizes="(max-width: 1024px) 100vw, 1024px"
              />
            </div>
          </section>

          {/* Features — all capabilities */}
          <section
            id="features"
            className="mx-auto mb-8 max-w-5xl scroll-mt-6 rounded-xl border border-stone-300/50 bg-white/50 p-4 text-left text-sm leading-relaxed text-stone-700 shadow-sm backdrop-blur-sm sm:p-6"
          >
            <h2 className="mb-4 text-xs font-semibold uppercase tracking-wide text-stone-900">
              Features
            </h2>
            <div className="grid gap-4 sm:grid-cols-2">

              <div className="rounded-lg border border-stone-300/60 bg-stone-50/80 p-3">
                <div className="mb-1 text-xs font-semibold text-stone-900">Multi-step task chaining</div>
                <p className="text-xs text-stone-600">Declare data dependencies with <code className="rounded bg-stone-200 px-1">depends_on</code>. Output from one task flows automatically into the next. Linear, declarative, and strictly not an orchestration engine.</p>
              </div>

              <div className="rounded-lg border border-stone-300/60 bg-stone-50/80 p-3">
                <div className="mb-1 text-xs font-semibold text-stone-900">Tool use | native, MCP, custom</div>
                <p className="text-xs text-stone-600">Declare tools in the spec. Built-in tools (<code className="rounded bg-stone-200 px-1">file.read</code>, <code className="rounded bg-stone-200 px-1">http.get</code>, …), any MCP server via JSON-RPC, or your own Python class. No SDK required.</p>
              </div>

              <div className="rounded-lg border border-stone-300/60 bg-stone-50/80 p-3">
                <div className="mb-1 text-xs font-semibold text-stone-900">Spec composition</div>
                <p className="text-xs text-stone-600">A task can delegate its implementation to another spec with <code className="rounded bg-stone-200 px-1">spec:</code> + <code className="rounded bg-stone-200 px-1">task:</code>. Build coordinator specs that reuse shared specialists, zero duplication, full traceability.</p>
              </div>

              <div className="rounded-lg border border-stone-300/60 bg-stone-50/80 p-3">
                <div className="mb-1 text-xs font-semibold text-stone-900">Test harness</div>
                <p className="text-xs text-stone-600">Run eval cases against any spec with <code className="rounded bg-stone-200 px-1">oa test</code>. Assert on task output fields. Make your agents verifiable before you ship them.</p>
              </div>

              <div className="rounded-lg border border-stone-300/60 bg-stone-50/80 p-3">
                <div className="mb-1 text-xs font-semibold text-stone-900">Provider interface | no SDK lock-in</div>
                <p className="text-xs text-stone-600">All LLM calls go through a thin HTTP interface. OpenAI, Anthropic, Grok, xAI, Codex, local, or custom, swap engines with one line. No OpenAI SDK. No Anthropic SDK.</p>
              </div>

              <div className="rounded-lg border border-stone-300/60 bg-stone-50/80 p-3">
                <div className="mb-1 text-xs font-semibold text-stone-900">Behavioural contracts (optional)</div>
                <p className="text-xs text-stone-600">Attach output contracts to tasks with the <code className="rounded bg-stone-200 px-1">behavioural-contracts</code> library. Validate required fields, confidence scores, and custom rules, after parsing, before returning. Degrades gracefully when not installed.</p>
              </div>

            </div>
          </section>

          {/* Registry teaser */}
          <section
            id="registry"
            className="mx-auto mb-8 max-w-5xl scroll-mt-6 rounded-xl border border-stone-300/50 bg-white/50 p-4 text-left text-sm leading-relaxed text-stone-700 shadow-sm backdrop-blur-sm sm:p-6"
          >
            <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="text-xs font-semibold uppercase tracking-wide text-stone-900">
                Spec Registry
              </h2>
              <Link
                href="/registry"
                className="text-[11px] font-medium text-stone-500 transition hover:text-stone-900"
              >
                Browse all specs →
              </Link>
            </div>
            <p className="mb-4 max-w-2xl text-xs leading-relaxed text-stone-600">
              Pull shared, versioned agent specs directly from the registry using the{" "}
              <code className="rounded bg-stone-200/80 px-1 text-stone-800">oa://</code>{" "}
              shorthand. No copy-paste. Just delegate.
            </p>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {[
                {
                  id: "prime-vector/summariser",
                  name: "Summariser",
                  task: "summarise",
                  desc: "Summarise text and extract key points.",
                },
                {
                  id: "prime-vector/sentiment",
                  name: "Sentiment Analyser",
                  task: "analyse_sentiment",
                  desc: "Label tone as positive, negative, neutral, or mixed.",
                },
                {
                  id: "prime-vector/classifier",
                  name: "Text Classifier",
                  task: "classify",
                  desc: "Classify text into runtime-provided categories.",
                },
                {
                  id: "prime-vector/keyword-extractor",
                  name: "Keyword Extractor",
                  task: "extract_keywords",
                  desc: "Extract keywords and phrases ordered by relevance.",
                },
                {
                  id: "prime-vector/code-reviewer",
                  name: "Code Reviewer",
                  task: "review_code",
                  desc: "Review code for bugs, security issues, and improvements.",
                },
              ].map((spec) => (
                <div
                  key={spec.id}
                  className="rounded-lg border border-stone-300/60 bg-stone-50/80 p-3"
                >
                  <div className="mb-0.5 flex items-center justify-between gap-1">
                    <span className="text-xs font-semibold text-stone-900">{spec.name}</span>
                    <span className="rounded-full border border-stone-300/60 bg-white px-1.5 py-0.5 text-[9px] font-medium text-stone-500">
                      v1.0.0
                    </span>
                  </div>
                  <p className="mb-2 text-[11px] text-stone-500">{spec.desc}</p>
                  <code className="block rounded bg-stone-200/80 px-1.5 py-1 font-mono text-[10px] text-stone-700">
                    oa://{spec.id}
                  </code>
                </div>
              ))}
              {/* Browse CTA */}
              <div className="flex items-center justify-center rounded-lg border border-dashed border-stone-300 bg-transparent p-3">
                <Link
                  href="/registry"
                  className="text-center text-[11px] font-medium text-stone-500 transition hover:text-stone-800"
                >
                  Browse registry →<br />
                  <span className="text-[10px] font-normal text-stone-400">Publish your own spec</span>
                </Link>
              </div>
            </div>
          </section>

          {/* Content sections — readable on textured bg */}
          <section
            id="problem"
            className="mx-auto mb-8 max-w-4xl scroll-mt-6 rounded-xl border border-stone-300/50 bg-white/50 p-4 text-left text-sm leading-relaxed text-stone-700 shadow-sm backdrop-blur-sm sm:p-6"
          >
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-900">
              The Problem
            </h2>
            <p className="mb-2">
              Today, AI agents are often locked inside frameworks and hidden in SaaS
              dashboards. They&apos;re tightly coupled to specific runtimes, hard to
              version and review, and rarely portable across engines.
            </p>
            <p>
              | There is no standard way to define an agent declaratively. Open Agent
              Spec solves that.
            </p>
          </section>

          {/* How it works — after The Problem */}
          <section
            id="how-it-works"
            className="mx-auto mb-8 max-w-4xl scroll-mt-6 rounded-xl border border-stone-300/50 bg-white/50 p-4 text-left text-sm leading-relaxed text-stone-700 shadow-sm backdrop-blur-sm sm:p-6"
          >
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-900">
              How It Works
            </h2>
            <ol className="mb-3 list-decimal space-y-1 pl-5 text-sm">
              <li>Define your agent in YAML using Open Agent Spec.</li>
              <li>
                Run locally with{" "}
                <code className="rounded bg-stone-200 px-1">
                  oa run --spec .agents/agent.yaml
                </code>
                .
              </li>
              <li>Trigger from CI or GitHub Actions using the same spec.</li>
            </ol>
            <p>
              The OA CLI handles validation, prompt rendering, and engine selection,
              then normalises outputs to match your declared schema.
            </p>
          </section>

          {/* Two ways to use — under How it works */}
          <section
            id="modes"
            className="mx-auto mb-8 max-w-4xl scroll-mt-6 rounded-xl border border-stone-300/50 bg-white/50 p-4 text-left text-sm leading-relaxed text-stone-700 shadow-sm backdrop-blur-sm sm:p-6"
          >
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-900">
              There are Two 'Main' Ways to Use Open Agent Spec
            </h2>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-lg border border-stone-300/60 bg-stone-50/80 p-3">
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-stone-900">
                  Run Agents Directly
                </h3>
                <p className="mb-2 text-sm">
                  Execute specs without generating code. Ideal for CI and
                  repo-native execution.
                </p>
                <pre className="mt-2 overflow-x-auto rounded-lg border border-stone-300/60 bg-white px-3 py-2 font-mono text-[10px] text-stone-800">
                  {`oa run --spec .agents/review.yaml \\
  --task review \\
  --input pr.json`}
                </pre>
              </div>
              <div className="rounded-lg border border-stone-300/60 bg-stone-50/80 p-3">
                <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-stone-900">
                  Generate Agent Code
                </h3>
                <p className="mb-2 text-sm">
                  Scaffold working agents when you want code you own and can customise.
                </p>
                <pre className="mt-2 overflow-x-auto rounded-lg border border-stone-300/60 bg-white px-3 py-2 font-mono text-[10px] text-stone-800">
                  {`oa init --spec .agents/review.yaml \\
  --output ./agents/review`}
                </pre>
              </div>
            </div>
          </section>

          {/* Repo-native agents + CI — after Two ways */}
          <section
            id="repo-native"
            className="mx-auto mb-8 flex w-full max-w-5xl scroll-mt-6 flex-col gap-4 rounded-xl border border-stone-300/50 bg-white/50 p-4 text-left text-sm leading-relaxed text-stone-700 shadow-sm backdrop-blur-sm sm:flex-row sm:p-6"
          >
            <div className="w-full space-y-4">
              <div className="flex flex-col gap-4 sm:flex-row">
                <div className="flex-1 space-y-3">
                  <div className="font-semibold text-stone-900">
                    Repo-native agents
                  </div>
                  <p>
                    Agents belong in your repo. The OA CLI reads{" "}
                    <code className="rounded bg-stone-200 px-1">
                      .agents/*.yaml
                    </code>
                    , validates them, and uses engine adapters to run your
                    agents.
                  </p>
                </div>
                <div id="ci" className="flex-1 space-y-3 scroll-mt-6">
                  <div className="font-semibold text-stone-900">
                    CI & sub-agents
                  </div>
                  <p>
                    A single spec can fan out into multiple sub-agent processes.
                    The OA runtime routes each task to the right engine via
                    adapters.
                  </p>
                </div>
              </div>
              <div className="overflow-hidden rounded-lg border border-stone-300/60 bg-stone-50 p-3">
                <Image
                  src={DiagramOa}
                  alt="Diagram showing repo-native agents and CI sub-agent flow"
                  className="mx-auto h-auto w-full max-w-xs"
                  priority
                />
              </div>
              <div className="overflow-hidden rounded-xl border border-stone-300/50 bg-stone-900/5 shadow-sm">
                <Image
                  src={AacDemoGif}
                  alt="Agent-as-code demo: oa run with review agent and jq in the terminal"
                  unoptimized
                  className="h-auto w-full object-cover object-top"
                  sizes="(max-width: 1024px) 100vw, 1024px"
                />
              </div>
            </div>
          </section>

          <section
            id="not"
            className="mx-auto mb-8 max-w-4xl scroll-mt-6 rounded-xl border border-stone-300/50 bg-white/50 p-4 text-left text-sm leading-relaxed text-stone-700 shadow-sm backdrop-blur-sm sm:p-6"
          >
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-900">
              What Open Agent Spec Is Not
            </h2>
            <ul className="mb-3 list-disc space-y-1 pl-5">
              <li>Not a framework.</li>
              <li>Not an orchestration engine.</li>
              <li>Not tied to any model provider.</li>
            </ul>
            <p>
              It&apos;s a declarative standard | a thin, portable layer on top of any
              runtime.
            </p>
          </section>

          <section className="mx-auto mb-8 max-w-4xl rounded-xl border border-stone-300/50 bg-white/50 p-4 text-left text-sm leading-relaxed text-stone-700 shadow-sm backdrop-blur-sm sm:p-6">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-900">
              Agent as Infrastructure
            </h2>
            <p className="mb-2">
              Just as Terraform defines infrastructure, Open Agent Spec defines AI
              agents. Specs are version-controlled, reviewable, and portable.
            </p>
            <div className="flex flex-wrap gap-2 text-[10px]">
              <span className="rounded-full border border-stone-400/50 bg-stone-100/80 px-3 py-1">
                Open source · MIT
              </span>
              <span className="rounded-full border border-stone-400/50 bg-stone-100/80 px-3 py-1">
                Spec v1.4.0
              </span>
            </div>
          </section>

          <section
            id="why"
            className="mx-auto mb-10 max-w-4xl scroll-mt-6 rounded-xl border border-stone-300/50 bg-white/50 p-4 text-left text-sm leading-relaxed text-stone-700 shadow-sm backdrop-blur-sm sm:p-6"
          >
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-900">
              Why Open Agent Spec
            </div>
            <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3">
              {[
                ["Framework-agnostic", "Specs describe agents, not the web stack."],
                ["Engine-agnostic", "OpenAI, Anthropic, Grok, xAI, Codex, local, custom, swap with one line."],
                ["Repo-native", "Agents live in .agents/, versioned and reviewed like code."],
                ["CI-friendly", "One spec for local runs, GitHub Actions, and sub-agent pipelines."],
                ["Tool-native", "Declare file, HTTP, MCP, or custom tools directly in the spec."],
                ["Composable", "Tasks delegate to shared specialist specs. Reuse without duplication."],
              ].map(([title, body]) => (
                <div
                  key={title}
                  className="rounded-lg border border-stone-300/60 bg-stone-50/80 px-3 py-2"
                >
                  <div className="text-xs font-semibold uppercase tracking-wide text-stone-900">
                    {title}
                  </div>
                  <p className="mt-1 text-xs">{body}</p>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
