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
                  Spec v1.2.5
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
                Spec v1.2.5
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
            <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4">
              {[
                ["Framework-agnostic", "Specs describe agents, not the web stack."],
                ["Engine-agnostic", "OpenAI, Claude, Codex, custom | via adapters."],
                [
                  "Repo-native",
                  "Agents live in .agents/, versioned like code.",
                ],
                ["CI-friendly", "One spec for local runs and GitHub Actions."],
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
