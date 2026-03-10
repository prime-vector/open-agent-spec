import Image from "next/image";
import Link from "next/link";
import Logo from "../OAS-Logo.webp";

export default function HomePage() {
  return (
    <main className="relative flex min-h-screen bg-surface overflow-hidden">
      {/* Subtle OA watermark in the background */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-24 -left-8 sm:-top-32 sm:-left-4"
      >
        <span className="select-none text-[260px] font-serif font-semibold tracking-[0.25em] text-white/5 sm:text-[320px] md:text-[380px] lg:text-[420px]">
          OA
        </span>
      </div>
      {/* Left navigation / sections */}
      <aside className="hidden w-52 border-r border-[var(--border)] bg-surface-muted px-4 py-6 text-xs text-[var(--text-muted)] sm:block">
        <div className="mb-4 text-[10px] font-semibold uppercase tracking-wide text-[var(--text)]">
          Open Agent Spec
        </div>
        <nav className="space-y-1">
          <a href="#overview" className="block rounded px-2 py-1 hover:bg-surface">
            Overview
          </a>
          <a href="#problem" className="block rounded px-2 py-1 hover:bg-surface">
            The Problem
          </a>
          <a href="#modes" className="block rounded px-2 py-1 hover:bg-surface">
            Two Modes
          </a>
          <a href="#how-it-works" className="block rounded px-2 py-1 hover:bg-surface">
            How It Works
          </a>
          <a href="#repo-native" className="block rounded px-2 py-1 hover:bg-surface">
            Repo-native Agents
          </a>
          <a href="#ci" className="block rounded px-2 py-1 hover:bg-surface">
            CI & Sub-agents
          </a>
          <a href="#not" className="block rounded px-2 py-1 hover:bg-surface">
            What OA Is Not
          </a>
          <a href="#why" className="block rounded px-2 py-1 hover:bg-surface">
            Why OA
          </a>
        </nav>
      </aside>

      {/* Main content */}
      <div className="flex-1 px-4 py-8 sm:px-8">
        <section
          id="overview"
          className="mx-auto mb-8 max-w-5xl rounded-xl bg-gradient-to-r from-[#0f172a] via-[#020617] to-[#111827] px-5 py-6 text-left text-sm text-slate-100 sm:px-8 sm:py-8"
        >
          <div className="flex flex-col items-start gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
              <div className="h-10 w-10 overflow-hidden rounded bg-white/5 p-1">
                <Image
                  src={Logo}
                  alt="Open Agent Spec logo"
                  className="h-full w-full object-contain"
                  priority
                />
              </div>
              <div>
                <h1
                  className="bg-gradient-to-r from-[#e2e8f0] via-[#cbd5f5] to-[#a5b4fc] bg-clip-text text-4xl font-semibold tracking-tight text-transparent sm:text-5xl md:text-6xl font-serif"
                >
                  OPEN AGENT SPEC
                </h1>
                <p
                  className="mt-3 bg-gradient-to-r from-[#cbd5f5] via-[#a5b4fc] to-[#7dd3fc] bg-clip-text text-lg font-medium text-transparent sm:text-2xl font-serif"
                >
                  AI Agents as Code
                </p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-[10px]">
              <span className="rounded-full border border-slate-600 bg-slate-900/60 px-3 py-1">
                Open source · MIT licensed
              </span>
              <span className="rounded-full border border-slate-600 bg-slate-900/60 px-3 py-1">
                Spec version v1.0.9
              </span>
            </div>
          </div>
          <div className="mt-6 max-w-3xl text-sm text-slate-100">
            <p className="mb-2">
              AI agents are fragmented across frameworks and runtimes.
            </p>
            <p className="text-slate-200">
              Open Agent Spec defines them once, and runs them anywhere.
            </p>
          </div>
          <div className="mt-5 flex flex-wrap items-center gap-3">
            <Link
              href="/examples"
              className="rounded bg-[var(--accent)] px-5 py-2 text-xs font-medium text-white shadow-sm transition hover:opacity-90"
            >
              Run a demo
            </Link>
            <Link
              href="https://github.com/prime-vector/open-agent-spec"
              className="rounded border border-slate-600 bg-slate-900/40 px-5 py-2 text-xs font-medium text-slate-100 transition hover:border-[var(--accent)] hover:text-white"
            >
              View on GitHub
            </Link>
            <Link
              href="https://github.com/prime-vector/open-agent-spec/blob/main/Website/content/defaultSpec.yaml"
              className="rounded border border-slate-600 bg-slate-900/40 px-5 py-2 text-xs font-medium text-slate-100 transition hover:border-[var(--accent)] hover:text-white"
            >
              Read the spec
            </Link>
          </div>
        </section>

        <section
          id="problem"
          className="mx-auto mb-8 max-w-4xl text-left text-[13px] leading-relaxed text-[var(--text-muted)] sm:text-sm"
        >
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--text)]">
            The Problem
          </h2>
          <p className="mb-2">
            Today, AI agents are often locked inside frameworks and hidden in SaaS
            dashboards. They&apos;re tightly coupled to specific runtimes, hard to
            version and review, and rarely portable across engines.
          </p>
          <ul className="mb-2 list-disc pl-5 text-xs sm:text-[13px]">
            <li>Locked inside frameworks</li>
            <li>Coupled to runtimes</li>
            <li>Hard to version and review</li>
            <li>Not portable across engines</li>
          </ul>
          <p>
            There is no standard way to define an agent declaratively. Open Agent
            Spec solves that.
          </p>
        </section>

        <section
          id="repo-native"
          className="mx-auto mt-4 flex w-full max-w-5xl flex-col gap-4 rounded-lg border border-[var(--border)] bg-surface-muted px-4 py-4 text-left text-[13px] leading-relaxed text-[var(--text-muted)] sm:flex-row sm:px-6 sm:py-5 sm:text-sm"
        >
          {/* Left: repo + flow diagram */}
          <div className="flex-1 space-y-3">
            <div className="font-semibold text-[var(--text)]">Repo-native agents</div>
            <div className="flex gap-4">
              <div className="rounded border border-[var(--border)] bg-surface px-3 py-2 font-mono text-[11px] leading-relaxed">
                <div>my-service-repo/</div>
                <div className="pl-4">.agents/</div>
                <div className="pl-8">review.yaml</div>
                <div className="pl-8">deploy.yaml</div>
              </div>
            </div>
            <p>
              Agents belong in your repo, not hidden in SaaS dashboards. The OA CLI
              reads <code>.agents/*.yaml</code>, validates them, and uses engine
              adapters to run your agents.
            </p>
            <div className="mt-2 rounded border border-dashed border-[var(--border)] bg-surface px-3 py-2 font-mono text-[10px] leading-relaxed">
              <div>.agents/review.yaml</div>
              <div className="pl-2">↓</div>
              <div>oas run --spec .agents/review.yaml</div>
              <div className="pl-2">↓</div>
              <div>Engine adapter (OpenAI / Claude / Codex / custom)</div>
              <div className="pl-2">↓</div>
              <div>Structured output</div>
            </div>
          </div>

          {/* Right: GitHub Actions + sub-agents */}
          <div id="ci" className="flex-1 space-y-3">
            <div className="font-semibold text-[var(--text)]">CI & sub-agents</div>
            <div className="rounded border border-[var(--border)] bg-surface px-3 py-2 font-mono text-[10px]">
              <div>- uses: open-agent-spec/run-agent@v1</div>
              <div className="pl-4">with:</div>
              <div className="pl-8">spec-path: .agents/review.yaml</div>
              <div className="pl-8">task: review</div>
              <div className="pl-8">input: pr.json</div>
            </div>
            <p>
              A single spec can fan out into multiple sub-agent processes: review,
              deploy, notify. The OA runtime routes each task to the right engine via
              adapters, engine-agnostic by design.
            </p>
          </div>
        </section>

        <section
          id="how-it-works"
          className="mx-auto mt-6 max-w-4xl text-left text-[13px] leading-relaxed text-[var(--text-muted)] sm:text-sm"
        >
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--text)]">
            How It Works
          </h2>
          <ol className="mb-3 list-decimal pl-5 space-y-1 text-xs sm:text-[13px]">
            <li>Define your agent in YAML using Open Agent Spec.</li>
            <li>
              Store it in <code>.agents/</code> in your repo.
            </li>
            <li>
              Run it locally with <code>oas run --spec .agents/agent.yaml</code>.
            </li>
            <li>Trigger it from CI or GitHub Actions using the same spec.</li>
          </ol>
          <p>
            The OA CLI handles validation, prompt rendering, and engine selection,
            then normalises outputs to match your declared schema.
          </p>
        </section>

        <section
          id="modes"
          className="mx-auto mt-6 max-w-4xl text-left text-[13px] leading-relaxed text-[var(--text-muted)] sm:text-sm"
        >
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--text)]">
            Two Ways to Use Open Agent Spec
          </h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="rounded border border-[var(--border)] bg-surface-muted px-3 py-3">
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-[var(--text)]">
                Run Agents Directly
              </h3>
              <p className="mb-2 text-[13px]">
                Execute specs without generating code. Ideal for CI, automation, and
                repo-native execution.
              </p>
              <ul className="mb-2 list-disc pl-5 text-xs">
                <li>Runtime-driven, infra-style, declarative execution.</li>
                <li>Engine-agnostic routing via adapters.</li>
              </ul>
              <pre className="mt-2 rounded border border-[var(--border)] bg-surface px-3 py-2 font-mono text-[10px]">
oas run --spec .agents/review.yaml \
  --task review \
  --input pr.json
              </pre>
              <p className="mt-2 text-[11px]">
                Spec → OA CLI → Engine Adapter → Structured output. No code generation
                required.
              </p>
            </div>

            <div className="rounded border border-[var(--border)] bg-surface-muted px-3 py-3">
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-[var(--text)]">
                Generate Agent Code
              </h3>
              <p className="mb-2 text-[13px]">
                Scaffold working agents from specs when you want code you own and can
                customise.
              </p>
              <ul className="mb-2 list-disc pl-5 text-xs">
                <li>Framework integration and app embedding.</li>
                <li>Full customization and traditional developer workflow.</li>
              </ul>
              <pre className="mt-2 rounded border border-[var(--border)] bg-surface px-3 py-2 font-mono text-[10px]">
oas init --spec .agents/review.yaml --output ./agents/review
              </pre>
              <p className="mt-2 text-[11px]">
                Spec → Code → You own it. Use the generated agent in any framework or
                runtime.
              </p>
            </div>
          </div>
        </section>

        <section
          id="not"
          className="mx-auto mt-6 max-w-4xl text-left text-[13px] leading-relaxed text-[var(--text-muted)] sm:text-sm"
        >
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--text)]">
            What Open Agent Spec Is Not
          </h2>
          <ul className="mb-3 list-disc pl-5 space-y-1">
            <li>Not a framework.</li>
            <li>Not an orchestration engine.</li>
            <li>Not tied to any model provider.</li>
          </ul>
          <p>
            It&apos;s a declarative standard for describing agents, a thin, portable
            layer that can sit on top of any runtime.
          </p>
        </section>

        <section className="mx-auto mt-6 max-w-4xl text-left text-[13px] leading-relaxed text-[var(--text-muted)] sm:text-sm">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--text)]">
            Agent as Infrastructure
          </h2>
          <p className="mb-2">
            Just as Terraform defines infrastructure, Open Agent Spec defines AI
            agents. Specs are version-controlled, reviewable, and portable across
            runtimes.
          </p>
          <div className="flex flex-wrap items-center gap-3 text-[10px]">
            <span className="rounded-full border border-[var(--border)] bg-surface-muted px-3 py-1">
              Open source · MIT licensed
            </span>
            <span className="rounded-full border border-[var(--border)] bg-surface-muted px-3 py-1">
              Current spec: v1.0.9
            </span>
            <span className="rounded-full border border-[var(--border)] bg-surface-muted px-3 py-1">
              Used in internal automation pipelines
            </span>
          </div>
        </section>

        <section
          id="why"
          className="mx-auto mt-8 mb-6 max-w-4xl text-left text-[13px] leading-relaxed text-[var(--text-muted)] sm:text-sm"
        >
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--text)]">
            Why Open Agent Spec
          </div>
          <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4">
            <div className="rounded border border-[var(--border)] bg-surface-muted px-3 py-2">
              <div className="text-xs font-semibold uppercase tracking-wide">
                Framework-agnostic
              </div>
              <p className="mt-1 text-xs">
                Specs describe agents, not the web stack or runtime framework.
              </p>
            </div>
            <div className="rounded border border-[var(--border)] bg-surface-muted px-3 py-2">
              <div className="text-xs font-semibold uppercase tracking-wide">
                Engine-agnostic
              </div>
              <p className="mt-1 text-xs">
                Engine-agnostic by design. Integrates with OpenAI, Claude, Codex and
                custom runtimes via adapters.
              </p>
            </div>
            <div className="rounded border border-[var(--border)] bg-surface-muted px-3 py-2">
              <div className="text-xs font-semibold uppercase tracking-wide">
                Repo-native
              </div>
              <p className="mt-1 text-xs">
                Agents live in <code>.agents/</code>, versioned and reviewed like
                code.
              </p>
            </div>
            <div className="rounded border border-[var(--border)] bg-surface-muted px-3 py-2">
              <div className="text-xs font-semibold uppercase tracking-wide">
                CI-friendly
              </div>
              <p className="mt-1 text-xs">
                One spec for local runs, CI pipelines, and reusable GitHub Actions.
              </p>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}


