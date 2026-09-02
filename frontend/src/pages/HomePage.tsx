import { useEffect, useState } from 'react'
import { Database, FlaskConical, Route, ShieldCheck } from 'lucide-react'
import { AppShell } from '../components/AppShell'
import { HealthStatus } from '../components/HealthStatus'
import { fetchApiHealth } from '../services/api'
import type { ApiHealthState } from '../types/api'

const foundations = [
  {
    icon: Database,
    title: 'Audited data',
    description: 'The application reads the compact cleaned dataset, never the malformed workbook.',
  },
  {
    icon: ShieldCheck,
    title: 'Evidence first',
    description: 'Future results will retain their corridor, period, benchmark, and method.',
  },
  {
    icon: FlaskConical,
    title: 'Neutral comparison',
    description: 'Classical and quantum approaches will be measured on the same instances.',
  },
]

export function HomePage() {
  const [health, setHealth] = useState<ApiHealthState>({ state: 'loading' })

  useEffect(() => {
    const controller = new AbortController()

    fetchApiHealth(controller.signal)
      .then((data) => setHealth({ state: 'online', data }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return
        const message = error instanceof Error ? error.message : 'Unknown API error'
        setHealth({ state: 'offline', message })
      })

    return () => controller.abort()
  }, [])

  return (
    <AppShell status={<HealthStatus health={health} />}>
      <section className="max-w-3xl">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium uppercase tracking-[0.18em] text-sky-200">
          <Route className="size-3.5" aria-hidden="true" />
          Foundation phase
        </div>
        <h1 className="text-balance text-4xl font-semibold tracking-[-0.04em] text-white sm:text-6xl">
          Better remittance decisions, tested with classical and quantum methods.
        </h1>
        <p className="mt-6 max-w-2xl text-pretty text-lg leading-8 text-slate-300">
          Remit-Q is an evidence-backed research prototype for comparing eligible
          cross-border provider services by cost and speed. Recommendations and
          optimization experiments are intentionally deferred until the shared
          formulation is validated.
        </p>
      </section>

      <section className="mt-14 grid gap-4 md:grid-cols-3" aria-label="Project foundations">
        {foundations.map(({ icon: Icon, title, description }) => (
          <article
            key={title}
            className="rounded-2xl border border-white/10 bg-slate-900/55 p-6 shadow-2xl shadow-slate-950/20 backdrop-blur"
          >
            <span className="grid size-10 place-items-center rounded-xl bg-white/5 text-teal-200">
              <Icon className="size-5" aria-hidden="true" />
            </span>
            <h2 className="mt-5 text-lg font-semibold text-white">{title}</h2>
            <p className="mt-2 text-sm leading-6 text-slate-400">{description}</p>
          </article>
        ))}
      </section>

      <footer className="mt-16 border-t border-white/10 pt-6 text-sm text-slate-500">
        Initial application scaffold · No optimization or recommendation results yet.
      </footer>
    </AppShell>
  )
}
