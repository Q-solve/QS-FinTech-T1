import { useEffect, useMemo, useState } from 'react'
import {
  ArrowDown,
  Atom,
  BadgeCheck,
  Banknote,
  BarChart3,
  BookOpen,
  Check,
  ChevronRight,
  CircleAlert,
  Clock3,
  Database,
  Info,
  Layers3,
  LoaderCircle,
  MapPin,
  Route,
  Scale,
  ShieldCheck,
  Sparkles,
  Zap,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AppShell } from '../components/AppShell'
import { HealthStatus } from '../components/HealthStatus'
import { fetchApiHealth, fetchShowcase } from '../services/api'
import type {
  ApiHealthState,
  ShowcaseExperiment,
  ShowcaseResponse,
  ShowcaseState,
} from '../types/api'

const percent = (value: number | null) =>
  value === null ? '—' : `${(value * 100).toFixed(1)}%`
const fixed = (value: number | null, digits = 3) =>
  value === null ? '—' : value.toFixed(digits)

function MetricCard({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{label}</p>
      <p className="mt-2 text-3xl font-semibold tracking-tight text-slate-950">{value}</p>
      <p className="mt-1 text-sm text-slate-500">{note}</p>
    </div>
  )
}

function ExperimentCard({ experiment }: { experiment: ShowcaseExperiment }) {
  const isXY = experiment.mixer === 'constraint_preserving_xy'
  return (
    <article
      className={`rounded-3xl border p-6 ${isXY ? 'border-teal-200 bg-teal-50/70' : 'border-slate-200 bg-white'}`}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
            {isXY ? 'Constraint preserving' : 'Penalty based'}
          </p>
          <h3 className="mt-2 text-xl font-semibold text-slate-950">{experiment.label}</h3>
        </div>
        <span
          className={`grid size-10 place-items-center rounded-xl ${isXY ? 'bg-teal-600 text-white' : 'bg-slate-100 text-slate-600'}`}
        >
          {isXY ? <ShieldCheck className="size-5" /> : <Atom className="size-5" />}
        </span>
      </div>
      <div className="mt-6 grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-slate-500">Feasible probability</p>
          <p className="mt-1 text-2xl font-semibold text-slate-950">
            {percent(experiment.feasible_probability.mean)}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Exact recovery</p>
          <p className="mt-1 text-2xl font-semibold text-slate-950">
            {experiment.exact_recovery_count}/{experiment.run_count}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Mean leakage</p>
          <p className="mt-1 font-semibold text-slate-800">
            {percent(experiment.leakage_probability.mean)}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500">Transpiled depth</p>
          <p className="mt-1 font-semibold text-slate-800">
            {fixed(experiment.transpiled_depth.mean, 0)}
          </p>
        </div>
      </div>
    </article>
  )
}

function Hero({ data }: { data: ShowcaseResponse }) {
  return (
    <section className="overflow-hidden rounded-[2rem] border border-slate-800 bg-slate-950 px-6 py-8 text-white shadow-2xl shadow-slate-950/20 sm:px-10 sm:py-12">
      <div className="grid gap-10 lg:grid-cols-[1.25fr_0.75fr] lg:items-center">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-teal-300/25 bg-teal-300/10 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] text-teal-200">
            <BadgeCheck className="size-3.5" /> Validated simulator study
          </div>
          <h1 className="mt-6 max-w-4xl text-balance text-4xl font-semibold tracking-[-0.045em] sm:text-6xl">
            Find a remittance service—and show exactly how the answer was tested.
          </h1>
          <p className="mt-6 max-w-3xl text-pretty text-lg leading-8 text-slate-300">
            A transparent comparison of 14 Kenya→Tanzania provider services using an exact
            classical baseline, standard QAOA, and a constraint-preserving XY variant.
          </p>
          <div className="mt-8 flex flex-wrap gap-3 text-sm">
            <span className="rounded-full bg-white/8 px-4 py-2">{data.scenario.period}</span>
            <span className="rounded-full bg-white/8 px-4 py-2">
              {data.scenario.benchmark_currency} {data.scenario.benchmark_amount.toLocaleString()}
            </span>
            <span className="rounded-full bg-white/8 px-4 py-2">
              {data.scenario.alternative_count} alternatives
            </span>
          </div>
        </div>
        <div className="rounded-3xl border border-white/10 bg-white/6 p-6 backdrop-blur">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
            Demonstration corridor
          </p>
          <div className="mt-5 flex items-center gap-4">
            <div className="grid size-12 place-items-center rounded-full bg-sky-400/15 font-semibold text-sky-200">KE</div>
            <div className="h-px flex-1 bg-gradient-to-r from-sky-400 to-teal-400" />
            <Route className="size-5 text-teal-300" />
            <div className="h-px flex-1 bg-gradient-to-r from-teal-400 to-emerald-400" />
            <div className="grid size-12 place-items-center rounded-full bg-emerald-400/15 font-semibold text-emerald-200">TZ</div>
          </div>
          <div className="mt-4 flex justify-between text-sm text-slate-300">
            <span>{data.scenario.source}</span><span>{data.scenario.destination}</span>
          </div>
          <div className="mt-6 flex gap-3 rounded-2xl bg-amber-300/10 p-4 text-sm leading-6 text-amber-100">
            <Info className="mt-0.5 size-4 shrink-0" />
            Zanzibar is represented by Tanzania because the workbook has no separate Zanzibar destination.
          </div>
        </div>
      </div>
      <a href="#recommendation" className="mt-8 inline-flex items-center gap-2 text-sm text-slate-400 hover:text-white">
        Explore the evidence <ArrowDown className="size-4" />
      </a>
    </section>
  )
}

function Showcase({ data }: { data: ShowcaseResponse }) {
  const [depth, setDepth] = useState<1 | 2>(1)
  const experiments = useMemo(
    () => data.experiments.filter((item) => item.depth === depth),
    [data.experiments, depth],
  )
  const chartData = experiments.map((item) => ({
    name: item.mixer === 'standard_x' ? 'Standard X' : 'XY + W',
    feasibility: (item.feasible_probability.mean ?? 0) * 100,
    optimum: (item.optimal_probability.mean ?? 0) * 100,
  }))
  const selected = data.recommendation
  const workflowIcons = [Database, Scale, Banknote, Atom, BarChart3]

  return (
    <>
      <Hero data={data} />

      <section id="recommendation" className="mt-10 grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <article className="rounded-3xl border border-emerald-200 bg-emerald-50 p-7">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">Exact classical recommendation</p>
              <h2 className="mt-2 text-3xl font-semibold tracking-tight text-slate-950">{selected.provider}</h2>
            </div>
            <span className="grid size-12 place-items-center rounded-2xl bg-emerald-600 text-white"><Check className="size-6" /></span>
          </div>
          <div className="mt-7 grid grid-cols-2 gap-4 text-sm">
            <div className="rounded-2xl bg-white/70 p-4">
              <Banknote className="size-4 text-emerald-700" />
              <p className="mt-2 text-slate-500">Total cost</p>
              <p className="text-xl font-semibold text-slate-950">{selected.total_cost_percentage.toFixed(2)}%</p>
            </div>
            <div className="rounded-2xl bg-white/70 p-4">
              <Clock3 className="size-4 text-emerald-700" />
              <p className="mt-2 text-slate-500">Speed</p>
              <p className="text-base font-semibold text-slate-950">{selected.speed}</p>
            </div>
          </div>
          <p className="mt-5 text-sm leading-6 text-slate-600">
            {selected.payment_instrument} payment · {selected.pickup_method} pickup · weighted score {selected.weighted_score.toFixed(6)}
          </p>
        </article>

        <article className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
          <div className="flex items-center gap-3">
            <Scale className="size-5 text-sky-700" />
            <div><p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Fixed objective</p><h2 className="text-xl font-semibold text-slate-950">What “best” means here</h2></div>
          </div>
          <div className="mt-7 space-y-5">
            {[
              { label: 'Fee', value: data.weights.fee, color: 'bg-sky-500' },
              { label: 'FX margin', value: data.weights.fx_margin, color: 'bg-teal-500' },
              { label: 'Speed', value: data.weights.speed, color: 'bg-violet-500' },
            ].map((item) => (
              <div key={item.label}>
                <div className="mb-2 flex justify-between text-sm"><span className="font-medium text-slate-700">{item.label}</span><span className="font-semibold text-slate-950">{Math.round(item.value * 100)}%</span></div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-100"><div className={`h-full rounded-full ${item.color}`} style={{ width: `${item.value * 100}%` }} /></div>
              </div>
            ))}
          </div>
          <p className="mt-6 rounded-2xl bg-slate-50 p-4 text-sm leading-6 text-slate-600">
            Fee and FX margin are optimized separately. Total cost is shown for explanation, not added again—avoiding double counting.
          </p>
        </article>
      </section>

      <section className="mt-12">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div><p className="text-xs font-semibold uppercase tracking-[0.18em] text-sky-700">Measured comparison</p><h2 className="mt-2 text-3xl font-semibold tracking-tight text-slate-950">Does the mixer keep answers feasible?</h2><p className="mt-2 max-w-2xl text-slate-600">{data.protocol.matched_seeds_by_depth[String(depth)]?.length ?? 0} matched seeds, {data.protocol.shots.toLocaleString()} shots, and the identical cost Hamiltonian.</p></div>
          <div className="inline-flex rounded-xl border border-slate-200 bg-white p-1" aria-label="QAOA depth">
            {([1, 2] as const).map((value) => (
              <button key={value} type="button" onClick={() => setDepth(value)} className={`rounded-lg px-4 py-2 text-sm font-semibold transition ${depth === value ? 'bg-slate-950 text-white' : 'text-slate-600 hover:bg-slate-100'}`}>Depth p={value}</button>
            ))}
          </div>
        </div>
        <div className="grid gap-6 lg:grid-cols-2">{experiments.map((item) => <ExperimentCard key={item.key} experiment={item} />)}</div>
        <div className="mt-6 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <h3 className="font-semibold text-slate-950">Probability comparison (%)</h3>
          <div className="mt-4 h-72" aria-label="QAOA probability comparison chart">
            <ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} margin={{ left: -12, right: 12 }}><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" /><XAxis dataKey="name" tickLine={false} axisLine={false} /><YAxis domain={[0, 100]} tickLine={false} axisLine={false} unit="%" /><Tooltip formatter={(value) => `${Number(value).toFixed(2)}%`} /><Legend /><Bar dataKey="feasibility" name="Feasible samples" fill="#0d9488" radius={[6, 6, 0, 0]} /><Bar dataKey="optimum" name="Exact optimum" fill="#0284c7" radius={[6, 6, 0, 0]} /></BarChart></ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="XY feasibility" value="100%" note="At both tested depths" />
        <MetricCard label="XY leakage" value="0%" note="Ideal noiseless simulator" />
        <MetricCard label="Matched runs" value={String(data.experiments.reduce((total, item) => total + item.run_count, 0))} note={`${data.experiments.length} experiments from stored artifacts`} />
        <MetricCard label="Mixer designs" value="2" note="Same validated cost target" />
      </section>

      <section className="mt-16">
        <div className="flex items-center gap-3"><Layers3 className="size-6 text-violet-700" /><div><p className="text-xs font-semibold uppercase tracking-[0.18em] text-violet-700">How it works</p><h2 className="text-3xl font-semibold tracking-tight text-slate-950">From workbook to measured result</h2></div></div>
        <div className="mt-8 grid gap-4 lg:grid-cols-5">
          {data.workflow.map((step, index) => {
            const Icon = workflowIcons[index]
            return <article key={step} className="relative rounded-2xl border border-slate-200 bg-white p-5"><div className="flex items-center justify-between"><span className="grid size-9 place-items-center rounded-xl bg-violet-50 text-violet-700"><Icon className="size-4" /></span><span className="text-xs font-semibold text-slate-400">0{index + 1}</span></div><p className="mt-4 text-sm leading-6 text-slate-600">{step}</p>{index < data.workflow.length - 1 && <ChevronRight className="absolute -right-3 top-1/2 z-10 hidden size-5 text-slate-300 lg:block" />}</article>
          })}
        </div>
      </section>

      <section className="mt-16">
        <div className="mb-6 flex items-center gap-3"><BookOpen className="size-5 text-sky-700" /><h2 className="text-2xl font-semibold text-slate-950">All 14 eligible services</h2></div>
        <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm"><div className="overflow-x-auto"><table className="w-full min-w-[860px] text-left text-sm"><thead className="bg-slate-50 text-xs uppercase tracking-[0.12em] text-slate-500"><tr><th className="px-5 py-4">Rank</th><th className="px-5 py-4">Provider / service</th><th className="px-5 py-4">Speed</th><th className="px-5 py-4">Fee</th><th className="px-5 py-4">FX margin</th><th className="px-5 py-4">Total cost</th><th className="px-5 py-4">Score</th></tr></thead><tbody className="divide-y divide-slate-100">
          {data.alternatives.map((item) => <tr key={item.alternative_id} className={item.selected ? 'bg-emerald-50/70' : 'hover:bg-slate-50'}><td className="px-5 py-4 font-semibold text-slate-700">#{item.rank}</td><td className="px-5 py-4"><p className="font-semibold text-slate-950">{item.provider}</p><p className="mt-1 text-xs text-slate-500">{item.payment_instrument} · {item.pickup_method}</p></td><td className="px-5 py-4 text-slate-600">{item.speed}</td><td className="px-5 py-4 tabular-nums text-slate-600">{item.fee_percentage.toFixed(2)}%</td><td className="px-5 py-4 tabular-nums text-slate-600">{item.fx_margin.toFixed(2)}%</td><td className="px-5 py-4 tabular-nums text-slate-600">{item.total_cost_percentage.toFixed(2)}%</td><td className="px-5 py-4 tabular-nums font-medium text-slate-800">{item.weighted_score.toFixed(4)}</td></tr>)}
        </tbody></table></div></div>
      </section>

      <section className="mt-16 grid gap-6 lg:grid-cols-2">
        <article className="rounded-3xl border border-sky-200 bg-sky-50 p-7"><div className="flex items-center gap-3"><Sparkles className="size-5 text-sky-700" /><h2 className="text-xl font-semibold text-slate-950">Why XY changes the result</h2></div><p className="mt-4 text-sm leading-7 text-slate-700">Standard QAOA begins across every binary string and flips individual bits, so it can spend most probability on states selecting zero or several services. The W state starts only on one-hot choices. Ring-XY gates move that single excitation between providers without creating or destroying it, keeping every ideal sample feasible.</p><div className="mt-5 flex items-center gap-3 rounded-2xl bg-white/70 p-4 text-sm text-slate-600"><Zap className="size-5 shrink-0 text-sky-700" />Same cost Hamiltonian and penalty; only initialization and mixer change.</div></article>
        <article className="rounded-3xl border border-amber-200 bg-amber-50 p-7"><div className="flex items-center gap-3"><CircleAlert className="size-5 text-amber-700" /><h2 className="text-xl font-semibold text-slate-950">What this does not prove</h2></div><ul className="mt-4 space-y-3">{data.limitations.map((item) => <li key={item} className="flex gap-3 text-sm leading-6 text-slate-700"><span className="mt-2 size-1.5 shrink-0 rounded-full bg-amber-600" />{item}</li>)}</ul></article>
      </section>

      <footer className="mt-16 border-t border-slate-200 py-8 text-sm text-slate-500"><div className="flex flex-wrap items-center justify-between gap-4"><p>Remit-Q · Evidence-backed research prototype</p><p className="flex items-center gap-2"><MapPin className="size-4" />{data.provenance.period} · {data.scenario.corridor} · comparison schema {data.provenance.comparison_schema}</p></div></footer>
    </>
  )
}

export function HomePage() {
  const [health, setHealth] = useState<ApiHealthState>({ state: 'loading' })
  const [showcase, setShowcase] = useState<ShowcaseState>({ state: 'loading' })

  useEffect(() => {
    const controller = new AbortController()
    fetchApiHealth(controller.signal)
      .then((data) => setHealth({ state: 'online', data }))
      .catch((error: unknown) => {
        if (!controller.signal.aborted)
          setHealth({ state: 'offline', message: error instanceof Error ? error.message : 'Unknown API error' })
      })
    fetchShowcase(controller.signal)
      .then((data) => setShowcase({ state: 'ready', data }))
      .catch((error: unknown) => {
        if (!controller.signal.aborted)
          setShowcase({ state: 'error', message: error instanceof Error ? error.message : 'Unknown API error' })
      })
    return () => controller.abort()
  }, [])

  return (
    <AppShell status={<HealthStatus health={health} />}>
      {showcase.state === 'loading' && <div className="grid min-h-[60vh] place-items-center text-slate-500" role="status"><div className="text-center"><LoaderCircle className="mx-auto size-8 animate-spin text-sky-600" /><p className="mt-4">Loading validated experiment evidence…</p></div></div>}
      {showcase.state === 'error' && <div className="mx-auto max-w-xl rounded-3xl border border-amber-200 bg-amber-50 p-8 text-center"><CircleAlert className="mx-auto size-8 text-amber-700" /><h1 className="mt-4 text-2xl font-semibold text-slate-950">Showcase data unavailable</h1><p className="mt-2 text-slate-600">{showcase.message}</p><p className="mt-4 text-sm text-slate-500">Start the FastAPI service and refresh this page.</p></div>}
      {showcase.state === 'ready' && <Showcase data={showcase.data} />}
    </AppShell>
  )
}
