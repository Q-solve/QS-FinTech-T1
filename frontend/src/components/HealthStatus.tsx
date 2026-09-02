import { CircleCheck, LoaderCircle, TriangleAlert } from 'lucide-react'
import type { ApiHealthState } from '../types/api'

interface HealthStatusProps {
  health: ApiHealthState
}

export function HealthStatus({ health }: HealthStatusProps) {
  if (health.state === 'loading') {
    return (
      <div
        className="inline-flex items-center gap-2 rounded-full border border-sky-300/20 bg-sky-300/10 px-3 py-1.5 text-sm text-sky-100"
        role="status"
        aria-live="polite"
      >
        <LoaderCircle className="size-4 animate-spin" aria-hidden="true" />
        Checking API
      </div>
    )
  }

  if (health.state === 'online') {
    return (
      <div
        className="inline-flex items-center gap-2 rounded-full border border-emerald-300/20 bg-emerald-300/10 px-3 py-1.5 text-sm text-emerald-100"
        role="status"
        aria-live="polite"
      >
        <CircleCheck className="size-4" aria-hidden="true" />
        API online · v{health.data.version}
      </div>
    )
  }

  return (
    <div
      className="inline-flex items-center gap-2 rounded-full border border-amber-300/20 bg-amber-300/10 px-3 py-1.5 text-sm text-amber-100"
      role="status"
      aria-live="polite"
      title={health.message}
    >
      <TriangleAlert className="size-4" aria-hidden="true" />
      API unavailable
    </div>
  )
}
