export interface HealthResponse {
  status: 'ok'
  service: string
  version: string
}

export type ApiHealthState =
  | { state: 'loading' }
  | { state: 'online'; data: HealthResponse }
  | { state: 'offline'; message: string }

export interface DistributionSummary {
  mean: number | null
  standard_deviation: number | null
}

export interface ShowcaseAlternative {
  rank: number
  alternative_id: string
  provider: string
  provider_type: string
  payment_instrument: string
  pickup_method: string
  speed: string
  fee_percentage: number
  fx_margin: number
  total_cost_percentage: number
  weighted_score: number
  selected: boolean
}

export interface ShowcaseExperiment {
  key: string
  label: string
  mixer: 'standard_x' | 'constraint_preserving_xy'
  depth: number
  run_count: number
  feasible_probability: DistributionSummary
  leakage_probability: DistributionSummary
  optimal_probability: DistributionSummary
  exact_recovery_count: number
  exact_recovery_rate: number
  modal_feasibility_rate: number
  best_feasible_gap: DistributionSummary
  transpiled_depth: DistributionSummary
  runtime_seconds: DistributionSummary
}

export interface ShowcaseResponse {
  title: string
  scenario: {
    period: string
    corridor: string
    source: string
    destination: string
    story_destination: string
    benchmark: string
    benchmark_amount: number
    benchmark_currency: string
    alternative_count: number
  }
  weights: { fee: number; fx_margin: number; speed: number }
  recommendation: ShowcaseAlternative
  alternatives: ShowcaseAlternative[]
  experiments: ShowcaseExperiment[]
  protocol: {
    shots: number
    optimizer: string
    maximum_iterations: number
    simulator: string
    matched_seeds_by_depth: Record<string, number[]>
  }
  workflow: string[]
  limitations: string[]
  provenance: Record<string, string>
}

export type ShowcaseState =
  | { state: 'loading' }
  | { state: 'ready'; data: ShowcaseResponse }
  | { state: 'error'; message: string }
