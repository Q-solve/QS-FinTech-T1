import type { HealthResponse, ShowcaseResponse } from '../types/api'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export async function fetchApiHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`, {
    headers: { Accept: 'application/json' },
    signal,
  })

  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`)
  }

  return (await response.json()) as HealthResponse
}

export async function fetchShowcase(signal?: AbortSignal): Promise<ShowcaseResponse> {
  const response = await fetch(`${API_BASE_URL}/api/showcase`, {
    headers: { Accept: 'application/json' },
    signal,
  })
  if (!response.ok) throw new Error(`Showcase request failed with status ${response.status}`)
  return (await response.json()) as ShowcaseResponse
}
