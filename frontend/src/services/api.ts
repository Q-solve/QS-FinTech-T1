import type { HealthResponse } from '../types/api'

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
