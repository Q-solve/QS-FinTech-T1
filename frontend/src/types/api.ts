export interface HealthResponse {
  status: 'ok'
  service: string
  version: string
}

export type ApiHealthState =
  | { state: 'loading' }
  | { state: 'online'; data: HealthResponse }
  | { state: 'offline'; message: string }
