import { Atom } from 'lucide-react'
import type { ReactNode } from 'react'

interface AppShellProps {
  status: ReactNode
  children: ReactNode
}

export function AppShell({ status, children }: AppShellProps) {
  return (
    <div className="app-grid min-h-screen bg-[radial-gradient(circle_at_top_right,rgba(14,165,233,0.15),transparent_32%),radial-gradient(circle_at_15%_15%,rgba(45,212,191,0.09),transparent_28%)]">
      <header className="border-b border-white/10 bg-slate-950/50 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-5 py-4 sm:px-8">
          <a
            href="/"
            className="flex items-center gap-3 rounded-lg focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-sky-300"
          >
            <span className="grid size-10 place-items-center rounded-xl border border-sky-300/25 bg-sky-300/10 text-sky-200">
              <Atom className="size-5" aria-hidden="true" />
            </span>
            <span>
              <span className="block text-base font-semibold tracking-tight text-white">
                Remit-Q
              </span>
              <span className="block text-xs text-slate-400">Research prototype</span>
            </span>
          </a>

          {status}
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-5 py-10 sm:px-8 sm:py-14">{children}</main>
    </div>
  )
}
