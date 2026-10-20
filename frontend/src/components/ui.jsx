export function Spinner({ className = 'h-5 w-5' }) {
  return (
    <svg className={`animate-spin text-brand-600 ${className}`} viewBox="0 0 24 24" fill="none" aria-label="Loading">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
      <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}

export function PageLoader() {
  return <div className="flex justify-center py-24"><Spinner className="h-8 w-8" /></div>
}

export function ErrorBox({ children, onRetry }) {
  if (!children) return null
  return (
    <div role="alert" className="flex items-start justify-between gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      <span>{children}</span>
      {onRetry && <button onClick={onRetry} className="font-semibold underline">Retry</button>}
    </div>
  )
}

const STATUS = {
  draft: 'bg-slate-100 text-slate-600',
  queued: 'bg-amber-100 text-amber-700',
  rendering: 'bg-blue-100 text-blue-700',
  done: 'bg-emerald-100 text-emerald-700',
  failed: 'bg-red-100 text-red-700',
}
export function StatusBadge({ status }) {
  return <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold capitalize ${STATUS[status] || STATUS.draft}`}>{status}</span>
}

export function Modal({ title, onClose, children, wide }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4" onClick={onClose}>
      <div role="dialog" aria-modal="true" aria-label={title} onClick={(e) => e.stopPropagation()}
        className={`card max-h-[90vh] w-full overflow-y-auto p-6 ${wide ? 'max-w-3xl' : 'max-w-md'}`}>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-bold">{title}</h2>
          <button onClick={onClose} className="text-2xl leading-none text-slate-400 hover:text-slate-600" aria-label="Close">×</button>
        </div>
        {children}
      </div>
    </div>
  )
}

export const fmtDuration = (s) => (s == null ? '' : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`)
export const fmtDate = (d) => new Date(d + 'Z').toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
export const ASPECT_CSS = { '16:9': '16 / 9', '9:16': '9 / 16', '1:1': '1 / 1' }
