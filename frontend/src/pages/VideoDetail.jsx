import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, mediaUrl } from '../api'
import { ASPECT_CSS, ErrorBox, fmtDuration, PageLoader, Spinner, StatusBadge } from '../components/ui'

export default function VideoDetail() {
  const { id } = useParams()
  const nav = useNavigate()
  const [p, setP] = useState(null)
  const [error, setError] = useState('')
  const [title, setTitle] = useState('')
  const [busy, setBusy] = useState(false)
  const editing = useRef(false)

  const load = useCallback(async () => {
    try {
      const data = await api.getProject(id)
      setP(data)
      if (!editing.current) setTitle(data.title)
      setError('')
      return data
    } catch (e) {
      setError(e.message)
    }
  }, [id])

  useEffect(() => { load() }, [load])
  const active = p && (p.status === 'queued' || p.status === 'rendering')
  useEffect(() => {
    if (!active) return
    const t = setInterval(load, 1500)
    return () => clearInterval(t)
  }, [active, load])

  const act = async (fn) => {
    setBusy(true)
    setError('')
    try { await fn() } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  const rename = () => act(async () => {
    editing.current = false
    if (title.trim() && title !== p.title) setP(await api.updateProject(id, { title: title.trim() }))
  })
  const rerender = () => act(async () => { await api.render(id); await load() })
  const remove = () => act(async () => {
    if (!confirm('Delete this video permanently?')) return
    await api.deleteProject(id)
    nav('/dashboard')
  })
  const duplicate = () => act(async () => { const c = await api.duplicateProject(id); nav(`/projects/${c.id}/edit`) })

  if (!p) return error ? <div className="mx-auto max-w-3xl p-8"><ErrorBox onRetry={load}>{error}</ErrorBox></div> : <PageLoader />
  const job = p.latest_job
  const portrait = p.aspect_ratio !== '16:9'

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <Link to="/dashboard" className="text-sm text-slate-500 hover:text-slate-800">← My videos</Link>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <input className="input max-w-md !text-lg !font-bold" value={title} aria-label="Video title" maxLength={120}
          onFocus={() => (editing.current = true)} onChange={(e) => setTitle(e.target.value)} onBlur={rename}
          onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()} />
        <StatusBadge status={p.status} />
      </div>
      <div className="mt-4"><ErrorBox>{error}</ErrorBox></div>

      <div className="mt-6 grid gap-8 lg:grid-cols-[1fr_320px]">
        <div>
          <div className={`mx-auto overflow-hidden rounded-2xl bg-slate-900 shadow-lg ${portrait ? 'max-w-sm' : ''}`} style={{ aspectRatio: ASPECT_CSS[p.aspect_ratio] }}>
            {p.status === 'done' && p.video_url ? (
              <video key={p.video_url} src={mediaUrl(p.video_url)} poster={mediaUrl(p.thumbnail_url)} controls playsInline className="h-full w-full" />
            ) : active ? (
              <div className="flex h-full flex-col items-center justify-center gap-4 px-8 text-center text-white">
                <Spinner className="h-10 w-10 !text-white" />
                <div className="w-full max-w-sm">
                  <div className="h-2 overflow-hidden rounded-full bg-white/20">
                    <div className="h-full rounded-full bg-brand-500 transition-all duration-500" style={{ width: `${job?.progress || 2}%` }} />
                  </div>
                  <p className="mt-3 text-sm">{job?.status === 'queued' ? 'Waiting in the queue…' : job?.stage} · {job?.progress || 0}%</p>
                </div>
              </div>
            ) : (
              <div className="flex h-full items-center justify-center px-8 text-center text-slate-300">
                {p.status === 'failed' ? 'Rendering failed.' : 'This video has not been rendered yet.'}
              </div>
            )}
          </div>
          {p.status === 'failed' && <div className="mt-4"><ErrorBox>{p.error || 'Rendering failed.'}</ErrorBox></div>}
          {job?.info && p.status === 'done' && <p className="mt-3 whitespace-pre-line text-xs text-slate-400">{job.info}</p>}
        </div>

        <aside className="space-y-3">
          <div className="card space-y-3 p-5">
            {p.status === 'done' && (
              <button className="btn-primary w-full" disabled={busy} onClick={() => act(() => api.download(id, p.title))}>⬇ Download MP4</button>
            )}
            <Link to={`/projects/${id}/edit`} className="btn-secondary w-full">✎ Edit script & settings</Link>
            <button className="btn-secondary w-full" disabled={busy || active} onClick={rerender}>
              {p.status === 'draft' ? '▶ Render video' : '↻ Render again'}
            </button>
            <button className="btn-secondary w-full" disabled={busy} onClick={duplicate}>⧉ Duplicate</button>
            <button className="btn-danger w-full" disabled={busy || active} onClick={remove}>Delete</button>
          </div>
          <div className="card p-5 text-sm text-slate-600">
            <dl className="space-y-1.5">
              <div className="flex justify-between"><dt>Format</dt><dd>{p.aspect_ratio}</dd></div>
              <div className="flex justify-between"><dt>Scenes</dt><dd>{p.scene_count}</dd></div>
              {p.duration != null && <div className="flex justify-between"><dt>Length</dt><dd>{fmtDuration(p.duration)}</dd></div>}
              <div className="flex justify-between"><dt>Voice</dt><dd>{p.voice.replace('Neural', '')}</dd></div>
            </dl>
          </div>
        </aside>
      </div>

      <section className="mt-10">
        <h2 className="text-lg font-semibold">Script</h2>
        <ol className="mt-3 space-y-2">
          {p.scenes.map((s, i) => (
            <li key={s.id} className="card flex gap-3 p-4 text-sm"><span className="font-bold text-brand-600">{i + 1}</span><span>{s.text || <em className="text-slate-400">Empty scene</em>}</span></li>
          ))}
        </ol>
      </section>
    </div>
  )
}
