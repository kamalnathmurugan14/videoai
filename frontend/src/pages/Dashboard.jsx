import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, mediaUrl } from '../api'
import { ASPECT_CSS, ErrorBox, fmtDate, fmtDuration, Modal, PageLoader, Spinner, StatusBadge } from '../components/ui'

function NewVideoModal({ onClose }) {
  const nav = useNavigate()
  const [templates, setTemplates] = useState([])
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => { api.templates().then(setTemplates).catch((e) => setError(e.message)) }, [])

  const create = async (tpl) => {
    setBusy(tpl?.id || 'blank')
    setError('')
    try {
      const p = await api.createProject(tpl ? `${tpl.name} video` : 'Untitled video', tpl?.id)
      nav(`/projects/${p.id}/edit`)
    } catch (e) {
      setError(e.message)
      setBusy(null)
    }
  }
  const options = [{ id: 'blank', name: 'Blank video', description: 'Start from scratch with an empty scene.', aspect_ratio: '16:9' }, ...templates]
  return (
    <Modal title="Start a new video" onClose={onClose} wide>
      <ErrorBox>{error}</ErrorBox>
      <div className="mt-2 grid gap-4 sm:grid-cols-2">
        {options.map((t) => (
          <button key={t.id} disabled={!!busy} onClick={() => create(t.id === 'blank' ? null : t)}
            className="rounded-xl border border-slate-200 p-4 text-left transition hover:border-brand-500 hover:bg-brand-50 disabled:opacity-60">
            <div className="flex items-center justify-between">
              <span className="font-semibold">{t.name}</span>
              {busy === t.id ? <Spinner className="h-4 w-4" /> : <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">{t.aspect_ratio}</span>}
            </div>
            <p className="mt-1 text-sm text-slate-600">{t.description}</p>
          </button>
        ))}
      </div>
    </Modal>
  )
}

function VideoCard({ p, onDelete, onDuplicate }) {
  const active = p.status === 'queued' || p.status === 'rendering'
  return (
    <div className="card overflow-hidden">
      <Link to={`/videos/${p.id}`} className="relative flex aspect-video items-center justify-center bg-slate-900">
        {p.thumbnail_url ? <img src={mediaUrl(p.thumbnail_url)} alt="" className="h-full w-full object-contain" />
          : <span className="text-sm text-slate-400">{active ? 'Rendering…' : 'No video yet'}</span>}
        {p.duration != null && <span className="absolute bottom-2 right-2 rounded bg-black/70 px-1.5 py-0.5 text-xs text-white">{fmtDuration(p.duration)}</span>}
      </Link>
      <div className="p-4">
        <div className="flex items-start justify-between gap-2">
          <Link to={`/videos/${p.id}`} className="truncate font-semibold hover:text-brand-600" title={p.title}>{p.title}</Link>
          <StatusBadge status={p.status} />
        </div>
        <p className="mt-1 text-xs text-slate-500">{p.scene_count} scene{p.scene_count === 1 ? '' : 's'} · {p.aspect_ratio} · {fmtDate(p.updated_at)}</p>
        <div className="mt-3 flex gap-2 text-sm">
          <Link to={`/projects/${p.id}/edit`} className="btn-secondary flex-1 !py-1.5">Edit</Link>
          <button className="btn-secondary !py-1.5" onClick={() => onDuplicate(p)} title="Duplicate">⧉</button>
          <button className="btn-danger !py-1.5" onClick={() => onDelete(p)} disabled={active} title="Delete">🗑</button>
        </div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [projects, setProjects] = useState(null)
  const [usage, setUsage] = useState(null)
  const [error, setError] = useState('')
  const [showNew, setShowNew] = useState(false)

  const load = useCallback(async () => {
    try {
      const [p, u] = await Promise.all([api.listProjects(), api.usage()])
      setProjects(p)
      setUsage(u)
      setError('')
    } catch (e) {
      setError(e.message)
    }
  }, [])
  useEffect(() => { load() }, [load])

  const anyActive = projects?.some((p) => p.status === 'queued' || p.status === 'rendering')
  useEffect(() => {
    if (!anyActive) return
    const t = setInterval(load, 2500)
    return () => clearInterval(t)
  }, [anyActive, load])

  const remove = async (p) => {
    if (!confirm(`Delete "${p.title}"? This cannot be undone.`)) return
    try { await api.deleteProject(p.id); load() } catch (e) { setError(e.message) }
  }
  const duplicate = async (p) => {
    try { await api.duplicateProject(p.id); load() } catch (e) { setError(e.message) }
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">My videos</h1>
          {usage && <p className="text-sm text-slate-500">{usage.used_today} of {usage.limit} free renders used today</p>}
        </div>
        <button className="btn-primary" onClick={() => setShowNew(true)}>＋ New video</button>
      </div>
      <div className="mt-4"><ErrorBox onRetry={load}>{error}</ErrorBox></div>
      {!projects && !error && <PageLoader />}
      {projects?.length === 0 && (
        <div className="card mt-8 p-12 text-center">
          <div className="text-5xl">🎬</div>
          <h2 className="mt-4 text-lg font-semibold">No videos yet</h2>
          <p className="mt-1 text-slate-600">Create your first AI avatar video in a couple of minutes.</p>
          <button className="btn-primary mt-6" onClick={() => setShowNew(true)}>Create a video</button>
        </div>
      )}
      <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {projects?.map((p) => <VideoCard key={p.id} p={p} onDelete={remove} onDuplicate={duplicate} />)}
      </div>
      {showNew && <NewVideoModal onClose={() => setShowNew(false)} />}
    </div>
  )
}
