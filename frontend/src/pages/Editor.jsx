import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { api, mediaUrl } from '../api'
import { ASPECT_CSS, ErrorBox, Modal, PageLoader, Spinner } from '../components/ui'

const TABS = [['script', '1 · Script'], ['avatar', '2 · Avatar'], ['voice', '3 · Voice'], ['look', '4 · Look & feel']]

const bgCss = (bg, assets) => {
  if (!bg) return undefined
  if (bg.type === 'color') return { background: bg.color }
  if (bg.type === 'image') {
    const a = assets.find((x) => x.id === bg.asset_id)
    return a ? { backgroundImage: `url(${mediaUrl(a.url)})`, backgroundSize: 'cover', backgroundPosition: 'center' } : { background: '#334155' }
  }
  return { background: `linear-gradient(135deg, ${bg.color}, ${bg.color2})` }
}

/* ---------- reusable bits ---------- */
function UploadButton({ kind, label, onUploaded, setError, className = 'btn-secondary' }) {
  const ref = useRef()
  const [busy, setBusy] = useState(false)
  const pick = async (e) => {
    const file = e.target.files[0]
    e.target.value = ''
    if (!file) return
    setBusy(true)
    try { onUploaded(await api.uploadAsset(file, kind)) } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  return (
    <>
      <input ref={ref} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={pick} />
      <button type="button" className={className} disabled={busy} onClick={() => ref.current.click()}>
        {busy ? <Spinner className="h-4 w-4" /> : '⬆'} {label}
      </button>
    </>
  )
}

function BackgroundPicker({ value, onChange, assets, onAsset, gradients, setError, allowInherit }) {
  const type = value ? value.type : 'inherit'
  const v = value || { type: 'gradient', color: '#4f46e5', color2: '#9333ea', asset_id: null }
  const imgs = assets.filter((a) => a.kind === 'image')
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        {[...(allowInherit ? [['inherit', 'Default']] : []), ['gradient', 'Gradient'], ['color', 'Solid'], ['image', 'Image']].map(([t, l]) => (
          <button key={t} type="button" onClick={() => onChange(t === 'inherit' ? null : { ...v, type: t })}
            className={`rounded-full border px-3 py-1 text-sm ${type === t ? 'border-brand-600 bg-brand-50 font-semibold text-brand-700' : 'border-slate-300 hover:bg-slate-50'}`}>{l}</button>
        ))}
      </div>
      {type === 'gradient' && (
        <div className="flex flex-wrap items-center gap-2">
          {gradients.map(([a, b]) => (
            <button key={a + b} type="button" aria-label={`Gradient ${a} to ${b}`} onClick={() => onChange({ ...v, type: 'gradient', color: a, color2: b })}
              className={`h-9 w-9 rounded-full border-2 ${v.color === a && v.color2 === b ? 'border-slate-900' : 'border-white shadow'}`} style={{ background: `linear-gradient(135deg, ${a}, ${b})` }} />
          ))}
          <input type="color" aria-label="Gradient start" value={v.color} onChange={(e) => onChange({ ...v, color: e.target.value })} className="h-9 w-9 cursor-pointer rounded" />
          <input type="color" aria-label="Gradient end" value={v.color2} onChange={(e) => onChange({ ...v, color2: e.target.value })} className="h-9 w-9 cursor-pointer rounded" />
        </div>
      )}
      {type === 'color' && <input type="color" aria-label="Background colour" value={v.color} onChange={(e) => onChange({ ...v, color: e.target.value })} className="h-10 w-20 cursor-pointer rounded" />}
      {type === 'image' && (
        <div className="flex flex-wrap items-center gap-2">
          {imgs.map((a) => (
            <button key={a.id} type="button" onClick={() => onChange({ ...v, type: 'image', asset_id: a.id })}
              className={`h-12 w-12 overflow-hidden rounded-lg border-2 ${v.asset_id === a.id ? 'border-brand-600' : 'border-transparent'}`}>
              <img src={mediaUrl(a.url)} alt={a.name} className="h-full w-full object-cover" />
            </button>
          ))}
          <UploadButton kind="image" label="Upload" setError={setError} onUploaded={(a) => { onAsset(a); onChange({ ...v, type: 'image', asset_id: a.id }) }} />
        </div>
      )}
    </div>
  )
}

/* ---------- live preview ---------- */
function Preview({ p, avatarUrl, assets, sceneIdx, mediaAsset }) {
  const scene = p.scenes[sceneIdx]
  const bg = scene?.background || p.background
  const words = (scene?.text || 'Your captions appear here').split(/\s+/).slice(0, 6)
  const portrait = p.aspect_ratio !== '16:9'
  const sideBySide = p.aspect_ratio === '16:9' && mediaAsset
  const capClass = p.caption_style === 'classic' ? 'bg-slate-900/80 text-white rounded-lg px-3 py-1'
    : p.caption_style === 'minimal' ? 'text-white [text-shadow:0_2px_4px_rgba(0,0,0,.6)]' : 'text-white [-webkit-text-stroke:2px_black] [paint-order:stroke_fill]'
  return (
    <div className={`relative mx-auto w-full overflow-hidden rounded-xl shadow-lg ${portrait ? 'max-w-[260px]' : ''}`} style={{ aspectRatio: ASPECT_CSS[p.aspect_ratio], ...bgCss(bg, assets) }}>
      {mediaAsset && (
        <img src={mediaUrl(mediaAsset.url)} alt="" className={`absolute rounded-lg bg-white/10 object-contain shadow-lg ${sideBySide ? 'left-[5%] top-[12%] h-[62%] w-[47%]' : 'left-[7%] top-[6%] h-[34%] w-[86%]'}`} />
      )}
      {avatarUrl && (
        <img src={mediaUrl(avatarUrl)} alt="Selected avatar"
          className={`absolute bottom-0 -translate-x-1/2 object-contain ${sideBySide ? 'left-[74%] h-[92%]' : portrait ? `left-1/2 ${mediaAsset ? 'w-[82%]' : 'w-[120%]'}` : 'left-1/2 h-[92%]'}`} />
      )}
      {p.captions_enabled && (
        <div className={`absolute inset-x-0 text-center font-extrabold leading-tight ${portrait ? `${mediaAsset ? 'top-[47%]' : 'top-[20%]'} text-[11px]` : 'bottom-[10%] text-sm'}`}>
          <span className={`inline-block max-w-[85%] ${capClass}`}>
            {words.map((w, i) => <span key={i} className={p.caption_style === 'bold' && i === 1 ? 'text-yellow-300' : ''}>{w} </span>)}
          </span>
        </div>
      )}
    </div>
  )
}

/* ---------- tabs ---------- */
function ScriptTab({ p, set, assets, onAsset, setError, cfg, sceneIdx, setSceneIdx, gradients }) {
  const [showImport, setShowImport] = useState(false)
  const updScene = (i, patch) => set({ scenes: p.scenes.map((s, j) => (j === i ? { ...s, ...patch } : s)) })
  const move = (i, d) => {
    const arr = [...p.scenes]
    const [it] = arr.splice(i, 1)
    arr.splice(i + d, 0, it)
    set({ scenes: arr })
    setSceneIdx(i + d)
  }
  const total = p.scenes.reduce((n, s) => n + s.text.length, 0)
  const maxScenes = cfg?.limits.max_scenes ?? 20
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-slate-500">{p.scenes.length} scene{p.scenes.length === 1 ? '' : 's'} · {total} characters</p>
        <button className="btn-secondary !py-1.5" onClick={() => setShowImport(true)}>✨ Import from text / URL</button>
      </div>
      {p.scenes.map((s, i) => {
        const media = assets.find((a) => a.id === s.media_asset_id)
        return (
          <div key={i} onClick={() => setSceneIdx(i)} className={`card p-4 ${i === sceneIdx ? 'ring-2 ring-brand-500' : ''}`}>
            <div className="mb-2 flex items-center justify-between">
              <span className="text-sm font-bold text-brand-600">Scene {i + 1}</span>
              <div className="flex gap-1 text-sm">
                <button className="btn-secondary !px-2 !py-1" disabled={i === 0} onClick={() => move(i, -1)} aria-label="Move up">↑</button>
                <button className="btn-secondary !px-2 !py-1" disabled={i === p.scenes.length - 1} onClick={() => move(i, 1)} aria-label="Move down">↓</button>
                <button className="btn-danger !px-2 !py-1" disabled={p.scenes.length === 1} aria-label="Delete scene"
                  onClick={() => { set({ scenes: p.scenes.filter((_, j) => j !== i) }); setSceneIdx(Math.max(0, i - 1)) }}>🗑</button>
              </div>
            </div>
            <textarea className="input min-h-24" placeholder="What should the avatar say in this scene?" value={s.text} maxLength={cfg?.limits.max_scene_chars ?? 1000}
              onChange={(e) => updScene(i, { text: e.target.value })} aria-label={`Scene ${i + 1} text`} />
            <details className="mt-2 text-sm">
              <summary className="cursor-pointer text-slate-500">Scene background & image</summary>
              <div className="mt-3 space-y-3">
                <BackgroundPicker allowInherit value={s.background} onChange={(b) => updScene(i, { background: b })} assets={assets} onAsset={onAsset} gradients={gradients} setError={setError} />
                <div className="flex items-center gap-3">
                  {media && <img src={mediaUrl(media.url)} alt="" className="h-12 w-12 rounded-lg object-cover" />}
                  <UploadButton kind="image" label={media ? 'Replace image' : 'Add image to scene'} setError={setError} onUploaded={(a) => { onAsset(a); updScene(i, { media_asset_id: a.id }) }} />
                  {media && <button className="text-red-600 underline" onClick={() => updScene(i, { media_asset_id: null })}>Remove</button>}
                </div>
              </div>
            </details>
          </div>
        )
      })}
      <button className="btn-secondary w-full" disabled={p.scenes.length >= maxScenes}
        onClick={() => { set({ scenes: [...p.scenes, { text: '', background: null, media_asset_id: null }] }); setSceneIdx(p.scenes.length) }}>＋ Add scene</button>
      {showImport && <ImportModal onClose={() => setShowImport(false)} cfg={cfg} onApply={(texts, replace) => {
        const scenes = texts.map((t) => ({ text: t, background: null, media_asset_id: null }))
        set({ scenes: replace ? scenes : [...p.scenes.filter((s) => s.text.trim()), ...scenes].slice(0, maxScenes) })
        setSceneIdx(0)
        setShowImport(false)
      }} />}
    </div>
  )
}

function ImportModal({ onClose, onApply, cfg }) {
  const [mode, setMode] = useState('text')
  const [text, setText] = useState('')
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const run = async () => {
    setBusy(true)
    setError('')
    try { setResult(mode === 'text' ? await api.importText(text, 6) : await api.importUrl(url, 6)) } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  return (
    <Modal title="Turn text or a blog post into scenes" onClose={onClose} wide>
      <div className="mb-3 flex gap-2">
        {[['text', 'Paste text'], ['url', 'Blog URL']].map(([m, l]) => (
          <button key={m} onClick={() => { setMode(m); setResult(null) }} className={`rounded-full border px-3 py-1 text-sm ${mode === m ? 'border-brand-600 bg-brand-50 font-semibold text-brand-700' : 'border-slate-300'}`}>{l}</button>
        ))}
      </div>
      {mode === 'text'
        ? <textarea className="input min-h-40" placeholder="Paste an article, notes, or any long text (min. 20 characters)…" value={text} onChange={(e) => setText(e.target.value)} />
        : <input className="input" type="url" placeholder="https://example.com/my-blog-post" value={url} onChange={(e) => setUrl(e.target.value)} />}
      <p className="mt-2 text-xs text-slate-500">{cfg?.llm_enabled ? 'Summarised with an LLM.' : 'Summarised with a simple built-in extractive summariser (set ANTHROPIC_API_KEY on the server for LLM summaries).'}</p>
      <div className="mt-3"><ErrorBox>{error}</ErrorBox></div>
      {result && (
        <div className="mt-4 space-y-2">
          <p className="text-sm font-semibold">{result.scenes.length} scenes generated ({result.method})</p>
          {result.scenes.map((s, i) => <p key={i} className="rounded-lg bg-slate-50 p-3 text-sm">{s}</p>)}
        </div>
      )}
      <div className="mt-5 flex flex-wrap justify-end gap-2">
        <button className="btn-secondary" onClick={onClose}>Cancel</button>
        {result ? (
          <>
            <button className="btn-secondary" onClick={() => onApply(result.scenes, false)}>Append to script</button>
            <button className="btn-primary" onClick={() => onApply(result.scenes, true)}>Replace script</button>
          </>
        ) : (
          <button className="btn-primary" disabled={busy || (mode === 'text' ? text.trim().length < 20 : !url)} onClick={run}>{busy && <Spinner className="h-4 w-4 text-white" />} Generate scenes</button>
        )}
      </div>
    </Modal>
  )
}

function AvatarTab({ p, set, avatars, assets, onAsset, setError, removeAsset }) {
  const mine = assets.filter((a) => a.kind === 'avatar')
  const Tile = ({ id, url, name, onDelete }) => (
    <div className="relative">
      <button type="button" onClick={() => set({ avatar_id: id })} aria-pressed={p.avatar_id === id}
        className={`block w-full overflow-hidden rounded-xl border-2 bg-gradient-to-br from-brand-100 to-slate-100 p-1 ${p.avatar_id === id ? 'border-brand-600 shadow-md' : 'border-transparent hover:border-slate-300'}`}>
        <img src={mediaUrl(url)} alt={name} className="aspect-square w-full object-contain" />
        <span className="block truncate pb-1 text-center text-sm font-medium">{name}</span>
      </button>
      {onDelete && <button onClick={onDelete} aria-label="Delete upload" className="absolute right-1 top-1 rounded-full bg-white/90 px-1.5 text-red-600 shadow">×</button>}
    </div>
  )
  return (
    <div className="space-y-6">
      <section>
        <h3 className="mb-3 font-semibold">Choose a presenter</h3>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">{avatars.map((a) => <Tile key={a.id} id={a.id} url={a.url} name={a.name} />)}</div>
      </section>
      <section>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="font-semibold">Your own image</h3>
          <UploadButton kind="avatar" label="Upload image" setError={setError} onUploaded={(a) => { onAsset(a); set({ avatar_id: `asset:${a.id}` }) }} />
        </div>
        {mine.length === 0 ? <p className="text-sm text-slate-500">Upload a front-facing portrait (PNG, JPG, WebP). The jaw is animated automatically — it works best with the mouth around 2/3 down the image and a neutral expression.</p> : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {mine.map((a) => <Tile key={a.id} id={`asset:${a.id}`} url={a.url} name={a.name || 'Upload'} onDelete={() => removeAsset(a)} />)}
          </div>
        )}
      </section>
    </div>
  )
}

function VoiceTab({ p, set, voices }) {
  const [text, setText] = useState('Hello! This is how I will sound in your video.')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const audio = useRef()
  const groups = useMemo(() => {
    const g = {}
    voices.forEach((v) => (g[v.language] ||= []).push(v))
    return g
  }, [voices])
  const preview = async () => {
    setBusy(true)
    setError('')
    try {
      const { url } = await api.previewVoice(p.voice, text, p.speed)
      audio.current.src = mediaUrl(url)
      await audio.current.play()
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  return (
    <div className="space-y-5">
      <div>
        <label className="label" htmlFor="voice">Voice & language</label>
        <select id="voice" className="input" value={p.voice} onChange={(e) => set({ voice: e.target.value })}>
          {Object.entries(groups).map(([lang, vs]) => (
            <optgroup key={lang} label={lang}>{vs.map((v) => <option key={v.id} value={v.id}>{v.name} ({v.gender})</option>)}</optgroup>
          ))}
        </select>
      </div>
      <div>
        <label className="label" htmlFor="speed">Speaking speed: {p.speed.toFixed(2)}×</label>
        <input id="speed" type="range" min="0.5" max="2" step="0.05" value={p.speed} onChange={(e) => set({ speed: parseFloat(e.target.value) })} className="w-full accent-brand-600" />
      </div>
      <div>
        <label className="label" htmlFor="ptext">Preview text</label>
        <textarea id="ptext" className="input" rows={2} maxLength={200} value={text} onChange={(e) => setText(e.target.value)} />
        <button className="btn-secondary mt-2" disabled={busy || !text.trim()} onClick={preview}>{busy ? <Spinner className="h-4 w-4" /> : '🔊'} Preview voice</button>
        <audio ref={audio} hidden />
        <div className="mt-2"><ErrorBox>{error}</ErrorBox></div>
      </div>
      <p className="rounded-lg bg-slate-50 p-3 text-xs text-slate-500">Voices use free Microsoft Edge TTS when the server has internet access, falling back to Google TTS, then an offline engine (voice quality and language coverage of the fallbacks are limited).</p>
    </div>
  )
}

function LookTab({ p, set, cfg, assets, onAsset, setError, music }) {
  const audio = useRef()
  const playTrack = (t) => { audio.current.src = mediaUrl(t.url); audio.current.play() }
  return (
    <div className="space-y-7">
      <section>
        <h3 className="mb-2 font-semibold">Format</h3>
        <div className="grid grid-cols-3 gap-3">
          {cfg?.aspects.map((a) => (
            <button key={a.id} onClick={() => set({ aspect_ratio: a.id })} aria-pressed={p.aspect_ratio === a.id}
              className={`rounded-xl border-2 p-3 text-center ${p.aspect_ratio === a.id ? 'border-brand-600 bg-brand-50' : 'border-slate-200 hover:border-slate-300'}`}>
              <div className="mx-auto mb-2 rounded border-2 border-slate-400" style={{ aspectRatio: ASPECT_CSS[a.id], height: 32 }} />
              <div className="text-sm font-semibold">{a.id}</div><div className="text-xs text-slate-500">{a.hint}</div>
            </button>
          ))}
        </div>
      </section>
      <section>
        <h3 className="mb-2 font-semibold">Default background</h3>
        <BackgroundPicker value={p.background} onChange={(b) => set({ background: b })} assets={assets} onAsset={onAsset} gradients={cfg?.gradients || []} setError={setError} />
      </section>
      <section>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="font-semibold">Captions</h3>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input type="checkbox" className="h-4 w-4 accent-brand-600" checked={p.captions_enabled} onChange={(e) => set({ captions_enabled: e.target.checked })} /> Show captions
          </label>
        </div>
        {p.captions_enabled && (
          <div className="grid gap-2 sm:grid-cols-3">
            {cfg?.caption_styles.map((s) => (
              <button key={s.id} onClick={() => set({ caption_style: s.id })} aria-pressed={p.caption_style === s.id}
                className={`rounded-xl border-2 p-3 text-left ${p.caption_style === s.id ? 'border-brand-600 bg-brand-50' : 'border-slate-200 hover:border-slate-300'}`}>
                <div className="text-sm font-semibold">{s.name}</div><div className="text-xs text-slate-500">{s.description}</div>
              </button>
            ))}
          </div>
        )}
      </section>
      <section>
        <h3 className="mb-2 font-semibold">Background music</h3>
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-sm"><input type="radio" name="music" className="accent-brand-600" checked={!p.music_id} onChange={() => set({ music_id: null })} /> None</label>
          {music.map((t) => (
            <div key={t.id} className="flex items-center justify-between gap-2">
              <label className="flex items-center gap-2 text-sm"><input type="radio" name="music" className="accent-brand-600" checked={p.music_id === t.id} onChange={() => set({ music_id: t.id })} /> {t.name}</label>
              <button className="btn-secondary !px-2 !py-1 text-xs" onClick={() => playTrack(t)}>▶ Play</button>
            </div>
          ))}
          <audio ref={audio} hidden />
          {p.music_id && (
            <div>
              <label className="label" htmlFor="mv">Music volume: {Math.round(p.music_volume * 100)}%</label>
              <input id="mv" type="range" min="0" max="0.6" step="0.01" value={p.music_volume} onChange={(e) => set({ music_volume: parseFloat(e.target.value) })} className="w-full accent-brand-600" />
            </div>
          )}
        </div>
      </section>
    </div>
  )
}

/* ---------- editor shell ---------- */
export default function Editor() {
  const { id } = useParams()
  const nav = useNavigate()
  const [p, setP] = useState(null)
  const [dirty, setDirty] = useState(false)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [rendering, setRendering] = useState(false)
  const [tab, setTab] = useState('script')
  const [sceneIdx, setSceneIdx] = useState(0)
  const [cfg, setCfg] = useState(null)
  const [avatars, setAvatars] = useState([])
  const [voices, setVoices] = useState([])
  const [music, setMusic] = useState([])
  const [assets, setAssets] = useState([])

  useEffect(() => {
    Promise.all([api.getProject(id), api.config(), api.avatars(), api.voices(), api.music(), api.listAssets()])
      .then(([proj, c, a, v, m, as]) => { setP(proj); setCfg(c); setAvatars(a); setVoices(v); setMusic(m); setAssets(as) })
      .catch((e) => setError(e.message))
  }, [id])

  useEffect(() => {
    if (!dirty) return
    const h = (e) => e.preventDefault()
    window.addEventListener('beforeunload', h)
    return () => window.removeEventListener('beforeunload', h)
  }, [dirty])

  const set = useCallback((patch) => { setP((cur) => ({ ...cur, ...patch })); setDirty(true) }, [])
  const onAsset = (a) => setAssets((cur) => (cur.some((x) => x.id === a.id) ? cur : [a, ...cur]))
  const removeAsset = async (a) => {
    try {
      await api.deleteAsset(a.id)
      setAssets((cur) => cur.filter((x) => x.id !== a.id))
      if (p.avatar_id === `asset:${a.id}`) set({ avatar_id: 'aria' })
    } catch (e) { setError(e.message) }
  }

  const save = async () => {
    const body = {
      title: p.title, aspect_ratio: p.aspect_ratio, avatar_id: p.avatar_id, voice: p.voice, speed: p.speed,
      captions_enabled: p.captions_enabled, caption_style: p.caption_style, music_volume: p.music_volume,
      background: p.background, clear_music: !p.music_id, music_id: p.music_id || undefined,
      scenes: p.scenes.map((s) => ({ text: s.text, background: s.background, media_asset_id: s.media_asset_id })),
    }
    const saved = await api.updateProject(id, body)
    setP(saved)
    setDirty(false)
  }
  const onSave = async () => {
    setSaving(true)
    setError('')
    try { await save() } catch (e) { setError(e.message) } finally { setSaving(false) }
  }
  const onRender = async () => {
    setRendering(true)
    setError('')
    try {
      await save()
      await api.render(id)
      nav(`/videos/${id}`)
    } catch (e) { setError(e.message); setRendering(false) }
  }

  const avatarUrl = useMemo(() => {
    if (!p) return null
    if (p.avatar_id.startsWith('asset:')) return assets.find((a) => `asset:${a.id}` === p.avatar_id)?.url
    return avatars.find((a) => a.id === p.avatar_id)?.url
  }, [p, avatars, assets])

  if (!p || !cfg) return error ? <div className="mx-auto max-w-3xl p-8"><ErrorBox>{error}</ErrorBox></div> : <PageLoader />
  const scene = p.scenes[Math.min(sceneIdx, p.scenes.length - 1)]
  const mediaAsset = assets.find((a) => a.id === scene?.media_asset_id)
  const canRender = p.scenes.some((s) => s.text.trim())

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <Link to="/dashboard" className="text-sm text-slate-500 hover:text-slate-800">←</Link>
          <input className="input !w-64 max-w-full !font-bold" aria-label="Video title" maxLength={120} value={p.title} onChange={(e) => set({ title: e.target.value })} />
          {dirty && <span className="text-xs text-amber-600">Unsaved changes</span>}
        </div>
        <div className="flex gap-2">
          <button className="btn-secondary" onClick={onSave} disabled={saving || rendering || !dirty}>{saving && <Spinner className="h-4 w-4" />} Save</button>
          <button className="btn-primary" onClick={onRender} disabled={rendering || saving || !canRender || p.status === 'queued' || p.status === 'rendering'}>
            {rendering ? <Spinner className="h-4 w-4 text-white" /> : '▶'} Render video
          </button>
        </div>
      </div>
      <div className="mt-3"><ErrorBox>{error}</ErrorBox></div>

      <div className="mt-4 grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div>
          <div className="mb-4 flex gap-1 overflow-x-auto rounded-xl bg-slate-200/60 p-1" role="tablist">
            {TABS.map(([k, l]) => (
              <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}
                className={`flex-1 whitespace-nowrap rounded-lg px-3 py-2 text-sm font-semibold ${tab === k ? 'bg-white text-brand-700 shadow-sm' : 'text-slate-600 hover:text-slate-900'}`}>{l}</button>
            ))}
          </div>
          {tab === 'script' && <ScriptTab p={p} set={set} assets={assets} onAsset={onAsset} setError={setError} cfg={cfg} sceneIdx={sceneIdx} setSceneIdx={setSceneIdx} gradients={cfg.gradients} />}
          {tab === 'avatar' && <AvatarTab p={p} set={set} avatars={avatars} assets={assets} onAsset={onAsset} setError={setError} removeAsset={removeAsset} />}
          {tab === 'voice' && <VoiceTab p={p} set={set} voices={voices} />}
          {tab === 'look' && <LookTab p={p} set={set} cfg={cfg} assets={assets} onAsset={onAsset} setError={setError} music={music} />}
        </div>
        <aside className="lg:sticky lg:top-24 lg:self-start">
          <div className="card p-4">
            <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">Preview · scene {Math.min(sceneIdx, p.scenes.length - 1) + 1} (approximate)</p>
            <Preview p={p} avatarUrl={avatarUrl} assets={assets} sceneIdx={Math.min(sceneIdx, p.scenes.length - 1)} mediaAsset={mediaAsset} />
            <p className="mt-3 text-xs text-slate-500">The final video animates the avatar's mouth to the voice.</p>
          </div>
        </aside>
      </div>
    </div>
  )
}
