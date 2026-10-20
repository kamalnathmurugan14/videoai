// Thin API client: attaches the JWT, normalises errors.
const BASE = import.meta.env.VITE_API_URL || ''
const TOKEN_KEY = 'videoai_token'

export const getToken = () => localStorage.getItem(TOKEN_KEY)
export const setToken = (t) => (t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY))
export const mediaUrl = (u) => (u && u.startsWith('/') ? BASE + u : u)

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

function describe(detail, fallback) {
  if (!detail) return fallback
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ')
  return fallback
}

async function request(path, { method = 'GET', body, form, raw } = {}) {
  const headers = {}
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`
  let payload
  if (form) payload = form
  else if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
    payload = JSON.stringify(body)
  }
  let res
  try {
    res = await fetch(BASE + '/api' + path, { method, headers, body: payload })
  } catch {
    throw new ApiError('Cannot reach the server. Is the backend running?', 0)
  }
  if (res.status === 401 && token) window.dispatchEvent(new Event('videoai:unauthorized'))
  if (!res.ok) {
    let detail
    try { detail = (await res.json()).detail } catch { /* not json */ }
    throw new ApiError(describe(detail, `Request failed (${res.status})`), res.status)
  }
  if (raw) return res
  return res.status === 204 ? null : res.json()
}

export const api = {
  signup: (email, password) => request('/auth/signup', { method: 'POST', body: { email, password } }),
  login: (email, password) => request('/auth/login', { method: 'POST', body: { email, password } }),
  guest: () => request('/auth/guest', { method: 'POST' }),
  me: () => request('/auth/me'),
  usage: () => request('/usage'),
  config: () => request('/config'),
  system: () => request('/system'),
  voiceCheck: () => request('/system/voice-check', { method: 'POST' }),
  avatars: () => request('/avatars'),
  sceneries: () => request('/sceneries'),
  voices: () => request('/voices'),
  templates: () => request('/templates'),
  music: () => request('/music'),
  previewVoice: (voice, text, speed) => request('/voices/preview', { method: 'POST', body: { voice, text, speed } }),
  importText: (text, max_scenes) => request('/import/text', { method: 'POST', body: { text, max_scenes } }),
  importUrl: (url, max_scenes) => request('/import/url', { method: 'POST', body: { url, max_scenes } }),
  listProjects: () => request('/projects'),
  createProject: (title, template_id) => request('/projects', { method: 'POST', body: { title, template_id } }),
  getProject: (id) => request(`/projects/${id}`),
  updateProject: (id, body) => request(`/projects/${id}`, { method: 'PUT', body }),
  deleteProject: (id) => request(`/projects/${id}`, { method: 'DELETE' }),
  duplicateProject: (id) => request(`/projects/${id}/duplicate`, { method: 'POST' }),
  render: (id) => request(`/projects/${id}/render`, { method: 'POST' }),
  listAssets: () => request('/media'),
  uploadAsset: (file, kind = 'image', extra = {}) => {
    const form = new FormData()
    form.append('kind', kind)
    Object.entries(extra).forEach(([k, v]) => form.append(k, String(v)))
    form.append('file', file)
    return request('/media', { method: 'POST', form })
  },
  deleteAsset: (id) => request(`/media/${id}`, { method: 'DELETE' }),
  setMouth: (id, mouth) => request(`/media/${id}/mouth`, { method: 'PUT', body: mouth }),
  async download(id, title) {
    const res = await request(`/projects/${id}/download`, { raw: true })
    const url = URL.createObjectURL(await res.blob())
    const a = document.createElement('a')
    a.href = url
    a.download = `${title || 'video'}.mp4`
    a.click()
    URL.revokeObjectURL(url)
  },
}
