// Madar Terrain service worker: app shell + target data offline; chips cached on request.
const C = "madar-terrain-v1";
const SHELL = ["./", "index.html", "manifest.webmanifest", "icon.svg", "../minerals/minerals.json", "../minerals/sahara/sahara.json"];
self.addEventListener("install", e => { e.waitUntil(caches.open(C).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener("activate", e => { e.waitUntil(self.clients.claim()); });
self.addEventListener("message", e => {
  if (e.data && e.data.cache) e.waitUntil(caches.open(C).then(c => Promise.all(e.data.cache.map(u => c.add(u).catch(() => null)))));
});
// network first (fresh targets when online), cache as fallback
self.addEventListener("fetch", e => {
  if (e.request.method !== "GET") return;
  e.respondWith(fetch(e.request).then(r => {
    if (r.ok && new URL(e.request.url).origin === location.origin) { const cp = r.clone(); caches.open(C).then(c => c.put(e.request, cp)); }
    return r;
  }).catch(() => caches.match(e.request, { ignoreSearch: true })));
});
