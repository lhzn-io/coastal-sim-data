import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── State ──
let renderer = null;
let scene = null;
let camera = null;
let controls = null;
let arrowGroup = null;
let animId = null;
let active = false;

// ── Color map: magnitude → deep blue → cyan → yellow → red ──
function velocityColor(mag, maxMag) {
    const t = Math.min(1, mag / Math.max(maxMag, 0.001));
    if (t < 0.33) {
        const s = t / 0.33;
        return new THREE.Color(0.05, 0.1 + 0.5 * s, 0.6 + 0.4 * s);
    } else if (t < 0.66) {
        const s = (t - 0.33) / 0.33;
        return new THREE.Color(0.1 + 0.9 * s, 0.6 + 0.4 * s, 1.0 - 0.8 * s);
    } else {
        const s = (t - 0.66) / 0.34;
        return new THREE.Color(1.0, 1.0 - 0.6 * s, 0.2 - 0.2 * s);
    }
}

// ── Merge simple geometries (no import needed) ──
function mergeGeos(geos) {
    let totalV = 0, allIdx = [];
    for (const g of geos) { totalV += g.attributes.position.count; }
    const pos = new Float32Array(totalV * 3);
    const nor = new Float32Array(totalV * 3);
    let vOff = 0;
    for (const g of geos) {
        const p = g.attributes.position, n = g.attributes.normal;
        for (let i = 0; i < p.count; i++) {
            pos[(vOff + i) * 3] = p.getX(i); pos[(vOff + i) * 3 + 1] = p.getY(i); pos[(vOff + i) * 3 + 2] = p.getZ(i);
            if (n) { nor[(vOff + i) * 3] = n.getX(i); nor[(vOff + i) * 3 + 1] = n.getY(i); nor[(vOff + i) * 3 + 2] = n.getZ(i); }
        }
        if (g.index) { for (let i = 0; i < g.index.count; i++) allIdx.push(g.index.array[i] + vOff); }
        vOff += p.count;
    }
    const merged = new THREE.BufferGeometry();
    merged.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    merged.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    merged.setIndex(allIdx);
    return merged;
}

function initThree(container) {
    const canvas = document.createElement('canvas');
    canvas.id = 'preview3dCanvas';
    container.appendChild(canvas);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0d1117);
    renderer.setSize(container.clientWidth, container.clientHeight);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 5000);
    camera.position.set(0, 150, 250);

    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.AmbientLight(0x404060, 0.8));
    const dir = new THREE.DirectionalLight(0xffffff, 0.7);
    dir.position.set(100, 200, 150);
    scene.add(dir);

    // Grid helper
    const grid = new THREE.GridHelper(200, 20, 0x1a2030, 0x1a2030);
    scene.add(grid);

    function animate() {
        animId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // Handle resize
    const ro = new ResizeObserver(() => {
        const w = container.clientWidth, h = container.clientHeight;
        if (w > 0 && h > 0) {
            renderer.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        }
    });
    ro.observe(container);
}

function clearArrows() {
    if (arrowGroup) {
        scene.remove(arrowGroup);
        arrowGroup.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
        arrowGroup = null;
    }
}

function buildVectorField(data) {
    clearArrows();
    if (!data.vectors || data.vectors.length === 0) return;

    arrowGroup = new THREE.Group();

    const [lonMin, latMin, lonMax, latMax] = data.bounds;
    const lonCenter = (lonMin + lonMax) / 2;
    const latCenter = (latMin + latMax) / 2;
    const cosLat = Math.cos(latCenter * Math.PI / 180);
    const degToM = 111320;

    // Scale scene to ~200 units across
    const lonSpan = Math.max(lonMax - lonMin, 0.001) * cosLat * degToM;
    const latSpan = Math.max(latMax - latMin, 0.001) * degToM;
    const maxSpan = Math.max(lonSpan, latSpan);
    const sceneScale = 200 / maxSpan;

    // Compute max magnitude
    let maxMag = 0;
    for (const v of data.vectors) {
        const mag = Math.sqrt(v.u * v.u + v.v * v.v);
        if (mag > maxMag) maxMag = mag;
    }

    // Build arrow geometry (shaft + cone)
    const shaft = new THREE.CylinderGeometry(0.12, 0.12, 1, 6);
    shaft.translate(0, 0.5, 0);
    shaft.rotateX(Math.PI / 2);
    const cone = new THREE.ConeGeometry(0.3, 0.5, 6);
    cone.rotateX(Math.PI / 2);
    cone.translate(0, 0, 1.0);
    const arrowGeo = mergeGeos([shaft, cone]);

    const count = data.vectors.length;
    const mat = new THREE.MeshPhongMaterial({ flatShading: true });
    const mesh = new THREE.InstancedMesh(arrowGeo, mat, count);

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    for (let i = 0; i < count; i++) {
        const v = data.vectors[i];
        const mag = Math.sqrt(v.u * v.u + v.v * v.v);

        const sx = (v.lon - lonCenter) * cosLat * degToM * sceneScale;
        const sz = -(v.lat - latCenter) * degToM * sceneScale;
        const sy = v.depth * sceneScale * 0.5; // vertical scale for depth

        const angle = Math.atan2(-v.v, v.u);
        const arrowLen = Math.max(1, Math.min(6, (mag / Math.max(maxMag, 0.001)) * 5));

        dummy.position.set(sx, sy, sz);
        dummy.rotation.set(0, -angle, 0);
        dummy.scale.set(arrowLen, arrowLen, arrowLen);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        mesh.setColorAt(i, velocityColor(mag, maxMag));
    }

    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    arrowGroup.add(mesh);

    // Add depth level planes as subtle reference
    for (const d of data.depth_levels) {
        if (d === 0) continue;
        const planeY = d * sceneScale * 0.5;
        const planeGeo = new THREE.PlaneGeometry(200, 200);
        const planeMat = new THREE.MeshBasicMaterial({ color: 0x1a2a40, transparent: true, opacity: 0.15, side: THREE.DoubleSide });
        const plane = new THREE.Mesh(planeGeo, planeMat);
        plane.rotation.x = -Math.PI / 2;
        plane.position.y = planeY;
        arrowGroup.add(plane);
    }

    scene.add(arrowGroup);

    // Fit camera
    const box = new THREE.Box3().setFromObject(arrowGroup);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const dist = Math.max(size.x, size.y, size.z) * 1.2;
    controls.target.copy(center);
    camera.position.set(center.x, center.y + dist * 0.6, center.z + dist * 0.8);
    controls.update();
}

// ── Public API (called from app.js) ──
window.preview3d = {
    active: false,

    show(container) {
        if (!renderer) initThree(container);
        const canvas = document.getElementById('preview3dCanvas');
        if (canvas) canvas.style.display = 'block';
        this.active = true;
    },

    hide() {
        const canvas = document.getElementById('preview3dCanvas');
        if (canvas) canvas.style.display = 'none';
        this.active = false;
        clearArrows();
    },

    async load(datasetId, timeIdx) {
        const url = `/api/v1/cache/preview3d?dataset_id=${encodeURIComponent(datasetId)}&time_idx=${timeIdx || 0}`;
        try {
            const res = await fetch(url);
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            buildVectorField(data);
            return { count: data.count, u_var: data.u_var, v_var: data.v_var, depths: data.depth_levels.length };
        } catch (e) {
            console.error('3D preview failed:', e);
            throw e;
        }
    },

    dispose() {
        clearArrows();
        if (animId) cancelAnimationFrame(animId);
        if (renderer) renderer.dispose();
        renderer = null; scene = null; camera = null; controls = null;
        this.active = false;
    }
};
