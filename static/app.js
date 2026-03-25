document.addEventListener("DOMContentLoaded", () => {
    console.log("coastal-sim-data UI Initialized");
    const datasetList = document.getElementById("datasetList");
    const datasetCount = document.getElementById("datasetCount");
    const refreshBtn = document.getElementById("refreshBtn");
    const purgeBtn = document.getElementById("purgeBtn");

    // Preview Canvas Elements
    const previewCanvas = document.getElementById("previewCanvas");
    const previewControls = document.getElementById("previewControls");
    const varSelect = document.getElementById("varSelect");
    const timeRange = document.getElementById("timeRange");
    const timeIdxDisplay = document.getElementById("timeIdxDisplay");

    let currentInventory = [];
    let selectedDataset = null;
    let previewDebounce = null;
    let mode3d = false;
    const vectorPairs = [["u", "v"], ["water_u", "water_v"], ["u10", "v10"]];

    // Initialize View
    fetchInventory().then(() => {
        // Deep-link: auto-select dataset from URL hash (e.g. #bc_a90a4eea6514)
        const hash = window.location.hash.replace('#', '');
        if (hash && currentInventory.length > 0) {
            const match = currentInventory.find(item => item.id === hash);
            if (match) {
                console.log('Deep-link: auto-selecting', hash);
                const cards = document.querySelectorAll('.dataset-card');
                const idx = currentInventory.indexOf(match);
                if (cards[idx]) {
                    selectDataset(match, cards[idx]);
                    cards[idx].scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
        }
    });

    if (refreshBtn) {
        refreshBtn.addEventListener("click", () => {
            console.log("Sync Cache Clicked");
            const icon = refreshBtn.querySelector("svg");
            if (icon) icon.style.animation = "spin 1s linear infinite";
            fetchInventory().finally(() => {
                setTimeout(() => { if (icon) icon.style.animation = "none"; }, 500);
            });
        });
    }

    if (purgeBtn) {
        console.log("Purge Button found in DOM, attaching listener");
        purgeBtn.addEventListener("click", async () => {
            console.log("Purge Cache Clicked - requesting confirmation");
            const confirmed = confirm("CRITICAL: Are you sure you want to purge the entire BC/IC data cache? This cannot be undone.");

            if (!confirmed) {
                console.log("Purge cancelled by user");
                return;
            }

            console.log("Purge confirmed - executing API call");
            purgeBtn.disabled = true;
            purgeBtn.textContent = "Purging...";

            try {
                const resp = await fetch("/api/v1/cache/purge", { method: "POST" });
                if (!resp.ok) throw new Error("Failed to purge cache (HTTP " + resp.status + ")");
                const data = await resp.json();
                console.log("Purge successful:", data.message);
                alert(data.message);

                selectedDataset = null;
                previewControls.classList.add("hidden");
                previewCanvas.innerHTML = '<div class="empty-state"><p>Cache purged. Select a new dataset after generation.</p></div>';
                fetchInventory();
            } catch (err) {
                console.error("Purge failed:", err);
                alert("Error: " + err.message);
            } finally {
                purgeBtn.disabled = false;
                purgeBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        <line x1="10" y1="11" x2="10" y2="17"></line>
                        <line x1="14" y1="11" x2="14" y2="17"></line>
                    </svg>
                    Purge Cache`;
            }
        });
    } else {
        console.error("Purge Button NOT found in DOM!");
    }

    async function fetchInventory() {
        try {
            const resp = await fetch("/api/v1/cache/inventory");
            if (!resp.ok) throw new Error(`HTTP Error: ${resp.status}`);
            currentInventory = await resp.json();
            renderInventory(currentInventory);
        } catch (err) {
            console.error("Failed to fetch inventory:", err);
            datasetList.innerHTML = `
                <div class="empty-state">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                    <p>Failed to load cache inventory.<br><small>${err.message}</small></p>
                </div>`;
        }
    }

    function renderInventory(items) {
        datasetCount.textContent = items.length;
        datasetList.innerHTML = "";

        if (items.length === 0) {
            datasetList.innerHTML = `
                <div class="empty-state">
                    <p>No processed datasets found in the local cache.</p>
                </div>`;
            return;
        }

        items.forEach(item => {
            const date = new Date(item.modified_time * 1000).toLocaleString(undefined, {
                month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
            });

            const card = document.createElement("div");
            card.className = "dataset-card";
            if (selectedDataset && selectedDataset.id === item.id) {
                card.classList.add("selected");
            }

            const isIC = item.id.startsWith("ic_");
            const isBC = item.id.startsWith("bc_");
            const isGRIB = item.type === "grib" || item.type === "grib2";
            const typeLabel = item.source || (isIC ? "Ocean IC" : (isBC ? "Boundary" : (isGRIB ? item.type.toUpperCase() : item.type)));
            const typeClass = isIC ? "type-ic" : (isBC ? "type-bc" : (isGRIB ? "type-grib" : `type-${item.type}`));

            // Build display label
            const displayName = item.label || item.id;
            const gridInfo = item.grid_shape ? `${item.grid_shape} grid` : '';
            const timeInfo = item.time_steps ? `${item.time_steps} steps` : '';
            const statsLine = [gridInfo, timeInfo].filter(Boolean).join(' \u00b7 ');
            const varsLine = item.variables ? item.variables.join(', ') : '';

            card.innerHTML = `
                <div class="card-title">${displayName}</div>
                <div class="card-meta">
                    <span class="type-tag ${typeClass}">${typeLabel}</span>
                    <span>${item.size_mb} MB</span>
                    <span>${date}</span>
                </div>
                ${statsLine ? `<div class="card-stats">${statsLine}</div>` : ''}
                ${varsLine ? `<div class="card-vars">${varsLine}</div>` : ''}
                ${item.spatial_extent ? `<div class="card-stats">\ud83c\udf10 ${item.spatial_extent}</div>` : ''}
                ${item.time_range ? `<div class="card-time">\ud83d\udcc5 ${item.time_range}</div>` : ''}
            `;

            card.addEventListener("click", () => selectDataset(item, card));
            datasetList.appendChild(card);
        });
    }

    function selectDataset(item, cardElement) {
        console.log("Dataset selected:", item.id);
        window.location.hash = item.id;  // Update URL for deep-linking
        document.querySelectorAll(".dataset-card").forEach(c => c.classList.remove("selected"));
        cardElement.classList.add("selected");

        selectedDataset = item;
        previewControls.classList.remove("hidden");
        timeRange.value = 0;
        timeIdxDisplay.textContent = "0";

        // Set time slider max from metadata; hide for single-snapshot datasets (e.g. ICs)
        const timeSliderGroup = document.getElementById("timeSliderGroup");
        const snapshotLabel = document.getElementById("snapshotLabel");
        const snapshotLabelText = document.getElementById("snapshotLabelText");
        if (item.time_steps && item.time_steps > 1) {
            timeRange.max = item.time_steps - 1;
            timeIdxDisplay.textContent = `0 (${item.time_steps})`;
            if (timeSliderGroup) timeSliderGroup.style.display = "";
            if (snapshotLabel) snapshotLabel.style.display = "none";
        } else {
            timeRange.max = 0;
            timeRange.value = 0;
            if (timeSliderGroup) timeSliderGroup.style.display = "none";
            // Show snapshot timestamp from target_date if available
            if (snapshotLabel && snapshotLabelText && item.target_date) {
                try {
                    const d = new Date(item.target_date);
                    snapshotLabelText.textContent = `Snapshot: ${d.toUTCString().replace('GMT', 'UTC')}`;
                } catch (_) {
                    snapshotLabelText.textContent = `Snapshot: ${item.target_date}`;
                }
                snapshotLabel.style.display = "";
            } else if (snapshotLabel) {
                snapshotLabel.style.display = "none";
            }
        }

        // Populate variable dropdown from metadata if available
        if (item.variables && item.variables.length > 0) {
            varSelect.innerHTML = '';
            item.variables.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v;
                opt.textContent = v;
                varSelect.appendChild(opt);
            });
            // Default to a vector variable for IC datasets so 3D auto-triggers
            const isIC = item.id.startsWith('ic_');
            const preferred = isIC
                ? ['u', 'water_u', 'u10', 'zeta', 'temp']
                : ['water_level', 'u10', 'zeta', 'temp'];
            const defaultVar = preferred.find(p => item.variables.includes(p)) || item.variables[0];
            varSelect.value = defaultVar;
        } else {
            // Fallback: show/hide hardcoded options
            Array.from(varSelect.options).forEach(opt => {
                opt.style.display = "none";
                if (item.id.startsWith("bc_") || item.id.startsWith("hrrr") || item.id.startsWith("era5")) {
                    if (["u10", "v10", "t2m", "water_level"].includes(opt.value)) opt.style.display = "";
                }
                if (item.id.startsWith("ic_") || item.id.includes("hycom") || item.id.includes("nyhops") || item.id.includes("maracoos")) {
                    if (["u", "v", "temp", "salt", "zeta", "water_u", "water_v", "hs"].includes(opt.value)) opt.style.display = "";
                }
            });
            for (let opt of varSelect.options) {
                if (opt.style.display !== "none") { varSelect.value = opt.value; break; }
            }
        }

        if (window.preview3d) window.preview3d.hide();

        updatePreview();
    }

    varSelect.addEventListener("change", updatePreview);

    timeRange.addEventListener("input", (e) => {
        const total = selectedDataset?.time_steps || '?';
        timeIdxDisplay.textContent = `${e.target.value} (${total})`;
        clearTimeout(previewDebounce);
        previewDebounce = setTimeout(updatePreview, 300);
    });

    async function updatePreview() {
        if (!selectedDataset) return;

        const t = timeRange.value;
        const id = selectedDataset.id;

        // Determine mode from selected variable: 3D if it's part of a u/v pair
        const selectedVar = varSelect.value;
        mode3d = selectedDataset.type === 'zarr' &&
            vectorPairs.some(([u, v]) => (selectedVar === u || selectedVar === v) &&
                selectedDataset.variables?.includes(u) && selectedDataset.variables?.includes(v));

        // ── 3D Vector Mode ──
        if (mode3d && window.preview3d) {
            // Hide 2D content, show 3D canvas
            const img = document.getElementById("renderedMap");
            if (img) img.style.display = 'none';
            // Clear empty-state divs
            const emptyState = previewCanvas.querySelector('.empty-state');
            if (emptyState) emptyState.style.display = 'none';

            window.preview3d.show(previewCanvas);

            try {
                const info = await window.preview3d.load(id, parseInt(t));
                console.log(`3D preview: ${info.count} vectors (${info.u_var}/${info.v_var}), ${info.depths} depth levels`);
            } catch (err) {
                console.error("3D preview failed:", err);
                // Fall through — keep the canvas visible with whatever was there
            }
            return;
        }

        // ── 2D Matplotlib Mode ──
        if (window.preview3d) window.preview3d.hide();

        let loader = document.querySelector(".preview-loader");
        if (!loader) {
            loader = document.createElement("div");
            loader.className = "preview-loader";
            loader.innerHTML = '<div class="spinner"></div>';
            previewCanvas.appendChild(loader);
        }

        const v = varSelect.value;
        const ext = selectedDataset.type;

        const url = `/api/v1/cache/preview?dataset_id=${encodeURIComponent(id)}&ext=${ext}&var_name=${v}&time_idx=${t}`;
        console.log("Fetching preview:", url);

        try {
            const resp = await fetch(url);
            if (!resp.ok) {
                const errJson = await resp.json().catch(() => ({}));
                throw new Error(errJson.detail || `HTTP ${resp.status}`);
            }

            const blob = await resp.blob();
            const imgUrl = URL.createObjectURL(blob);

            let img = document.getElementById("renderedMap");
            if (!img) {
                previewCanvas.innerHTML = "";
                img = document.createElement("img");
                img.id = "renderedMap";
                img.className = "preview-image";
                previewCanvas.appendChild(img);
            }
            img.style.display = '';

            img.onload = () => {
                img.classList.add("loaded");
                if (loader.parentNode) loader.parentNode.removeChild(loader);
            };
            img.src = imgUrl;

        } catch (err) {
            console.error("Preview render failed:", err);
            previewCanvas.innerHTML = `
                <div class="empty-state">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                    <p>Render Failed<br><small style="word-break: break-all;">${err.message}</small></p>
                </div>`;
            if (loader && loader.parentNode) loader.parentNode.removeChild(loader);
        }
    }
});
