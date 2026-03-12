document.addEventListener("DOMContentLoaded", () => {
    console.log("Coastal Edge UI Initialized");
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
            const typeLabel = isIC ? "Initial Conditions" : (isBC || item.type === "zarr" ? "Boundary Forcing" : item.type);
            const typeClass = isIC ? "type-ic" : (isBC ? "type-bc" : `type-${item.type}`);

            card.innerHTML = `
                <div class="card-title">${item.id}</div>
                <div class="card-meta">
                    <span class="type-tag ${typeClass}">${typeLabel}</span>
                    <span>${item.size_mb} MB</span>
                    <span>${date}</span>
                </div>
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

        Array.from(varSelect.options).forEach(opt => {
            opt.style.display = "none";
            if (item.id.startsWith("bc_") || item.id.startsWith("hrrr") || item.id.startsWith("era5")) {
                if (["u10", "v10", "t2m", "water_level"].includes(opt.value)) opt.style.display = "";
            }
            if (item.id.startsWith("ic_") || item.id.includes("hycom") || item.id.includes("nyhops") || item.id.includes("maracoos")) {
                if (["u", "v", "temp", "salt", "zeta", "water_u", "water_v", "hs"].includes(opt.value)) {
                    opt.style.display = "";
                }
            }
        });

        for (let opt of varSelect.options) {
            if (opt.style.display !== "none") {
                varSelect.value = opt.value;
                break;
            }
        }

        updatePreview();
    }

    varSelect.addEventListener("change", updatePreview);

    timeRange.addEventListener("input", (e) => {
        timeIdxDisplay.textContent = e.target.value;
        clearTimeout(previewDebounce);
        previewDebounce = setTimeout(updatePreview, 300);
    });

    async function updatePreview() {
        if (!selectedDataset) return;

        let loader = document.querySelector(".preview-loader");
        if (!loader) {
            loader = document.createElement("div");
            loader.className = "preview-loader";
            loader.innerHTML = '<div class="spinner"></div>';
            previewCanvas.appendChild(loader);
        }

        const v = varSelect.value;
        const t = timeRange.value;
        const ext = selectedDataset.type;
        const id = selectedDataset.id;

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
