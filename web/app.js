const gridInput = document.querySelector("#grid");
const reynoldsInput = document.querySelector("#reynolds");
const stepsInput = document.querySelector("#steps");
const runButton = document.querySelector("#run-button");
const runStatus = document.querySelector("#run-status");
const emptyState = document.querySelector("#empty-state");
const resultContent = document.querySelector("#result-content");
const resultBadge = document.querySelector("#result-badge");
const flowImage = document.querySelector("#flow-image");
const plotCaption = document.querySelector("#plot-caption");
const benchButton = document.querySelector("#bench-button");
const benchStatus = document.querySelector("#bench-status");
const openmpButton = document.querySelector("#openmp-button");
const openmpStatus = document.querySelector("#openmp-status");
const mpiButton = document.querySelector("#mpi-button");
const mpiStatus = document.querySelector("#mpi-status");
const history = [];

let currentRun = null;
let currentImage = "velocity";

function updateLabels() {
  document.querySelector("#reynolds-value").textContent = reynoldsInput.value;
  document.querySelector("#steps-value").textContent = Number(stepsInput.value).toLocaleString("en-US");
}

function formatSeconds(value) {
  return value < 1 ? `${value.toFixed(3)} s` : `${value.toFixed(2)} s`;
}

function poissonShare(run) {
  return 100 * run.poisson_seconds / run.runtime_seconds;
}

function showImage() {
  if (!currentRun) return;
  const isVelocity = currentImage === "velocity";
  flowImage.src = isVelocity ? currentRun.velocity_image : currentRun.vorticity_image;
  flowImage.alt = isVelocity
    ? "Computed velocity magnitude with streamlines"
    : "Computed vorticity contour";
  plotCaption.textContent = `Grid ${currentRun.grid_size} × ${currentRun.grid_size} · Re ${currentRun.reynolds} · ${currentRun.steps.toLocaleString("en-US")} steps · transient run`;

  document.querySelectorAll(".image-tabs button").forEach((button) => {
    const selected = button.dataset.image === currentImage;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
}

function showRun(run) {
  currentRun = run;
  emptyState.hidden = true;
  resultContent.hidden = false;
  resultBadge.textContent = run.steady_converged ? "STEADY" : "TRANSIENT";
  resultBadge.classList.add("ready");
  showImage();

  document.querySelector("#runtime-value").textContent = formatSeconds(run.runtime_seconds);
  document.querySelector("#poisson-value").textContent = `${poissonShare(run).toFixed(1)}%`;
  document.querySelector("#sweeps-value").textContent = String(Math.round(run.mean_poisson_iterations));
  document.querySelector("#residual-value").textContent = run.last_poisson_residual.toExponential(2);

  const share = poissonShare(run);
  document.querySelector("#time-strip").hidden = false;
  document.querySelector("#poisson-bar").style.width = `${share}%`;
  document.querySelector("#other-bar").style.width = `${100 - share}%`;
  document.querySelector("#time-share-label").textContent = `${share.toFixed(1)}% in Poisson solves`;
}

function updateHistory() {
  const section = document.querySelector("#history-section");
  const body = document.querySelector("#history-body");
  section.hidden = history.length === 0;
  body.replaceChildren();

  history.forEach((run) => {
    const row = document.createElement("tr");
    const values = [
      `${run.grid_size} × ${run.grid_size} · Re ${run.reynolds}`,
      run.steps.toLocaleString("en-US"),
      formatSeconds(run.runtime_seconds),
      `${poissonShare(run).toFixed(1)}%`,
    ];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    const actionCell = document.createElement("td");
    const action = document.createElement("button");
    action.type = "button";
    action.textContent = "View";
    action.addEventListener("click", () => {
      showRun(run);
      document.querySelector("#workspace").scrollIntoView({ behavior: "smooth" });
    });
    actionCell.append(action);
    row.append(actionCell);
    body.append(row);
  });
}

async function runBenchmark() {
  benchButton.disabled = true;
  benchStatus.classList.remove("error");
  benchStatus.textContent = "Measuring four grid sizes. This can take a few seconds…";

  try {
    const response = await fetch("/api/poisson", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The benchmark failed.");

    const body = document.querySelector("#bench-body");
    body.replaceChildren();
    data.results.forEach((result, index) => {
      const previous = data.results[index - 1];
      const change = previous
        ? `${(result.median_seconds / previous.median_seconds).toFixed(1)}×`
        : "—";
      const values = [
        `${result.grid_size} × ${result.grid_size}`,
        String(result.interior_cells),
        String(result.iterations),
        `${(result.median_seconds * 1000).toFixed(2)} ms`,
        change,
        result.max_error.toExponential(2),
      ];
      const row = document.createElement("tr");
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      body.append(row);
    });

    const changes = data.results.slice(1).map((result, index) =>
      result.median_seconds / data.results[index].median_seconds);
    const smallestChange = Math.min(...changes);
    const largestChange = Math.max(...changes);
    document.querySelector("#bench-note").textContent =
      `On this machine, each doubling of grid intervals took ${smallestChange.toFixed(1)}–${largestChange.toFixed(1)}× longer in this run. The solver had more cells to update and needed more sweeps to meet the same residual limit. This compares problem sizes, not processor counts.`;
    document.querySelector("#bench-results").hidden = false;
    benchStatus.textContent = "Completed: median time per solve from three batches per grid, using the Release build.";
  } catch (error) {
    benchStatus.classList.add("error");
    benchStatus.textContent = error.message;
  } finally {
    benchButton.disabled = false;
  }
}

async function runOpenMPComparison() {
  openmpButton.disabled = true;
  openmpStatus.classList.remove("error");
  const started = Date.now();
  const timer = setInterval(() => {
    openmpStatus.textContent = `Running serial and OpenMP cases… ${Math.floor((Date.now() - started) / 1000)}s`;
  }, 1000);
  openmpStatus.textContent = "Running serial and OpenMP cases…";

  try {
    const response = await fetch("/api/openmp", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The comparison failed.");

    const body = document.querySelector("#openmp-body");
    body.replaceChildren();
    data.results.forEach((result) => {
      const isSerial = result.implementation === "serial";
      const values = [
        isSerial ? "Serial" : "OpenMP",
        String(result.threads),
        formatSeconds(result.median_seconds),
        isSerial ? "baseline" : `${result.speedup.toFixed(2)}×`,
        isSerial ? "—" : `${(100 * result.efficiency).toFixed(1)}%`,
      ];
      const row = document.createElement("tr");
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      body.append(row);
    });

    document.querySelector("#openmp-note").textContent =
      `Maximum difference across saved fields: ${data.max_field_difference.toExponential(2)}. ` +
      `All cases used ${data.grid_size} × ${data.grid_size} points and ${data.steps} timesteps. ` +
      "These are local measurements; background load and thread placement can change the timing.";
    document.querySelector("#openmp-results").hidden = false;
    openmpStatus.textContent = "Completed: three runs per configuration, all fields checked against serial.";
  } catch (error) {
    openmpStatus.classList.add("error");
    openmpStatus.textContent = error.message;
  } finally {
    clearInterval(timer);
    openmpButton.disabled = false;
  }
}

async function runMPIComparison() {
  mpiButton.disabled = true;
  mpiStatus.classList.remove("error");
  const started = Date.now();
  const timer = setInterval(() => {
    mpiStatus.textContent = `Running serial and MPI cases… ${Math.floor((Date.now() - started) / 1000)}s`;
  }, 1000);
  mpiStatus.textContent = "Running serial and MPI cases…";

  try {
    const response = await fetch("/api/mpi", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The MPI comparison failed.");

    const body = document.querySelector("#mpi-body");
    body.replaceChildren();
    data.results.forEach((result) => {
      const isSerial = result.implementation === "serial";
      const values = [
        isSerial ? "Serial" : "MPI",
        String(result.ranks),
        formatSeconds(result.median_seconds),
        isSerial ? "baseline" : `${result.speedup.toFixed(2)}×`,
        isSerial ? "—" : `${(100 * result.efficiency).toFixed(1)}%`,
      ];
      const row = document.createElement("tr");
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      body.append(row);
    });

    document.querySelector("#mpi-note").textContent =
      `Maximum difference across saved fields: ${data.max_field_difference.toExponential(2)}. ` +
      `All cases used ${data.grid_size} × ${data.grid_size} points and ${data.steps} timesteps. ` +
      "Communication and global reductions are included in the MPI runtime; process launch and file output are excluded.";
    document.querySelector("#mpi-results").hidden = false;
    mpiStatus.textContent = "Completed: three runs per configuration, all fields checked against serial.";
  } catch (error) {
    mpiStatus.classList.add("error");
    mpiStatus.textContent = error.message;
  } finally {
    clearInterval(timer);
    mpiButton.disabled = false;
  }
}

async function runSimulation() {
  runButton.disabled = true;
  runStatus.classList.remove("error");
  const started = Date.now();
  const timer = setInterval(() => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    runStatus.textContent = `Computing the flow and plots… ${seconds}s`;
  }, 1000);
  runStatus.textContent = "Computing the flow and plots…";

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        grid: Number(gridInput.value),
        reynolds: Number(reynoldsInput.value),
        steps: Number(stepsInput.value),
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The run failed.");
    history.unshift(result);
    showRun(result);
    updateHistory();
    runStatus.textContent = "Completed. The runtime above excludes plotting and file output.";
  } catch (error) {
    runStatus.classList.add("error");
    runStatus.textContent = error.message;
  } finally {
    clearInterval(timer);
    runButton.disabled = false;
  }
}

reynoldsInput.addEventListener("input", updateLabels);
stepsInput.addEventListener("input", updateLabels);
runButton.addEventListener("click", runSimulation);
benchButton.addEventListener("click", runBenchmark);
openmpButton.addEventListener("click", runOpenMPComparison);
mpiButton.addEventListener("click", runMPIComparison);
document.querySelectorAll(".presets button").forEach((button) => {
  button.addEventListener("click", () => {
    gridInput.value = button.dataset.grid;
    reynoldsInput.value = button.dataset.re;
    stepsInput.value = button.dataset.steps;
    updateLabels();
  });
});
document.querySelectorAll(".image-tabs button").forEach((button) => {
  button.addEventListener("click", () => {
    currentImage = button.dataset.image;
    showImage();
  });
});
updateLabels();
