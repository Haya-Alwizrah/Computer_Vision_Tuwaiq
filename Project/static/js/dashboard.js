const lanes = ["North", "East", "South", "West"];

// DATE / TIME
function updateDateTime() {
    const now = new Date();
    const date = now.toLocaleDateString("en-GB");
    const time = now.toLocaleTimeString("en-GB", {hour12: false});

    document.getElementById("date").textContent = date;
    document.getElementById("time").textContent = time;
}

setInterval(updateDateTime, 1000);
updateDateTime();

// SIMULATION CONTROL
async function startSimulation() {
    try {
        const response = await fetch("/api/start", {method: "POST"});
        const data = await response.json();

        console.log(data);
        updateSystemButtons(true);
        updateTrafficStatus();

    } catch (error) {
        console.error("Unable to start simulation:", error);
    }
}

async function stopSimulation() {
    try {
        const response = await fetch("/api/stop", {method: "POST"});
        const data = await response.json();

        console.log(data);
        updateSystemButtons(false);
        updateTrafficStatus();

    } catch (error) {
        console.error("Unable to stop simulation:", error);
    }
}

function updateSystemButtons(running) {
    const startButton = document.getElementById("start-btn");
    const stopButton = document.getElementById("stop-btn");
    const systemStatus = document.getElementById("system-status");

    if (running) {
        startButton.disabled = true;
        stopButton.disabled = false;
        systemStatus.textContent = "RUNNING";
    } else {
        startButton.disabled = false;
        stopButton.disabled = true;
        systemStatus.textContent = "STOPPED";
    }
}

// TRAFFIC STATUS
async function updateTrafficStatus() {
    try {
        const response = await fetch("/api/status");
        const data = await response.json();

        updateSignals(data.signals);

        if (data.emergency_active) {
            showEmergency(data.priority_lane);
        } else {
            hideEmergency();
        }

        updateSystemButtons(data.simulation_running);

    } catch (error) {
        console.error("Unable to update traffic status:", error);
    }
}

// SIGNALS
function updateSignals(signals) {
    lanes.forEach(lane => {
        const signal = signals[lane];

        document
            .querySelectorAll(`.light[data-lane="${lane}"]`)
            .forEach(light => {
                light.classList.remove("active");
            });

        const activeLight = document.querySelector(`.light[data-lane="${lane}"][data-color="${signal}"]`);

        if (activeLight) {
            activeLight.classList.add("active");
        }

        const text = document.getElementById(`signal-text-${lane}`);

        if (text) {
            text.textContent = signal;

            if (signal === "GREEN") {
                text.style.color = "var(--green)";

            } else if (signal === "YELLOW") {
                text.style.color = "var(--yellow)";

            } else {
                text.style.color = "var(--red)";
            }
        }
    });
}

// EMERGENCY
function showEmergency(lane) {
    const banner = document.getElementById("emergency-banner");
    const message = document.getElementById("emergency-message");

    message.textContent = `${lane} Approach — Ambulance and Emergency Siren Confirmed`;
    banner.classList.remove("hidden");
}

function hideEmergency() {
    document
        .getElementById("emergency-banner")
        .classList.add("hidden");
}

// INITIAL STATUS
updateTrafficStatus();
setInterval(updateTrafficStatus, 1000);