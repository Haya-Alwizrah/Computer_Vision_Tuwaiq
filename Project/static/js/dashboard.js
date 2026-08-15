const lanes = ["North", "East", "South", "West"];

function updateDateTime() {
    const now = new Date();
    const date =
        now.toLocaleDateString("en-GB");
    const time =
        now.toLocaleTimeString("en-GB", {
            hour12: false
        });

    document.getElementById("date").textContent = date;
    document.getElementById("time").textContent = time;
}

setInterval(updateDateTime, 1000);
updateDateTime();

async function updateTrafficStatus() {
    try {
        const response = await fetch("/api/status");
        const data = await response.json();

        updateSignals(data.signals);

        if (data.emergency_active) {
            showEmergency(data.priority_lane);
        }else {
            hideEmergency();
        }
    }
    catch (error) {
        console.error(
            "Unable to update traffic status:",
            error
        );
    }
}

function updateSignals(signals) {
    lanes.forEach(lane => {
        const signal = signals[lane];

        document
            .querySelectorAll(`.light[data-lane="${lane}"]`)
            .forEach(light => {light.classList.remove("active");});

        const activeLight =
            document.querySelector(
                `.light[data-lane="${lane}"][data-color="${signal}"]`
            );

        if (activeLight) {
            activeLight.classList.add("active");
        }

        const text = document.getElementById(`signal-text-${lane}`);
        if (text) {
            text.textContent = signal;

            if (signal === "GREEN") {
                text.style.color =
                    "var(--green)";
            }else if (signal === "YELLOW") {
                text.style.color =
                    "var(--yellow)";
            }else {
                text.style.color =
                    "var(--red)";
            }
        }
    });
}

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

updateTrafficStatus();

setInterval(updateTrafficStatus, 1000);