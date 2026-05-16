// --- DOM Elements ---
const rollingWindowSlider = document.getElementById('rollingWindow');
const windowVal = document.getElementById('windowVal');
const stlSeasonalSlider = document.getElementById('stlSeasonal');
const stlVal = document.getElementById('stlVal');
const nlagsSlider = document.getElementById('nlags');
const lagVal = document.getElementById('lagVal');
const runBtn = document.getElementById('runAnalysisBtn');

// --- Global Chart Instances ---
let trendChartInstance = null;
let stlChartInstance = null;
let acfChartInstance = null;

// --- Mock Data Generators (Simulating your Python Backend) ---
function generateRawData(points = 100) {
    const data = [];
    let base = 50;
    for (let i = 0; i < points; i++) {
        // Sine wave for seasonality + random noise for variance + slight upward trend
        const noise = (Math.random() - 0.5) * 30;
        const season = Math.sin(i / 5) * 20;
        const trend = i * 0.2;
        data.push(Math.max(0, base + season + noise + trend));
    }
    return data;
}

function calculateMovingAverage(data, windowSize) {
    const ma = [];
    for (let i = 0; i < data.length; i++) {
        if (i < windowSize - 1) {
            ma.push(null); // Not enough data for rolling window
        } else {
            let sum = 0;
            for (let j = 0; j < windowSize; j++) {
                sum += data[i - j];
            }
            ma.push(sum / windowSize);
        }
    }
    return ma;
}

// --- Chart Rendering Engines ---
function renderTrendChart(rawData, windowSize) {
    const ctx = document.getElementById('trendChart').getContext('2d');
    const maData = calculateMovingAverage(rawData, windowSize);
    const labels = Array.from({length: rawData.length}, (_, i) => `Day ${i+1}`);

    if (trendChartInstance) trendChartInstance.destroy();

    trendChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Raw Data',
                    data: rawData,
                    borderColor: 'rgba(128, 128, 128, 0.4)',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: `Rolling Trend (${windowSize}-Period)`,
                    data: maData,
                    borderColor: 'darkorange',
                    borderWidth: 2.5,
                    pointRadius: 0,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false }
        }
    });
}

function renderMockSTL(rawData, seasonalWindow) {
    const ctx = document.getElementById('stlChart').getContext('2d');
    const labels = Array.from({length: rawData.length}, (_, i) => `Day ${i+1}`);

    // Simulating the extraction of a seasonal signal based on the window
    const seasonalData = rawData.map((val, i) => Math.sin(i / (seasonalWindow/4)) * 20);

    if (stlChartInstance) stlChartInstance.destroy();

    stlChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Extracted Seasonal Component',
                data: seasonalData,
                borderColor: 'darkblue',
                backgroundColor: 'rgba(0, 0, 139, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}

function renderCorrelogram(lags) {
    const ctx = document.getElementById('acfChart').getContext('2d');
    const labels = Array.from({length: lags}, (_, i) => i);

    // Simulating ACF decay
    const acfData = labels.map(lag => {
        if (lag === 0) return 1.0;
        return (Math.cos(lag / 2) * Math.exp(-lag / 20)) + (Math.random() * 0.1 - 0.05);
    });

    if (acfChartInstance) acfChartInstance.destroy();

    acfChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Autocorrelation',
                data: acfData,
                backgroundColor: 'teal',
                barPercentage: 0.1 // Makes it look like a stem/lollipop plot
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { min: -0.5, max: 1, title: { display: true, text: 'ACF' } },
                x: { title: { display: true, text: 'Lag' } }
            }
        }
    });
}

// --- Event Listeners & Interaction Logic ---

// Update labels dynamically when sliders move
rollingWindowSlider.addEventListener('input', (e) => windowVal.textContent = e.target.value);
nlagsSlider.addEventListener('input', (e) => lagVal.textContent = e.target.value);

// Force STL Seasonal to be an odd number natively in JS, just like the Python script
stlSeasonalSlider.addEventListener('input', (e) => {
    let val = parseInt(e.target.value);
    if (val % 2 === 0) val += 1;
    e.target.value = val;
    stlVal.textContent = val;
});

// Main execution function
function executePipeline() {
    // 1. Gather Variables
    const windowSize = parseInt(rollingWindowSlider.value);
    const seasonalWindow = parseInt(stlSeasonalSlider.value);
    const lags = parseInt(nlagsSlider.value);

    // 2. Generate Base Data
    const rawData = generateRawData(150);

    // 3. Render Visualizations
    renderTrendChart(rawData, windowSize);
    renderMockSTL(rawData, seasonalWindow);
    renderCorrelogram(lags);
}

// Button Click
runBtn.addEventListener('click', executePipeline);

// Run once on load to populate the dashboard
window.addEventListener('DOMContentLoaded', executePipeline);