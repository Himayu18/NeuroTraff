// Function to show specific section
function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
        section.style.display = 'none';
    });

    // Show selected section
    const targetSection = document.getElementById(sectionId);
    targetSection.style.display = 'block';
    targetSection.classList.add('active');

    // Scroll to top
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Function to handle dropdown selection
function selectFromDropdown() {
    const dropdown = document.getElementById('roadDropdown');

    if (dropdown.value) {
        predictTraffic();
    }
}
function selectRoad(roadName) {
            document.getElementById('roadDropdown').value = roadName;
            predictTraffic();
        }

function updateVehicleRecommendation(trafficLevel) {
    const vehicleIcon = document.getElementById('vehicleIcon');
    const vehicleType = document.getElementById('vehicleType');
    const vehicleStatus = document.getElementById('vehicleStatus');
    const vehicleProgress = document.getElementById('vehicleProgress');
    const vehicleIndicator = document.getElementById('vehicleIndicator');
    
    let vehicles = '';
    let icon = '';
    let statusText = '';
    let progressWidth = 0;
    let progressClass = '';
    
    if (trafficLevel === 'low') {
        vehicles = 'Car / Rickshaw';
        icon = '🚗';
        statusText = 'Roads are clear - comfortable travel options available';
        progressWidth = 33;
        progressClass = 'traffic-low';
        vehicleIndicator.className = 'change-indicator change-positive';
    } else if (trafficLevel === 'med' || trafficLevel === 'medium') {
        vehicles = 'Scooter / Bike';
        icon = '🛵';
        statusText = 'Moderate traffic - two-wheelers recommended';
        progressWidth = 66;
        progressClass = 'traffic-medium';
        vehicleIndicator.className = 'change-indicator change-neutral';
    } else if (trafficLevel === 'high') {
        vehicles = 'Walk / Bicycle';
        icon = '🚶';
        statusText = 'Heavy congestion - active transport recommended';
        progressWidth = 100;
        progressClass = 'traffic-high';
        vehicleIndicator.className = 'change-indicator change-negative';
    }
    
    vehicleIcon.textContent = icon;
    vehicleType.textContent = vehicles;
    vehicleStatus.textContent = statusText;
    vehicleProgress.style.width = progressWidth + '%';
    vehicleProgress.className = 'progress-fill ' + progressClass;
}

// Store recent predictions (max 5)
let recentPredictions = [];

function addRecentPrediction(roadName, trafficLevel, clearTime) {
    const timestamp = new Date();
    
    // Create prediction object
    const prediction = {
        road: roadName,
        level: trafficLevel,
        time: clearTime || '--',
        timestamp: timestamp
    };
    
    // Add to beginning of array
    recentPredictions.unshift(prediction);
    
    // Keep only last 5 predictions
    if (recentPredictions.length > 5) {
        recentPredictions.pop();
    }
    
    // Update display
    displayRecentPredictions();
}

function displayRecentPredictions() {
    const predictionsList = document.getElementById('predictionsList');
    const noDataMessage = document.getElementById('noDataMessage');
    
    // If no predictions, show no-data message
    if (recentPredictions.length === 0) {
        noDataMessage.style.display = 'block';
        return;
    }
    
    // Hide no-data message
    noDataMessage.style.display = 'none';
    
    // Clear existing predictions (except no-data message)
    const existingItems = predictionsList.querySelectorAll('.prediction-item');
    existingItems.forEach(item => item.remove());
    
    // Add each prediction
    recentPredictions.forEach(pred => {
        const predItem = createPredictionItem(pred);
        predictionsList.appendChild(predItem);
    });
}

function createPredictionItem(prediction) {
    const div = document.createElement('div');
    
    // Determine CSS class based on traffic level
    let predictionClass = 'prediction-moderate';
    let levelText = 'Moderate Traffic';
    
    if (prediction.level.toLowerCase() === 'low') {
        predictionClass = 'prediction-light';
        levelText = 'Light Traffic';
    } else if (prediction.level.toLowerCase() === 'high') {
        predictionClass = 'prediction-heavy';
        levelText = 'Heavy Traffic';
    } else if (prediction.level.toLowerCase() === 'medium' || prediction.level.toLowerCase() === 'med') {
        predictionClass = 'prediction-moderate';
        levelText = 'Moderate Traffic';
    }
    
    div.className = `prediction-item ${predictionClass}`;
    div.innerHTML = `
        <div class="prediction-road">${prediction.road}</div>
        <div class="prediction-details">
            <span>${levelText}</span>
            <span>${prediction.time} min to clear</span>
        </div>
    `;
    
    return div;
}

function predictTraffic() {
    const dropdown = document.getElementById('roadDropdown');
    const roadName = dropdown.value;

    if (!roadName.trim()) return;

    // Show dashboard with animation
    const dashboard = document.getElementById('dashboard');
    dashboard.style.display = 'grid';
    dashboard.style.opacity = '0';
    dashboard.style.transform = 'translateY(20px)';

    setTimeout(() => {
        dashboard.style.transition = 'all 0.5s ease';
        dashboard.style.opacity = '1';
        dashboard.style.transform = 'translateY(0)';
    }, 100);

    // Scroll to dashboard
    dashboard.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });

    // Show success feedback on button
    const button = document.querySelector('.search-btn');
    const originalText = button.textContent;
    button.textContent = '✓ Predicting...';
    button.style.background = 'linear-gradient(45deg, #2ed573, #20bf6b)';

    setTimeout(() => {
        button.textContent = originalText;
        button.style.background = 'linear-gradient(45deg, #00d4ff, #0099cc, #ff6b6b)';
    }, 2000);

    // Fetch traffic prediction from backend
    fetch('/selected_road', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            road: roadName
        })
    })
        .then(response => response.json())
        .then(data => {
            console.log('API response:', data);

            const trafficChange = document.getElementById('trafficChange');
            const trafficLevel = document.getElementById('trafficLevel');
            const trafficProgress = document.getElementById('trafficProgress');

            if (data.traffic_level) {
                // Display traffic level text
                trafficChange.textContent = data.traffic_level;
                trafficLevel.textContent = data.traffic_level;
                // Update styling based on traffic level
                let parentDiv = trafficChange.parentElement;
                switch (data.traffic_level.toLowerCase()) {
                    case 'low':
                        parentDiv.className = 'change-indicator change-positive';
                        trafficProgress.style.width = '30%';
                        trafficProgress.className = 'progress-fill traffic-low';
                        break;
                    case 'medium':
                        parentDiv.className = 'change-indicator change-neutral';
                        trafficProgress.style.width = '60%';
                        trafficProgress.className = 'progress-fill traffic-medium';
                        break;
                    case 'high':
                        parentDiv.className = 'change-indicator change-negative';
                        trafficProgress.style.width = '90%';
                        trafficProgress.className = 'progress-fill traffic-high';
                        break;
                    default:
                        parentDiv.className = 'change-indicator';
                        trafficProgress.style.width = '0%';
                        trafficProgress.className = 'progress-fill';
                        break;
                }
                updateVehicleRecommendation(data.traffic_level.toLowerCase());
                addRecentPrediction(roadName, data.traffic_level, data.clear_time_estimate);
            } else {
                trafficChange.textContent = 'Waiting for prediction...';
                trafficChange.parentElement.className = 'change-indicator change-negative';
            }

            // Optional: Add other metric updates here
            const clearTimeElement = document.getElementById('clearTime');
            if (data.clear_time_estimate !== undefined && clearTimeElement) {
                clearTimeElement.textContent = `${data.clear_time_estimate} Minutes to Clear`;
            } else {
                clearTimeElement.textContent = '--';
            }
        })
        .catch(error => {
            console.error('Error fetching prediction:', error);
        });
}

// Star rating functionality
const starRating = document.querySelector('.star-rating');
const ratingText = document.getElementById('ratingText');

if (starRating) {
    starRating.addEventListener('change', function (e) {
        if (e.target.type === 'radio') {
            const rating = e.target.value;
            const ratingLabels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'];
            ratingText.textContent = `${ratingLabels[rating - 1]} (${rating} stars)`;
            ratingText.style.color = '#ffa502';
        }
    });
}

// Analytics Dashboard Functions
let peakHoursChart = null;

function loadAnalyticsDashboard() {
    const dropdown = document.getElementById('analyticsRoadDropdown');
    const roadName = dropdown.value;

    if (!roadName) return;

    // Display selected road name
    document.getElementById('selectedRoadName').textContent = roadName;

    // Show dashboard with fade-in animation
    const dashboard = document.getElementById('analyticsDashboard');
    dashboard.style.display = 'block';
    dashboard.style.opacity = '0';

    setTimeout(() => {
        dashboard.style.transition = 'opacity 0.5s ease';
        dashboard.style.opacity = '1';
    }, 100);

    // Scroll to dashboard
    setTimeout(() => {
        dashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);

    // Reset to loading state
    resetAnalyticsDashboard();

    // Fetch analytics data from Flask backend
    fetchAnalyticsData(roadName);
}

function resetAnalyticsDashboard() {
    // Reset metric values to loading state
    document.getElementById('carbonEmission').innerHTML = '<span class="loading-shimmer">--</span>';
    document.getElementById('travelVariability').innerHTML = '<span class="loading-shimmer">--</span>';
    document.getElementById('gaugeValue').textContent = '--';
    document.getElementById('gaugeValue').classList.add('loading-shimmer');
    document.getElementById('gaugeLabel').textContent = 'Loading...';

    document.getElementById('carbonTrend').innerHTML = '⏳ Loading data...';
    document.getElementById('variabilityTrend').innerHTML = '⏳ Loading data...';

    // Reset gauge
    const gaugeFill = document.getElementById('gaugeFill');
    gaugeFill.style.strokeDashoffset = '251.2';

    // Reset badges
    document.querySelectorAll('.badge').forEach(b => b.classList.remove('active'));

    // Show chart loading states
    document.getElementById('peakChartLoading').style.display = 'flex';
    document.getElementById('peakHoursChart').style.display = 'none';
}

function fetchAnalyticsData(roadName) {
    // Call Flask backend API
    fetch('/analytics_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ road: roadName })
    })
        .then(response => response.json())
        .then(data => {
            console.log('Analytics data received:', data);
            updateAnalyticsDisplay(data);
        })
        .catch(error => {
            console.error('Error fetching analytics data:', error);
            showAnalyticsError();
        });
}

function updateAnalyticsDisplay(data) {
    // Update Carbon Emission
    if (data.carbon_emission !== undefined) {
        document.getElementById('carbonEmission').textContent = data.carbon_emission;
        const carbonTrend = data.carbon_emission > 400 ? '↑ Higher than average' : '↓ Lower than average';
        document.getElementById('carbonTrend').innerHTML = carbonTrend;
    }

    // Update Travel Time Variability
    if (data.travel_variability !== undefined) {
    // Convert CV to percentage
    let variabilityPercent = (data.travel_variability * 100);

    // Cap at 100%
    if (variabilityPercent > 100) {
        variabilityPercent = 100;
    }

    // Round to 1 decimal place
    variabilityPercent = variabilityPercent.toFixed(1);

    // Display ±percentage
    document.getElementById('travelVariability').textContent = `±${variabilityPercent}%`;

    // Set trend label (you can keep the 15% threshold or adjust)
    const variabilityTrend = variabilityPercent > 15 ? '↑ High variance' : '↓ Low variance';
    document.getElementById('variabilityTrend').innerHTML = variabilityTrend;
}



    // Update Congestion Gauge
    if (data.congestion_rate !== undefined) {
        updateCongestionGauge(data.congestion_rate);
    }

    // Update Peak Hours Chart
    if (data.peak_hours_data) {
        updatePeakHoursChart(data.peak_hours_data);
    }
}

function showAnalyticsError() {
    document.getElementById('carbonEmission').textContent = 'Error';
    document.getElementById('travelVariability').textContent = 'Error';
    document.getElementById('carbonTrend').innerHTML = '⚠️ Failed to load data';
    document.getElementById('variabilityTrend').innerHTML = '⚠️ Failed to load data';
    document.getElementById('gaugeValue').textContent = 'Error';
    document.getElementById('gaugeLabel').textContent = 'Failed to load';
}

function updateCongestionGauge(percentage) {
    const gaugeFill = document.getElementById('gaugeFill');
    const gaugeValue = document.getElementById('gaugeValue');
    const gaugeLabel = document.getElementById('gaugeLabel');
    const maxDashOffset = 251.2;
    const dashOffset = maxDashOffset - (maxDashOffset * percentage / 100);

    // Remove loading shimmer
    gaugeValue.classList.remove('loading-shimmer');

    // Update gauge with animation
    setTimeout(() => {
        gaugeFill.style.strokeDashoffset = dashOffset;
        gaugeValue.textContent = percentage + '%';
        gaugeLabel.textContent = 'Current Level';
    }, 300);

    // Update badges
    document.querySelectorAll('.badge').forEach(b => b.classList.remove('active'));
    if (percentage < 40) {
        document.getElementById('badgeLow').classList.add('active');
    } else if (percentage < 70) {
        document.getElementById('badgeMed').classList.add('active');
    } else {
        document.getElementById('badgeHigh').classList.add('active');
    }
}

function updatePeakHoursChart(peakData) {
    // Hide loading, show chart
    document.getElementById('peakChartLoading').style.display = 'none';
    document.getElementById('peakHoursChart').style.display = 'block';

    const ctx = document.getElementById('peakHoursChart');

    if (peakHoursChart) {
        peakHoursChart.destroy();
    }

    // Expected format: { hours: [...], speeds: [...] }
    const hours = peakData.map(item => `Hour ${item.Hour}`);
    const speeds = peakData.map(item => item.mean_currentSpeed);

    peakHoursChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: hours,
            datasets: [{
                label: 'Average Speed (km/h)',
                data: speeds,
                backgroundColor: speeds.map(speed => {
                    if (speed < 30) return 'rgba(255, 71, 87, 0.8)';
                    if (speed < 45) return 'rgba(255, 165, 2, 0.8)';
                    return 'rgba(46, 213, 115, 0.8)';
                }),
                borderColor: speeds.map(speed => {
                    if (speed < 30) return '#ff4757';
                    if (speed < 45) return '#ffa502';
                    return '#2ed573';
                }),
                borderWidth: 2,
                borderRadius: 10,
                hoverBackgroundColor: speeds.map(speed => {
                    if (speed < 30) return 'rgba(255, 71, 87, 1)';
                    if (speed < 45) return 'rgba(255, 165, 2, 1)';
                    return 'rgba(46, 213, 115, 1)';
                })
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#00d4ff',
                        font: {
                            size: 13,
                            weight: '600'
                        },
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'rectRounded'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 10, 30, 0.95)',
                    titleColor: '#00d4ff',
                    bodyColor: '#fff',
                    borderColor: '#00d4ff',
                    borderWidth: 2,
                    padding: 15,
                    displayColors: true,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    boxPadding: 6,
                    cornerRadius: 10,
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += context.parsed.y + ' km/h';
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: Math.max(...speeds) + 10,
                    ticks: {
                        color: '#a0a0a0',
                        font: {
                            size: 12,
                            weight: '500'
                        },
                        callback: function (value) {
                            return value + ' km/h';
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.08)',
                        lineWidth: 1
                    },
                    border: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#a0a0a0',
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    },
                    grid: {
                        display: false
                    },
                    border: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// Form submission handling
const feedbackForm = document.getElementById('feedbackForm');

if (feedbackForm) {
    feedbackForm.addEventListener('submit', function (e) {
        e.preventDefault();

        // Gather form data
        const formData = {
            name: document.getElementById('userName').value,
            road: document.getElementById('feedbackRoad').value,
            trafficCondition: document.getElementById('trafficCondition').value,
            delay: document.getElementById('delayMinutes').value,
            weather: document.getElementById('weatherCondition').value,
            description: document.getElementById('description').value,
            rating: document.querySelector('input[name="rating"]:checked').value
        };

        console.log('Feedback submitted:', formData);

        // Send to backend
        fetch('/submit_feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        })
            .then(response => response.json())
            .then(data => {
                console.log('Feedback response:', data);

                // Show success message
                document.getElementById('feedbackForm').style.display = 'none';
                document.getElementById('feedbackSuccess').style.display = 'block';

                // Reset form after 3 seconds
                setTimeout(() => {
                    document.getElementById('feedbackForm').reset();
                    document.getElementById('feedbackForm').style.display = 'block';
                    document.getElementById('feedbackSuccess').style.display = 'none';
                    ratingText.textContent = 'Select a rating';
                    ratingText.style.color = '#a0a0a0';
                }, 3000);
            })
            .catch(error => {
                console.error('Error submitting feedback:', error);
                alert('Error submitting feedback. Please try again.');
            });
    });
}

// Create floating particles
function createParticle() {
    const particle = document.createElement('div');
    particle.className = 'particle';
    particle.style.left = Math.random() * 100 + '%';
    particle.style.width = particle.style.height = Math.random() * 6 + 3 + 'px';
    particle.style.animationDuration = (Math.random() * 15 + 10) + 's';
    particle.style.animationDelay = Math.random() * 5 + 's';

    // Random colors for particles
    const colors = ['#00d4ff', '#ff6b6b', '#4ecdc4', '#ffa726', '#ab47bc'];
    particle.style.background =
        `radial-gradient(circle, ${colors[Math.floor(Math.random() * colors.length)]}, transparent)`;

    document.getElementById('particles').appendChild(particle);

    setTimeout(() => {
        particle.remove();
    }, 25000);
}

// Generate more particles
setInterval(createParticle, 1500);

// Initialize with more particles
for (let i = 0; i < 8; i++) {
    setTimeout(createParticle, i * 500);
}

// Add mouse trail effect
document.addEventListener('mousemove', function (e) {
    if (Math.random() > 0.85) { // 15% chance
        const trail = document.createElement('div');
        trail.style.position = 'fixed';
        trail.style.left = e.clientX + 'px';
        trail.style.top = e.clientY + 'px';
        trail.style.width = '4px';
        trail.style.height = '4px';
        trail.style.background = 'radial-gradient(circle, #00d4ff, transparent)';
        trail.style.borderRadius = '50%';
        trail.style.pointerEvents = 'none';
        trail.style.zIndex = '1000';
        trail.style.animation = 'fadeOut 1s ease-out forwards';

        document.body.appendChild(trail);

        setTimeout(() => {
            trail.remove();
        }, 1000);
    }
});

// What-if analysis
document.addEventListener('DOMContentLoaded', function () {
    initializeWhatIfControls();
});

function initializeWhatIfControls() {
    // Time slider - Update value in real-time
    const timeSlider = document.getElementById('timeSlider');
    const timeValue = document.getElementById('timeValue');

    if (timeSlider && timeValue) {
        timeValue.textContent = timeSlider.value.padStart(2, '0') + ':00';

        timeSlider.oninput = function () {
            const hour = this.value.padStart(2, '0');
            timeValue.textContent = hour + ':00';
        };
    }

    // Vehicle slider - Update value in real-time
    const vehicleSlider = document.getElementById('vehicleSlider');
    const vehicleValue = document.getElementById('vehicleValue');

    if (vehicleSlider && vehicleValue) {
        vehicleValue.textContent = 'Normal (' + vehicleSlider.value + '%)';

        vehicleSlider.oninput = function () {
            const value = this.value;
            let label = '';

            if (value < 75) {
                label = 'Low (' + value + '%)';
            } else if (value <= 125) {
                label = 'Normal (' + value + '%)';
            } else if (value <= 175) {
                label = 'High (' + value + '%)';
            } else {
                label = 'Very High (' + value + '%)';
            }

            vehicleValue.textContent = label;
        };
    }

    // Accident checkbox and severity
    const accidentCheckbox = document.getElementById('accidentCheckbox');
    const severityGroup = document.getElementById('severityGroup');

    if (accidentCheckbox && severityGroup) {
        accidentCheckbox.addEventListener('change', function () {
            if (this.checked) {
                severityGroup.style.display = 'block';
            } else {
                severityGroup.style.display = 'none';
            }
        });
    }

    // Severity buttons - Only one active at a time
    const severityBtns = document.querySelectorAll('.severity-btn');
    severityBtns.forEach(function (btn) {
        btn.addEventListener('click', function () {
            severityBtns.forEach(function (b) {
                b.classList.remove('active');
            });
            this.classList.add('active');
        });
    });

    // Day type buttons - Only one active at a time
    const dayTypeBtns = document.querySelectorAll('.day-type-btn');
    dayTypeBtns.forEach(function (btn) {
        btn.addEventListener('click', function () {
            dayTypeBtns.forEach(function (b) {
                b.classList.remove('active');
            });
            this.classList.add('active');
        });
    });
}

// Analyze What-If Scenario
function analyzeWhatIf() {
    const timeOfDay = document.getElementById('timeSlider').value;
    const isRain = document.getElementById('rainCheckbox').checked;
    const vehicleVolume = document.getElementById('vehicleSlider').value;
    const hasAccident = document.getElementById('accidentCheckbox').checked;

    let accidentSeverity = null;
    if (hasAccident) {
        const activeSeverity = document.querySelector('.severity-btn.active');
        accidentSeverity = activeSeverity ? activeSeverity.getAttribute('data-severity') : 'severe';
    }

    const activeDayType = document.querySelector('.day-type-btn.active');
    const dayType = activeDayType ? activeDayType.getAttribute('data-day') : 'weekday';

    const selectedRoad = document.getElementById('roadSelector').value;

    if (!selectedRoad) {
        alert('Please select a road first!');
        return;
    }

    const whatIfData = {
        road: selectedRoad,
        time_of_day: parseInt(timeOfDay),
        is_rain: isRain,
        vehicle_volume: parseInt(vehicleVolume),
        has_accident: hasAccident,
        accident_severity: accidentSeverity,
        day_type: dayType
    };

    // Show loading state
    const analyzeBtn = document.querySelector('.analyze-btn');
    const originalText = analyzeBtn.innerHTML;
    analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span> Analyzing...';
    analyzeBtn.disabled = true;

    fetch('/analyze_whatif', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(whatIfData)
    })
        .then(response => response.json())
        .then(data => {
            displayWhatIfResult(data.traffic_level);
            analyzeBtn.innerHTML = originalText;
            analyzeBtn.disabled = false;
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error analyzing what-if scenario. Please try again.');
            analyzeBtn.innerHTML = originalText;
            analyzeBtn.disabled = false;
        });
}

function displayWhatIfResult(trafficLevel) {
    const resultSection = document.getElementById('whatIfResult');
    const resultLevelEl = document.getElementById('resultLevel');
    const resultDescEl = document.getElementById('resultDescription');

    const level = trafficLevel.toLowerCase();

    resultLevelEl.textContent = trafficLevel.charAt(0).toUpperCase() + trafficLevel.slice(1);
    resultLevelEl.className = 'result-level ' + level;

    const descriptions = {
        'low': 'Traffic is expected to be light. Good time to travel!',
        'med': 'Moderate traffic expected. Allow extra time for your journey.',
        'medium': 'Moderate traffic expected. Allow extra time for your journey.',
        'high': 'Heavy traffic predicted. Consider alternative routes or times.'
    };

    resultDescEl.textContent = descriptions[level] || 'Traffic prediction complete.';

    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}



// About Page JavaScript - Optional Enhancements

// Animate elements on scroll (Intersection Observer)
document.addEventListener('DOMContentLoaded', function() {
    
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.2,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe all content cards
    const cards = document.querySelectorAll('.content-card');
    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });

    // Animate workflow steps sequentially
    const workflowSteps = document.querySelectorAll('.workflow-step');
    workflowSteps.forEach((step, index) => {
        step.style.opacity = '0';
        step.style.transform = 'translateY(30px)';
        step.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        step.style.transitionDelay = `${index * 0.15}s`;
        
        const stepObserver = new IntersectionObserver(function(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);
        
        stepObserver.observe(step);
    });

    // Animate tech badges
    const techBadges = document.querySelectorAll('.tech-badge');
    techBadges.forEach((badge, index) => {
        badge.style.opacity = '0';
        badge.style.transform = 'scale(0.8)';
        badge.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
        badge.style.transitionDelay = `${(index % 5) * 0.1}s`;
        
        const badgeObserver = new IntersectionObserver(function(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'scale(1)';
                }
            });
        }, observerOptions);
        
        badgeObserver.observe(badge);
    });

    // Animate stat items
    const statItems = document.querySelectorAll('.stat-item');
    statItems.forEach((stat, index) => {
        stat.style.opacity = '0';
        stat.style.transform = 'scale(0.8)';
        stat.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        stat.style.transitionDelay = `${index * 0.2}s`;
        
        const statObserver = new IntersectionObserver(function(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'scale(1)';
                }
            });
        }, observerOptions);
        
        statObserver.observe(stat);
    });

    // Add click effect to tech badges
    techBadges.forEach(badge => {
        badge.addEventListener('click', function() {
            // Create ripple effect
            const ripple = document.createElement('span');
            ripple.style.position = 'absolute';
            ripple.style.width = '20px';
            ripple.style.height = '20px';
            ripple.style.background = 'rgba(255, 255, 255, 0.5)';
            ripple.style.borderRadius = '50%';
            ripple.style.transform = 'scale(0)';
            ripple.style.animation = 'ripple 0.6s ease-out';
            ripple.style.pointerEvents = 'none';
            
            this.style.position = 'relative';
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });

    // Add CSS for ripple animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);

    // Add particle effect on tech badge hover
    techBadges.forEach(badge => {
        badge.addEventListener('mouseenter', function() {
            for (let i = 0; i < 3; i++) {
                setTimeout(() => {
                    createBadgeParticle(this);
                }, i * 100);
            }
        });
    });

    function createBadgeParticle(badge) {
        const particle = document.createElement('div');
        const rect = badge.getBoundingClientRect();
        
        particle.style.position = 'fixed';
        particle.style.left = (rect.left + rect.width / 2) + 'px';
        particle.style.top = (rect.top + rect.height / 2) + 'px';
        particle.style.width = '6px';
        particle.style.height = '6px';
        particle.style.borderRadius = '50%';
        particle.style.pointerEvents = 'none';
        particle.style.zIndex = '9999';
        
        // Get badge color
        const computedStyle = window.getComputedStyle(badge);
        const bgColor = computedStyle.backgroundColor;
        particle.style.background = bgColor;
        particle.style.boxShadow = `0 0 10px ${bgColor}`;
        
        const angle = Math.random() * Math.PI * 2;
        const velocity = 2 + Math.random() * 2;
        const vx = Math.cos(angle) * velocity;
        const vy = Math.sin(angle) * velocity;
        
        document.body.appendChild(particle);
        
        let x = 0;
        let y = 0;
        let opacity = 1;
        
        function animate() {
            x += vx;
            y += vy;
            opacity -= 0.02;
            
            particle.style.transform = `translate(${x}px, ${y}px)`;
            particle.style.opacity = opacity;
            
            if (opacity > 0) {
                requestAnimationFrame(animate);
            } else {
                particle.remove();
            }
        }
        
        animate();
    }

    // Add smooth scroll behavior for page title
    const pageTitle = document.querySelector('.page-title');
    if (pageTitle) {
        pageTitle.style.opacity = '0';
        pageTitle.style.transform = 'translateY(-20px)';
        
        setTimeout(() => {
            pageTitle.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
            pageTitle.style.opacity = '1';
            pageTitle.style.transform = 'translateY(0)';
        }, 100);
    }

    // Add counter animation for stat numbers (optional - if you want numerical stats)
    function animateCounter(element, target, duration = 2000) {
        let start = 0;
        const increment = target / (duration / 16);
        
        function updateCounter() {
            start += increment;
            if (start < target) {
                element.textContent = Math.floor(start);
                requestAnimationFrame(updateCounter);
            } else {
                element.textContent = target;
            }
        }
        
        updateCounter();
    }

    // Log page load
    console.log('NeuroTraff About Page Loaded Successfully! 🚀');
});

// Add parallax effect to story card background (optional)
window.addEventListener('scroll', function() {
    const scrolled = window.pageYOffset;
    const storyCard = document.querySelector('.story-card');
    
    if (storyCard) {
        const cardTop = storyCard.offsetTop;
        const cardHeight = storyCard.offsetHeight;
        
        if (scrolled > cardTop - window.innerHeight && scrolled < cardTop + cardHeight) {
            const parallax = (scrolled - cardTop) * 0.1;
            storyCard.style.backgroundPositionY = parallax + 'px';
        }
    }
});

// Add keyboard navigation for accessibility
document.addEventListener('keydown', function(e) {
    if (e.key === 'Tab') {
        const focusedElement = document.activeElement;
        if (focusedElement.classList.contains('tech-badge')) {
            focusedElement.style.outline = '3px solid #00d4ff';
            focusedElement.style.outlineOffset = '5px';
        }
    }
});

document.addEventListener('blur', function(e) {
    if (e.target.classList.contains('tech-badge')) {
        e.target.style.outline = 'none';
    }
}, true);


