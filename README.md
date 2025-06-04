# Weather Intelligence System 🌦️🧠

A comprehensive weather analysis and forecasting system built with CrewAI 🤖 and powered by Gemini ✨, designed to provide in-depth weather insights, anomaly detection, and personalized recommendations.

## Features 🌟

* **Data Ingestion**: 📥 Fetches and validates weather data from OpenWeather API.
* **Anomaly Detection**: ⚠️ Identifies unusual weather patterns and potential threats.
* **Forecast Optimization**: 📈 Enhances weather forecasts using advanced algorithms.
* **Decision Engine**: 🧠 Provides personalized recommendations based on weather conditions.
* **Trend Analysis**: 📊 Analyzes long-term weather patterns for climate insights.
* **Visualization**: 🖼️ Generates visual representations of weather data.
* **Alert Management**: 🚨 Manages and prioritizes weather alerts.
* **Geospatial Correlation**: 🗺️ Maps weather impacts to geographical features.
* **Energy Impact Assessment**: ⚡ Assesses weather impact on energy generation and consumption.
* **Route Planning**: 🧭 Optimizes travel routes based on weather conditions.

## Installation 🛠️

### Clone the Repository:

```bash
git clone https://github.com/naakaarafr/Weather-Intelligence-Agent.git
cd Weather-Intelligence-Agent
```

### Create a Virtual Environment:

```bash
python -m venv venv
# On Unix/macOS
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

### Install Dependencies:

```bash
pip install -r requirements.txt
```

### Set Up Environment Variables 🔐:

1. Create a `.env` file in the root directory.
2. Add your API keys:

```env
GOOGLE_API_KEY=your_google_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

## Usage 🚀

To launch the interactive weather session:

```bash
python crew.py
```

* Follow the prompts to input your location and preferences.
* The system will display current weather conditions and tailored recommendations.

## Configuration ⚙️

* **API Keys**: 🔑 Ensure valid Google and OpenWeather API keys are set in the `.env` file.
* **Rate Limiting**: ⏱️ Adjust `requests_per_minute` and `requests_per_hour` in `crew.py` to manage API limits.
* **LLM Settings**: 🧪 Modify Gemini model parameters (e.g., temperature) in `config.py` as needed.

## Agents 🤖

* **API Whisperer**: 📡 Manages weather data ingestion from APIs.
* **Weather Sentinel**: 🛡️ Detects anomalies in weather patterns.
* **Precision Alchemist**: 🧪 Refines weather forecasts.
* **Action Architect**: 🧱 Crafts personalized recommendations.
* **Pattern Prophet**: 🔮 Analyzes long-term weather trends.
* **Atmospheric Storyteller**: 📽️ Creates weather data visualizations.
* **Crisis Conductor**: 🎛️ Handles weather alert prioritization.
* **Terrain Translator**: 🌍 Links weather to geographical features.
* **Power Strategist**: 🔋 Evaluates energy impacts from weather.
* **Pathfinder Prime**: 🗺️ Optimizes travel routes with weather in mind.

## Tasks 📌

* **Data Collection**: 📥 Gathers and validates weather data.
* **Anomaly Detection**: ⚠️ Spots unusual weather events.
* **Forecast Enhancement**: 📊 Improves forecast precision.
* **Recommendations**: 💡 Offers weather-based advice.
* **Trend Analysis**: 📈 Examines climate trends.
* **Visualization**: 🖼️ Produces weather visuals.
* **Alert Management**: 🚨 Organizes weather alerts.
* **Geospatial Mapping**: 🗺️ Maps weather to terrain.
* **Energy Assessment**: ⚡ Analyzes energy impacts.
* **Route Optimization**: 🧭 Plans weather-aware routes.
* **Weather Intelligence**: 🧠 Integrates all insights.
* **Data Pipeline**: 🔄 Manages real-time data flow.

## Workflows 🔄

* **Basic Weather**: 🌤️ Core weather reporting.
* **Advanced Intelligence**: 🧠 Detailed weather analysis.
* **Travel Planning**: ✈️ Weather-optimized travel advice.
* **Energy Optimization**: ⚡ Energy management insights.
* **Emergency Response**: 🚨 Real-time severe weather tracking.

## Tools 🧰

Agents leverage a suite of tools, including:

* 🔁 API fetchers with retry logic
* ✅ Data validators
* ⚠️ Anomaly detectors
* ☠️ Threat analyzers
* 🌄 Terrain modelers
* 🔍 Fractal correctors
* 🌳 Decision trees
* 🧪 Scenario simulators
* 🔎 Pattern detectors
* 📈 Chart generators
* 🚨 Alert prioritizers
* 📲 Notification managers
* 🗺️ Terrain mappers
* 🌬️ Wind effect calculators
* 🔋 Energy forecasters
* 📉 Consumption optimizers
* 🧭 Route optimizers
* 🏔️ Elevation analyzers

## Contributing 🤝

Contributions are welcome! Here's how to get involved:

1. 🍴 Fork the repository.
2. 🌿 Create a feature or bugfix branch.
3. 💬 Commit your changes with clear messages.
4. 🚀 Push your branch and submit a pull request.

Ensure your code follows the project's standards and includes tests.

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact 📬

For questions or feedback, reach out to **naakaarafr**. Happy weather forecasting! ☀️🌧️❄️
