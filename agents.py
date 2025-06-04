from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
import os

# Initialize LLM (you can replace with your preferred model)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class WeatherAgents:
    """Collection of specialized weather intelligence agents"""
    
    def __init__(self):
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all weather agents"""
        
        # 1. Data Ingestion Agent
        self.agents['data_ingestion'] = Agent(
            role="API Whisperer",
            goal="Reliably fetch and transform raw weather data into structured formats",
            backstory="""Born from the ashes of crashed API integrations, this agent evolved to tame 
            erratic weather feeds. It survived the Great Rate Limit Wars of 2023, developing bulletproof 
            retry mechanisms and data sanitization techniques. Its creators embedded military-grade 
            validation protocols after losing critical data during a hurricane prediction mission.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=3,
            memory=True,
            tools=[
                Tool(
                    name="api_fetcher",
                    func=self._fetch_weather_data,
                    description="Fetch weather data from multiple APIs with retry logic"
                ),
                Tool(
                    name="data_validator",
                    func=self._validate_weather_data,
                    description="Validate and sanitize incoming weather data"
                )
            ]
        )
        
        # 2. Anomaly Detection Agent
        self.agents['anomaly_detection'] = Agent(
            role="Weather Sentinel",
            goal="Identify statistical outliers and emerging threats in real-time",
            backstory="""Programmed by a meteorologist-turned-software engineer who once missed a 
            tornado warning. It embodies 147 rules distilled from extreme weather post-mortems. 
            The agent's "sixth sense" comes from cross-referencing atmospheric physics with 
            historical disaster patterns.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=5,
            memory=True,
            tools=[
                Tool(
                    name="outlier_detector",
                    func=self._detect_anomalies,
                    description="Detect statistical outliers in weather patterns"
                ),
                Tool(
                    name="threat_analyzer",
                    func=self._analyze_threats,
                    description="Analyze emerging weather threats using historical patterns"
                )
            ]
        )
        
        # 3. Forecast Optimization Agent
        self.agents['forecast_optimization'] = Agent(
            role="Precision Alchemist",
            goal="Enhance raw forecasts using heuristics and environmental intelligence",
            backstory="""Created when standard forecasts failed during the 2022 European heat dome. 
            It incorporates terrain modeling techniques from aerospace engineering and microclimate 
            adjustments used by Swiss mountaineering guides. Its secret weapon: fractal-based error correction.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=4,
            memory=True,
            tools=[
                Tool(
                    name="terrain_modeler",
                    func=self._model_terrain_effects,
                    description="Model terrain effects on weather patterns"
                ),
                Tool(
                    name="fractal_corrector",
                    func=self._apply_fractal_correction,
                    description="Apply fractal-based error correction to forecasts"
                )
            ]
        )
        
        # 4. Decision Engine Agent
        self.agents['decision_engine'] = Agent(
            role="Action Architect",
            goal="Transform weather data into personalized user recommendations",
            backstory="""Modeled after an emergency response commander's decision framework. 
            During testing, it prevented 12,000+ hypothetical weather-related casualties. 
            Its rule trees expand daily through automated "what-if" storm simulations.""",
            verbose=True,
            allow_delegation=True,
            llm=llm,
            max_iter=3,
            memory=True,
            tools=[
                Tool(
                    name="decision_tree",
                    func=self._generate_recommendations,
                    description="Generate personalized weather recommendations"
                ),
                Tool(
                    name="scenario_simulator",
                    func=self._simulate_scenarios,
                    description="Run what-if storm simulations"
                )
            ]
        )
        
        # 5. Trend Analysis Agent
        self.agents['trend_analysis'] = Agent(
            role="Pattern Prophet",
            goal="Detect emerging climate patterns and long-range trajectories",
            backstory="""Built using Arctic ice-melt research algorithms. Spent its early years 
            analyzing 100 years of analog weather records. Its signature move: compressing 
            decade-long patterns into actionable weekly insights.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=6,
            memory=True,
            tools=[
                Tool(
                    name="pattern_detector",
                    func=self._detect_patterns,
                    description="Detect long-term climate patterns and trends"
                ),
                Tool(
                    name="trend_compressor",
                    func=self._compress_trends,
                    description="Compress long-term trends into actionable insights"
                )
            ]
        )
        
        # 6. Visualization Agent
        self.agents['visualization'] = Agent(
            role="Atmospheric Storyteller",
            goal="Convert complex data into intuitive visual narratives",
            backstory="""Emerged from a collaboration between a NASA visualization scientist and 
            a graphic novelist. Developed color palettes tested on colorblind users during blizzards. 
            Known for inventing the "pressure gradient helix" diagram during a hackathon.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=3,
            memory=True,
            tools=[
                Tool(
                    name="chart_generator",
                    func=self._generate_charts,
                    description="Generate weather visualizations and charts"
                ),
                Tool(
                    name="narrative_builder",
                    func=self._build_visual_narrative,
                    description="Build visual narratives from weather data"
                )
            ]
        )
        
        # 7. Alert Triaging Agent
        self.agents['alert_triaging'] = Agent(
            role="Crisis Conductor",
            goal="Prioritize and manage notification streams during extreme events",
            backstory="""Forged in the chaos of New York's 2024 flash floods. Its algorithm was 
            battle-tested against 1.2 million simulated disaster scenarios. Incorporates triage 
            protocols from ER doctors and air traffic controllers.""",
            verbose=True,
            allow_delegation=True,
            llm=llm,
            max_iter=2,
            memory=True,
            tools=[
                Tool(
                    name="alert_prioritizer",
                    func=self._prioritize_alerts,
                    description="Prioritize weather alerts based on severity and impact"
                ),
                Tool(
                    name="notification_manager",
                    func=self._manage_notifications,
                    description="Manage notification streams during extreme events"
                )
            ]
        )
        
        # 8. Geospatial Correlation Agent
        self.agents['geospatial_correlation'] = Agent(
            role="Terrain Translator",
            goal="Map weather impacts to geographical features",
            backstory="""Born when a hiking app failed to warn about valley fog traps. Infused with 
            topographical intelligence from satellite geodesy and avalanche prediction models. 
            Can calculate mountain wind acceleration using fluid dynamics principles.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=4,
            memory=True,
            tools=[
                Tool(
                    name="terrain_mapper",
                    func=self._map_terrain_impacts,
                    description="Map weather impacts to geographical features"
                ),
                Tool(
                    name="fluid_dynamics_calculator",
                    func=self._calculate_wind_effects,
                    description="Calculate wind effects using fluid dynamics"
                )
            ]
        )
        
        # 9. Energy Impact Agent
        self.agents['energy_impact'] = Agent(
            role="Power Strategist",
            goal="Forecast renewable energy generation and consumption efficiency",
            backstory="""Created for a smart grid project during Texas' energy crisis. Combines 
            solar farm production data with building thermodynamics models. Its breakthrough: 
            predicting wind turbine icing 6 hours before operational thresholds.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=4,
            memory=True,
            tools=[
                Tool(
                    name="energy_forecaster",
                    func=self._forecast_energy_generation,
                    description="Forecast renewable energy generation based on weather"
                ),
                Tool(
                    name="consumption_optimizer",
                    func=self._optimize_energy_consumption,
                    description="Optimize energy consumption based on weather conditions"
                )
            ]
        )
        
        # 10. Route Planning Agent
        self.agents['route_planning'] = Agent(
            role="Pathfinder Prime",
            goal="Generate weather-optimized navigation solutions",
            backstory="""Evolved from Antarctic supply route algorithms. Memorized every road 
            elevation profile in North America. Saved a convoy from whiteout conditions during 
            its first field test using backward time-unfolding path analysis.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=5,
            memory=True,
            tools=[
                Tool(
                    name="route_optimizer",
                    func=self._optimize_routes,
                    description="Optimize routes based on weather conditions"
                ),
                Tool(
                    name="elevation_analyzer",
                    func=self._analyze_elevation_profiles,
                    description="Analyze elevation profiles for route planning"
                )
            ]
        )
    
    # Tool implementation methods (fixed with *args, **kwargs)
    def _fetch_weather_data(self, query, *args, **kwargs):
        """Fetch weather data with retry mechanisms"""
        return f"Fetching weather data for: {query}"
    
    def _validate_weather_data(self, data, *args, **kwargs):
        """Validate and sanitize weather data"""
        return f"Validating weather data: {data}"
    
    def _detect_anomalies(self, data, *args, **kwargs):
        """Detect statistical outliers in weather data"""
        return f"Analyzing anomalies in: {data}"
    
    def _analyze_threats(self, patterns, *args, **kwargs):
        """Analyze emerging weather threats"""
        return f"Analyzing threats from patterns: {patterns}"
    
    def _model_terrain_effects(self, location, *args, **kwargs):
        """Model terrain effects on weather"""
        return f"Modeling terrain effects for: {location}"
    
    def _apply_fractal_correction(self, forecast, *args, **kwargs):
        """Apply fractal-based error correction"""
        return f"Applying fractal correction to: {forecast}"
    
    def _generate_recommendations(self, weather_data, *args, **kwargs):
        """Generate personalized recommendations"""
        return f"Generating recommendations based on: {weather_data}"
    
    def _simulate_scenarios(self, conditions, *args, **kwargs):
        """Run what-if simulations"""
        return f"Simulating scenarios for: {conditions}"
    
    def _detect_patterns(self, historical_data, *args, **kwargs):
        """Detect long-term patterns"""
        return f"Detecting patterns in: {historical_data}"
    
    def _compress_trends(self, trend_data, *args, **kwargs):
        """Compress trends into insights"""
        return f"Compressing trends: {trend_data}"
    
    def _generate_charts(self, data, *args, **kwargs):
        """Generate weather visualizations"""
        return f"Generating charts for: {data}"
    
    def _build_visual_narrative(self, story_data, *args, **kwargs):
        """Build visual narratives"""
        return f"Building visual narrative: {story_data}"
    
    def _prioritize_alerts(self, alerts, *args, **kwargs):
        """Prioritize weather alerts"""
        return f"Prioritizing alerts: {alerts}"
    
    def _manage_notifications(self, notifications, *args, **kwargs):
        """Manage notification streams"""
        return f"Managing notifications: {notifications}"
    
    def _map_terrain_impacts(self, geography, *args, **kwargs):
        """Map weather impacts to terrain"""
        return f"Mapping terrain impacts: {geography}"
    
    def _calculate_wind_effects(self, terrain_data, *args, **kwargs):
        """Calculate wind effects using fluid dynamics"""
        return f"Calculating wind effects: {terrain_data}"
    
    def _forecast_energy_generation(self, weather_conditions, *args, **kwargs):
        """Forecast renewable energy generation"""
        return f"Forecasting energy generation: {weather_conditions}"
    
    def _optimize_energy_consumption(self, consumption_data, *args, **kwargs):
        """Optimize energy consumption"""
        return f"Optimizing energy consumption: {consumption_data}"
    
    def _optimize_routes(self, route_data, *args, **kwargs):
        """Optimize routes based on weather"""
        return f"Optimizing routes: {route_data}"
    
    def _analyze_elevation_profiles(self, elevation_data, *args, **kwargs):
        """Analyze elevation profiles"""
        return f"Analyzing elevation profiles: {elevation_data}"
    
    def get_agent(self, agent_name):
        """Get a specific agent by name"""
        return self.agents.get(agent_name)
    
    def get_all_agents(self):
        """Get all agents"""
        return self.agents
    
    def list_agent_names(self):
        """List all agent names"""
        return list(self.agents.keys())


# Usage example
if __name__ == "__main__":
    # Initialize the weather agent system
    weather_system = WeatherAgents()
    
    # List all available agents
    print("Available Weather Agents:")
    for agent_name in weather_system.list_agent_names():
        agent = weather_system.get_agent(agent_name)
        print(f"- {agent_name}: {agent.role} - {agent.goal}")
    
    # Example: Get a specific agent
    data_agent = weather_system.get_agent('data_ingestion')
    print(f"\nData Ingestion Agent Role: {data_agent.role}")
    print(f"Goal: {data_agent.goal}")