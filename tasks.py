from crewai import Task
from agents import WeatherAgents
import json

class WeatherTasks:
    """Collection of specialized weather intelligence tasks"""
    
    def __init__(self):
        self.weather_agents = WeatherAgents()
        self.tasks = {}
        self._initialize_tasks()
    
    def _initialize_tasks(self):
        """Initialize all weather-related tasks"""
        
        # 1. Real-time Data Collection Task
        self.tasks['data_collection'] = Task(
            description="""
            Fetch comprehensive weather data from OpenWeather API for the specified location.
            
            Requirements:
            - Collect current weather conditions
            - Gather 5-day forecast data
            - Retrieve historical weather patterns (if available)
            - Fetch air quality index data
            - Gather UV index information
            - Collect precipitation probability and intensity
            - Ensure data integrity through validation checks
            
            Expected Output: Structured JSON containing all weather metrics with timestamps
            """,
            agent=self.weather_agents.get_agent('data_ingestion'),
            expected_output="Clean, validated weather data in structured JSON format with all required metrics"
        )
        
        # 2. Weather Anomaly Detection Task
        self.tasks['anomaly_detection'] = Task(
            description="""
            Analyze incoming weather data to identify unusual patterns and potential threats.
            
            Requirements:
            - Compare current conditions against historical averages
            - Identify temperature, pressure, and humidity anomalies
            - Detect rapid weather changes that could indicate severe conditions
            - Flag potential extreme weather events (storms, heat waves, cold snaps)
            - Calculate deviation scores from normal patterns
            - Generate early warning indicators
            
            Expected Output: Anomaly report with risk scores and threat assessments
            """,
            agent=self.weather_agents.get_agent('anomaly_detection'),
            expected_output="Detailed anomaly analysis with risk scores and early warning flags"
        )
        
        # 3. Forecast Enhancement Task
        self.tasks['forecast_enhancement'] = Task(
            description="""
            Enhance raw weather forecasts using advanced correction algorithms and local factors.
            
            Requirements:
            - Apply terrain-based corrections to temperature and wind predictions
            - Implement fractal error correction for improved accuracy
            - Adjust forecasts based on microclimate factors
            - Incorporate urban heat island effects for city locations
            - Apply elevation-based temperature corrections
            - Generate confidence intervals for each forecast parameter
            
            Expected Output: Enhanced forecast with improved accuracy and confidence metrics
            """,
            agent=self.weather_agents.get_agent('forecast_optimization'),
            expected_output="Optimized weather forecast with accuracy improvements and confidence scores"
        )
        
        # 4. Personalized Recommendations Task
        self.tasks['recommendations'] = Task(
            description="""
            Generate personalized weather-based recommendations for user activities.
            
            Requirements:
            - Analyze weather conditions for outdoor activity suitability
            - Provide clothing recommendations based on temperature and conditions
            - Suggest optimal times for specific activities (running, cycling, gardening)
            - Generate travel advisories for planned routes
            - Create health-related weather warnings (air quality, UV exposure)
            - Recommend energy-saving actions based on weather conditions
            
            Expected Output: Comprehensive activity and safety recommendations
            """,
            agent=self.weather_agents.get_agent('decision_engine'),
            expected_output="Personalized recommendations covering activities, clothing, travel, and health"
        )
        
        # 5. Climate Trend Analysis Task
        self.tasks['trend_analysis'] = Task(
            description="""
            Analyze long-term weather patterns to identify emerging climate trends.
            
            Requirements:
            - Process historical weather data to identify trends
            - Calculate seasonal pattern shifts
            - Detect long-term temperature and precipitation changes
            - Identify recurring weather cycles
            - Generate monthly and seasonal outlook predictions
            - Create trend visualization data points
            
            Expected Output: Climate trend analysis with future pattern predictions
            """,
            agent=self.weather_agents.get_agent('trend_analysis'),
            expected_output="Comprehensive trend analysis with seasonal outlooks and pattern insights"
        )
        
        # 6. Weather Visualization Task
        self.tasks['visualization'] = Task(
            description="""
            Create compelling visual representations of weather data and forecasts.
            
            Requirements:
            - Generate temperature and precipitation charts
            - Create weather map overlays with pressure systems
            - Build interactive forecast timelines
            - Design weather alert visual indicators
            - Create comparative weather charts (current vs historical)
            - Generate infographic-style weather summaries
            
            Expected Output: Collection of weather visualizations and chart specifications
            """,
            agent=self.weather_agents.get_agent('visualization'),
            expected_output="Weather visualization specifications and chart data for frontend rendering"
        )
        
        # 7. Alert Management Task
        self.tasks['alert_management'] = Task(
            description="""
            Manage and prioritize weather alerts and notifications for users.
            
            Requirements:
            - Process severe weather warnings from multiple sources
            - Prioritize alerts based on severity and user location
            - Create custom alert thresholds for different user preferences
            - Generate push notification content
            - Manage alert escalation procedures
            - Track alert effectiveness and user response
            
            Expected Output: Prioritized alert system with notification management
            """,
            agent=self.weather_agents.get_agent('alert_triaging'),
            expected_output="Organized alert priority system with notification schedules and content"
        )
        
        # 8. Geospatial Weather Mapping Task
        self.tasks['geospatial_mapping'] = Task(
            description="""
            Map weather impacts across different geographical features and terrains.
            
            Requirements:
            - Correlate weather patterns with topographical features
            - Calculate terrain-specific weather effects
            - Map microclimate variations within regions
            - Identify weather corridors and wind patterns
            - Generate location-specific weather adjustments
            - Create geographical weather risk assessments
            
            Expected Output: Geospatial weather analysis with terrain-corrected forecasts
            """,
            agent=self.weather_agents.get_agent('geospatial_correlation'),
            expected_output="Terrain-aware weather mapping with geographical risk assessments"
        )
        
        # 9. Energy Impact Assessment Task
        self.tasks['energy_assessment'] = Task(
            description="""
            Assess weather impact on energy generation and consumption patterns.
            
            Requirements:
            - Forecast solar panel efficiency based on cloud cover and sun angle
            - Predict wind turbine energy generation potential
            - Calculate heating and cooling energy demands
            - Assess weather-related energy grid stress factors
            - Generate energy-saving opportunity alerts
            - Provide renewable energy optimization recommendations
            
            Expected Output: Energy impact analysis with optimization recommendations
            """,
            agent=self.weather_agents.get_agent('energy_impact'),
            expected_output="Comprehensive energy impact assessment with optimization strategies"
        )
        
        # 10. Route Optimization Task
        self.tasks['route_optimization'] = Task(
            description="""
            Optimize travel routes based on weather conditions and forecasts.
            
            Requirements:
            - Analyze weather conditions along planned routes
            - Identify weather-related travel hazards (ice, fog, storms)
            - Calculate optimal departure times to avoid adverse conditions
            - Suggest alternative routes during severe weather
            - Provide real-time route adjustments based on changing conditions
            - Generate travel time estimates with weather delays
            
            Expected Output: Weather-optimized route recommendations with timing
            """,
            agent=self.weather_agents.get_agent('route_planning'),
            expected_output="Optimized travel routes with weather-aware timing and alternatives"
        )
        
        # 11. Comprehensive Weather Intelligence Task (Orchestrator)
        self.tasks['weather_intelligence'] = Task(
            description="""
            Orchestrate all weather agents to provide comprehensive weather intelligence.
            
            Requirements:
            - Coordinate data collection and processing across all agents
            - Synthesize insights from anomaly detection and trend analysis
            - Integrate enhanced forecasts with personalized recommendations
            - Combine visualization data with alert management
            - Merge geospatial analysis with energy and route optimization
            - Generate a unified weather intelligence report
            - Ensure all outputs are synchronized and consistent
            
            Expected Output: Master weather intelligence report with all integrated insights
            """,
            agent=self.weather_agents.get_agent('decision_engine'),
            expected_output="Comprehensive weather intelligence dashboard with integrated insights from all agents"
        )
        
        # 12. Weather App Data Pipeline Task
        self.tasks['data_pipeline'] = Task(
            description="""
            Create a complete data pipeline for the weather app's real-time operations.
            
            Requirements:
            - Establish continuous data fetching from OpenWeather API
            - Implement data quality checks and error handling
            - Create data transformation workflows
            - Set up real-time data streaming for live updates
            - Implement data caching strategies for performance
            - Generate API response formats for frontend consumption
            
            Expected Output: Complete data pipeline architecture with real-time capabilities
            """,
            agent=self.weather_agents.get_agent('data_ingestion'),
            expected_output="Robust data pipeline with real-time processing and error handling"
        )
    
    def get_task(self, task_name):
        """Get a specific task by name"""
        return self.tasks.get(task_name)
    
    def get_all_tasks(self):
        """Get all tasks"""
        return self.tasks
    
    def list_task_names(self):
        """List all task names"""
        return list(self.tasks.keys())
    
    def get_task_by_agent_role(self, agent_role):
        """Get tasks assigned to a specific agent role"""
        matching_tasks = []
        for task_name, task in self.tasks.items():
            if task.agent.role == agent_role:
                matching_tasks.append((task_name, task))
        return matching_tasks
    
    def create_task_sequence(self, task_names):
        """Create a sequence of tasks for execution"""
        task_sequence = []
        for task_name in task_names:
            if task_name in self.tasks:
                task_sequence.append(self.tasks[task_name])
        return task_sequence
    
    def get_core_weather_tasks(self):
        """Get the core tasks needed for basic weather app functionality"""
        core_tasks = [
            'data_collection',
            'forecast_enhancement', 
            'recommendations',
            'visualization',
            'alert_management'
        ]
        return self.create_task_sequence(core_tasks)
    
    def get_advanced_weather_tasks(self):
        """Get advanced tasks for comprehensive weather intelligence"""
        advanced_tasks = [
            'data_collection',
            'anomaly_detection',
            'forecast_enhancement',
            'trend_analysis',
            'geospatial_mapping',
            'energy_assessment',
            'route_optimization',
            'recommendations',
            'visualization',
            'alert_management',
            'weather_intelligence'
        ]
        return self.create_task_sequence(advanced_tasks)
    
    def get_real_time_tasks(self):
        """Get tasks optimized for real-time weather operations"""
        real_time_tasks = [
            'data_pipeline',
            'anomaly_detection',
            'alert_management',
            'recommendations'
        ]
        return self.create_task_sequence(real_time_tasks)


# Task execution workflows
class WeatherWorkflows:
    """Pre-defined workflows for different weather app scenarios"""
    
    def __init__(self):
        self.tasks = WeatherTasks()
    
    def basic_weather_workflow(self, location, user_preferences=None):
        """Basic weather app workflow"""
        workflow_tasks = self.tasks.get_core_weather_tasks()
        return {
            'workflow_name': 'basic_weather',
            'location': location,
            'user_preferences': user_preferences or {},
            'tasks': workflow_tasks,
            'description': 'Core weather functionality with forecasts and recommendations'
        }
    
    def advanced_intelligence_workflow(self, location, historical_days=30):
        """Advanced weather intelligence workflow"""
        workflow_tasks = self.tasks.get_advanced_weather_tasks()
        return {
            'workflow_name': 'advanced_intelligence',
            'location': location,
            'historical_period': historical_days,
            'tasks': workflow_tasks,
            'description': 'Comprehensive weather analysis with AI-driven insights'
        }
    
    def travel_planning_workflow(self, origin, destination, travel_date):
        """Travel-focused weather workflow"""
        travel_tasks = [
            'data_collection',
            'forecast_enhancement',
            'route_optimization', 
            'alert_management',
            'recommendations'
        ]
        workflow_tasks = self.tasks.create_task_sequence(travel_tasks)
        return {
            'workflow_name': 'travel_planning',
            'origin': origin,
            'destination': destination,
            'travel_date': travel_date,
            'tasks': workflow_tasks,
            'description': 'Weather-optimized travel planning and route recommendations'
        }
    
    def energy_optimization_workflow(self, location, energy_systems):
        """Energy-focused weather workflow"""
        energy_tasks = [
            'data_collection',
            'forecast_enhancement',
            'energy_assessment',
            'recommendations',
            'visualization'
        ]
        workflow_tasks = self.tasks.create_task_sequence(energy_tasks)
        return {
            'workflow_name': 'energy_optimization',
            'location': location,
            'energy_systems': energy_systems,
            'tasks': workflow_tasks,
            'description': 'Weather-driven energy generation and consumption optimization'
        }
    
    def emergency_response_workflow(self, location, alert_types):
        """Emergency weather response workflow"""
        emergency_tasks = [
            'data_pipeline',
            'anomaly_detection',
            'alert_management',
            'geospatial_mapping',
            'route_optimization',
            'recommendations'
        ]
        workflow_tasks = self.tasks.create_task_sequence(emergency_tasks)
        return {
            'workflow_name': 'emergency_response',
            'location': location,
            'alert_types': alert_types,
            'tasks': workflow_tasks,
            'description': 'Real-time severe weather monitoring and emergency response'
        }


# Usage example
if __name__ == "__main__":
    # Initialize the weather task system
    task_system = WeatherTasks()
    workflows = WeatherWorkflows()
    
    # List all available tasks
    print("Available Weather Tasks:")
    for task_name in task_system.list_task_names():
        task = task_system.get_task(task_name)
        print(f"- {task_name}: {task.agent.role}")
        print(f"  Goal: {task.description[:100]}...")
        print()
    
    # Example workflow creation
    print("\n" + "="*50)
    print("Example Workflows:")
    
    # Basic weather workflow
    basic_workflow = workflows.basic_weather_workflow(
        location="New York, NY",
        user_preferences={"units": "metric", "alerts": True}
    )
    print(f"\n1. {basic_workflow['workflow_name']}: {basic_workflow['description']}")
    print(f"   Tasks: {len(basic_workflow['tasks'])}")
    
    # Advanced intelligence workflow  
    advanced_workflow = workflows.advanced_intelligence_workflow(
        location="San Francisco, CA",
        historical_days=60
    )
    print(f"\n2. {advanced_workflow['workflow_name']}: {advanced_workflow['description']}")
    print(f"   Tasks: {len(advanced_workflow['tasks'])}")
    
    # Travel planning workflow
    travel_workflow = workflows.travel_planning_workflow(
        origin="Los Angeles, CA",
        destination="Las Vegas, NV", 
        travel_date="2025-06-15"
    )
    print(f"\n3. {travel_workflow['workflow_name']}: {travel_workflow['description']}")
    print(f"   Tasks: {len(travel_workflow['tasks'])}")