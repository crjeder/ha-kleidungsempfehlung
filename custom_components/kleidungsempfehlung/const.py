"""Constants for the Kleidungsempfehlung integration."""

DOMAIN = "kleidungsempfehlung"
PLATFORMS = ["sensor"]
DEFAULT_NAME = "Kleidungsempfehlung"

# Configuration keys
CONF_INVENTORY = "inventory"
CONF_BASE_ENSEMBLE = "base_ensemble"
CONF_MAX_LAYERS = "max_layers"
CONF_SOLVER = "solver"
CONF_LAYERING_FACTOR = "layering_factor"

# Weather entity (required: the sole source for temperature and wind data)
CONF_WEATHER_ENTITY = "weather_entity"

# Optional humidity sensor override (used alongside weather_entity)
CONF_SENSOR_HUMIDITY = "sensor_humidity"

# Person configuration
CONF_PERSON_ENTITY = "person_entity"
CONF_SENSOR_ACTIVITY = "sensor_activity"
CONF_SENSOR_AGE = "sensor_age"
CONF_SENSOR_GENDER = "sensor_gender"
CONF_PMV_TARGET = "pmv_target"

# Defaults
DEFAULT_MAX_LAYERS = 4
DEFAULT_SOLVER = "ilp"
DEFAULT_LAYERING_FACTOR = 0.835
DEFAULT_PMV_TARGET = 0.0
