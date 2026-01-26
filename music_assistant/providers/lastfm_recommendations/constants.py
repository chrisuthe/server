"""Constants for Last.fm Recommendations provider."""

from music_assistant_models.enums import ProviderFeature

# Supported provider features
SUPPORTED_FEATURES = {
    ProviderFeature.RECOMMENDATIONS,
}

# Config action constants
CONF_ACTION_CLEAR_CACHE = "clear_cache"

# Cache categories for persistent storage
CACHE_CATEGORY_MBID_ISRC = 0  # MBID->ISRC mappings
CACHE_CATEGORY_RESOLVED_ITEMS = 1  # Resolved Artist/Track/Album objects

# Curated list of popular countries for Last.fm geo charts
# Last.fm API expects full country names (not ISO codes)
# This list covers major music markets and can be expanded based on user requests
GEO_COUNTRIES = [
    "Argentina",
    "Australia",
    "Austria",
    "Belgium",
    "Brazil",
    "Canada",
    "China",
    "Czech Republic",
    "Denmark",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "India",
    "Ireland",
    "Israel",
    "Italy",
    "Japan",
    "Lithuania",
    "Mexico",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Philippines",
    "Poland",
    "Portugal",
    "Serbia",
    "Singapore",
    "Slovenia",
    "South Africa",
    "South Korea",
    "Spain",
    "Sweden",
    "Switzerland",
    "Thailand",
    "Turkey",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
]
