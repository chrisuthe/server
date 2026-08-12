"""Constants for the Pandora music provider."""

# API Endpoints
API_HOST = "https://www.pandora.com/api"
API_BASE = f"{API_HOST}/v1"
# The music catalogue lives on v4. Note v1/aesop/annotateObjects is the *podcast* annotator
# and does not answer for TR:/AL:/AR: ids.
CATALOG_ANNOTATE_ENDPOINT = f"{API_HOST}/v4/catalog/annotateObjects"
# getDetails answers for one id and returns an annotations map alongside a per-type detail
# block. Measured on a TR: id, where the map carried the track plus its album and artist.
CATALOG_DETAILS_ENDPOINT = f"{API_HOST}/v4/catalog/getDetails"
LOGIN_ENDPOINT = f"{API_BASE}/auth/login"
STATIONS_ENDPOINT = f"{API_BASE}/station/getStations"
PLAYLIST_FRAGMENT_ENDPOINT = f"{API_BASE}/playlist/getFragment"
PLAYBACK_RESUMED_ENDPOINT = f"{API_BASE}/station/playbackResumed"
# fullSearch is a station-*seed* search: its results are things a station can be built
# around, not things that can be played. The catalogue search lives on v3.
SEED_SEARCH_ENDPOINT = f"{API_BASE}/search/fullSearch"
SOD_SEARCH_ENDPOINT = f"{API_HOST}/v3/sod/search"
PLAYBACK_SOURCE_ENDPOINT = f"{API_BASE}/playback/source"
CREATE_STATION_ENDPOINT = f"{API_BASE}/station/createStation"
REMOVE_STATION_ENDPOINT = f"{API_BASE}/station/removeStation"
ADD_SEED_ENDPOINT = f"{API_BASE}/station/addSeed"

# Type-prefixed Pandora ids that createStation accepts as a station seed.
# fullSearch returns AR/CO/GE/TR on a Premium account; measured in probe 1.
SEEDABLE_PREFIXES = ("AR", "CO", "GE", "TR")

# Pandora Error Code Categories
# Authentication and authorization failures
AUTH_ERRORS = {12, 13, 1001, 1002, 1003}
# Missing stations, tracks, or other media
NOT_FOUND_ERRORS = {4, 5, 1006}
# Temporary service unavailability
UNAVAILABLE_ERRORS = {1, 9, 10, 34, 1000}

# Pandora API Error Code Descriptions
PANDORA_ERROR_CODES = {
    0: "Internal error",
    1: "Maintenance mode",
    2: "URL parameter missing method",
    3: "URL parameter missing auth_token",
    4: "URL parameter missing partner_id",
    5: "URL parameter missing user_id",
    6: "Secure protocol required",
    7: "Certificate required",
    8: "Parameter type mismatch",
    9: "Parameter missing",
    10: "Parameter value invalid",
    11: "API version not supported",
    12: "Invalid username",
    13: "Invalid password",
    14: "Listener not authorized",
    15: "Partner not authorized",
    1000: "Read only mode",
    1001: "Invalid auth token",
    1002: "Invalid partner login",
    1003: "Listener not authorized",
    1004: "Partner not authorized",
    1005: "Station limit reached",
    1006: "Station does not exist",
    1009: "Device not found",
    1010: "Partner not authorized",
    1011: "Invalid username",
    1012: "Invalid password",
    1023: "Device model invalid",
    1035: "Explicit pin incorrect",
    1036: "Explicit pin malformed",
    1037: "Device already associated to account",
    1039: "Device not found",
}

RETRY_REASON_AUTH = "auth"
RETRY_REASON_STREAM_VIOLATION = "stream_violation"

CONF_TAKEOVER_ACTION = "takeover_stream"
CONF_QUALITY = "quality"
CONF_ALLOW_STATION_DELETE = "allow_station_delete"
QUALITY_HIGH = "high"
QUALITY_STANDARD = "standard"

ACCOUNT_FLAG_HIGH_QUALITY = "highQualityStreamingAvailable"
# The account may play a chosen track without watching an ad. Free accounts carry
# adSupportedReplay/adSupportedSkip instead and can play on demand only after an ad
# value-exchange, which Music Assistant cannot present - so this gate means "can play
# without an ad", not "is Premium".
ACCOUNT_FLAG_ON_DEMAND = "onDemand"

# What the listener is told wherever an on-demand request is refused, whether this provider
# answered for the account itself or Pandora refused the request with NO_ENTITLEMENTS.
NO_ON_DEMAND_MESSAGE = "On-demand playback is not available on this Pandora account"

CONF_DEVICE_UUID = "device_uuid"

# Reference level a ReplayGain-style adjustment is measured against, so a gain can be
# expressed as the integrated loudness Music Assistant normalises on. Same figure the
# core uses for a REPLAYGAIN_TRACK_GAIN tag - see helpers/tags.py.
REPLAY_GAIN_REFERENCE_LUFS = -18.0
