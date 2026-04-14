"""Dataset-level default profiles for canonical global attributes."""

from typing import Dict


DEFAULT_CREATOR_NAME = "Zhongwang Wei"
DEFAULT_CREATOR_EMAIL = "weizhw6@mail.sysu.edu.cn"
DEFAULT_CREATOR_INSTITUTION = "Sun Yat-sen University, China"
DEFAULT_PROCESSING_LEVEL = "Quality controlled and standardized"
DEFAULT_FEATURE_TYPE = "timeSeries"


DEFAULT_PROFILE = {
    "data_source_name": "",
    "source_data_link": "",
    "creator_name": DEFAULT_CREATOR_NAME,
    "creator_email": DEFAULT_CREATOR_EMAIL,
    "creator_institution": DEFAULT_CREATOR_INSTITUTION,
    "default_observation_type": "",
    "default_source": "In-situ station data",
    "default_summary": "",
    "default_comment": "",
    "default_geographic_coverage": "",
    "default_processing_level": DEFAULT_PROCESSING_LEVEL,
    "default_feature_type": DEFAULT_FEATURE_TYPE,
}


DATASET_PROFILES = {
    "RiverSed": {
        "data_source_name": "RiverSed / Aquasat (satellite-derived TSS)",
        "source_data_link": "https://doi.org/10.1029/2020GL088946",
        "default_observation_type": "Satellite",
        "default_source": "Satellite-derived TSS from Aquasat/RiverSed database",
        "default_comment": "TSS values derived from Landsat satellite imagery.",
    },
    "GFQA_v2": {
        "data_source_name": "Global Flow and Water Quality Archive v2",
        "source_data_link": "",
        "default_observation_type": "In-situ station data",
        "default_source": "Global Flow and Water Quality Archive v2",
    },
    "USGS": {
        "data_source_name": "USGS NWIS",
        "source_data_link": "https://waterdata.usgs.gov/nwis",
        "default_observation_type": "In-situ station data",
        "default_source": "In-situ station data",
    },
    "HYDAT": {
        "data_source_name": "HYDAT Dataset",
        "source_data_link": "https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html",
        "default_observation_type": "In-situ",
        "default_source": "In-situ station data",
    },
    "Hydat": {
        "data_source_name": "HYDAT Dataset",
        "source_data_link": "https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html",
        "default_observation_type": "In-situ",
        "default_source": "In-situ station data",
    },
    "Milliman": {
        "data_source_name": "Milliman & Farnsworth Global River Sediment Database",
        "source_data_link": "https://doi.org/10.1126/science.abn7980",
        "default_observation_type": "In-situ",
        "default_source": "In-situ station data",
    },
    "Vanmaercke": {
        "data_source_name": "Vanmaercke et al. (2014) African Sediment Yield Database",
        "source_data_link": "https://doi.org/10.1016/j.earscirev.2014.06.004",
        "default_observation_type": "In-situ",
        "default_source": "In-situ station data",
    },
    "Dethier": {
        "data_source_name": "Dethier glacier-fed rivers dataset",
        "source_data_link": "https://doi.org/10.1126/science.abn7980",
        "default_observation_type": "Satellite station",
        "default_source": "Satellite station",
    },
    "EUSEDcollab": {
        "data_source_name": "EUSEDcollab Dataset",
        "source_data_link": "https://esdac.jrc.ec.europa.eu/content/european-sediment-collaboration-eusedcollab-database",
        "default_observation_type": "In-situ",
        "default_source": "In-situ station data",
    },
    "GSED": {
        "data_source_name": "GSED Dataset",
        "source_data_link": "https://doi.org/10.1038/s41597-023-02233-0",
        "default_observation_type": "Satellite",
        "default_source": "Satellite station",
        "default_geographic_coverage": "Global rivers",
    },
}


DATASET_ALIASES = {
    "hydat": "HYDAT",
}


def _pretty_dataset_name(name):
    return str(name or "").replace("_", " ").strip()


def normalize_dataset_name(name):
    """Normalize dataset names with case-insensitive matching."""
    raw = str(name or "").strip()
    if not raw:
        return ""

    lower = raw.lower()
    if lower in DATASET_ALIASES:
        return DATASET_ALIASES[lower]

    for canonical in DATASET_PROFILES:
        if canonical.lower() == lower:
            return canonical
    return raw


def get_dataset_profile(dataset_name):
    """Return a merged profile with defaults applied."""
    normalized = normalize_dataset_name(dataset_name)
    profile = dict(DEFAULT_PROFILE)
    if normalized in DATASET_PROFILES:
        profile.update(DATASET_PROFILES[normalized])

    if not profile.get("data_source_name"):
        profile["data_source_name"] = _pretty_dataset_name(normalized)

    if not profile.get("default_source"):
        obs = str(profile.get("default_observation_type", "")).lower()
        if "satellite" in obs:
            profile["default_source"] = "Satellite station"
        else:
            profile["default_source"] = "In-situ station data"

    profile["dataset_name"] = normalized
    return profile

