"""Dataset-level default profiles for canonical global attributes."""

from typing import Dict


DEFAULT_CREATOR_NAME = "Zhongwang Wei"
DEFAULT_CREATOR_EMAIL = "weizhw6@mail.sysu.edu.cn"
DEFAULT_CREATOR_INSTITUTION = "Sun Yat-sen University, China"
DEFAULT_PROCESSING_LEVEL = "Quality controlled and standardized"
DEFAULT_FEATURE_TYPE = "timeSeries"

OBS_IN_SITU = "In-situ station data"
OBS_SATELLITE = "Satellite"

SOURCE_IN_SITU = "In-situ station data"


DEFAULT_PROFILE = {
    "data_source_name": "",
    "source_data_link": "",
    "creator_name": DEFAULT_CREATOR_NAME,
    "creator_email": DEFAULT_CREATOR_EMAIL,
    "creator_institution": DEFAULT_CREATOR_INSTITUTION,
    "default_observation_type": "",
    "default_source": SOURCE_IN_SITU,
    "default_summary": "",
    "default_comment": "",
    "default_geographic_coverage": "",
    "default_processing_level": DEFAULT_PROCESSING_LEVEL,
    "default_feature_type": DEFAULT_FEATURE_TYPE,
}


DATASET_PROFILES = {
    "ALi_De_Boer": {
        "data_source_name": "Ali & De Boer Upper Indus Sediment Yield Dataset",
        "source_data_link": "https://doi.org/10.1016/j.jhydrol.2006.10.013",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Upper Indus River Basin, Northern Pakistan and Western Himalayas",
    },
    "Chao_Phraya_River": {
        "data_source_name": "Chao_Phraya_River Dataset",
        "source_data_link": "https://doi.org/10.1594/PANGAEA.981111",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Chao Phraya River Basin, Thailand",
    },
    "Dethier": {
        "data_source_name": "Dethier glacier-fed rivers dataset",
        "source_data_link": "https://doi.org/10.1126/science.abn7980",
        "default_observation_type": OBS_SATELLITE,
        "default_source": "Satellite station",
    },
    "EUSEDcollab": {
        "data_source_name": "EUSEDcollab Dataset",
        "source_data_link": "https://esdac.jrc.ec.europa.eu/content/european-sediment-collaboration-eusedcollab-database",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Europe",
    },
    "Eurasian_River": {
        "data_source_name": "Eurasian River Historical Sediment Flux Data",
        "source_data_link": "https://doi.org/10.5065/D6F769PB",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Eurasian Arctic river basins",
    },
    "Fukushima": {
        "data_source_name": "Fukushima Niida River Dataset",
        "source_data_link": "https://doi.org/10.34355/CRiED.U.Tsukuba.00147",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Niida River Basin, Fukushima Prefecture, Japan",
    },
    "GFQA_v2": {
        "data_source_name": "Global Flow and Water Quality Archive v2",
        "source_data_link": "",
        "default_observation_type": OBS_IN_SITU,
        "default_source": "Global Flow and Water Quality Archive v2",
    },
    "GloRiSe": {
        "data_source_name": "GloRiSe Dataset",
        "source_data_link": "https://doi.org/10.5281/zenodo.4485795",
        "default_observation_type": OBS_IN_SITU,
        "default_source": "Global River Sediment Database v1.1 - quality controlled and standardized",
    },
    "GSED": {
        "data_source_name": "GSED Dataset",
        "source_data_link": "https://doi.org/10.1038/s41597-023-02233-0",
        "default_observation_type": OBS_SATELLITE,
        "default_source": "Satellite station",
        "default_geographic_coverage": "Global rivers",
    },
    "HMA": {
        "data_source_name": "HMA Dataset (Li et al. 2021)",
        "source_data_link": "https://doi.org/10.1126/science.abi9649",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "High Mountain Asia",
        "default_comment": "Mean annual climatology derived from source observations.",
    },
    "HYBAM": {
        "data_source_name": "HYBAM Dataset",
        "source_data_link": "https://hybam.obs-mip.fr/",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Amazon Basin",
    },
    "HYDAT": {
        "data_source_name": "HYDAT Dataset",
        "source_data_link": "https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
    },
    "Huanghe": {
        "data_source_name": "Yellow River Sediment Bulletin Dataset",
        "source_data_link": "https://doi.org/10.12072/ncdc.YRiver.db0054.2021",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Yellow River Basin, China",
        "default_comment": "Annual average SSC observations; discharge and sediment load are not provided in the source dataset.",
    },
    "Hydat": {
        "data_source_name": "HYDAT Dataset",
        "source_data_link": "https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
    },
    "Mekong_Delta": {
        "data_source_name": "Mekong Delta (Darby et al., 2020)",
        "source_data_link": "https://doi.org/10.5285/ac5b28ca-e087-4aec-974a-5a9f84b06595",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Mekong River Delta, Vietnam",
    },
    "Milliman": {
        "data_source_name": "Milliman & Farnsworth Global River Sediment Database",
        "source_data_link": "https://doi.org/10.1126/science.abn7980",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
    },
    "Myanmar": {
        "data_source_name": "Myanmar (Irrawaddy and Salween Rivers)",
        "source_data_link": "https://doi.org/10.5285/86f17d61-141f-4500-9aa5-26a82aef0b33",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Irrawaddy and Salween Rivers, Myanmar",
    },
    "NERC": {
        "data_source_name": "NERC Hampshire Avon Dataset",
        "source_data_link": "https://doi.org/10.5285/0dd10858-7b96-41f1-8db5-e7b4c4168af5",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Hampshire Avon Basin, Southern England, UK",
    },
    "Rhine": {
        "data_source_name": "Rhine Dataset",
        "source_data_link": "https://doi.org/10.1002/hyp.70070",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Rhine River Basin",
    },
    "RiverSed": {
        "data_source_name": "RiverSed / Aquasat (satellite-derived TSS)",
        "source_data_link": "https://doi.org/10.1029/2020GL088946",
        "default_observation_type": OBS_SATELLITE,
        "default_source": "Satellite-derived TSS from Aquasat/RiverSed database",
        "default_comment": "TSS values derived from Landsat satellite imagery.",
    },
    "Robotham": {
        "data_source_name": "Robotham et al. (2022)",
        "source_data_link": "https://doi.org/10.5285/9f80e349-0594-4ae1-bff3-b055638569f8",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Littlestock Brook, England",
    },
    "Shashi_Jianli": {
        "data_source_name": "Shashi_Jianli Dataset",
        "source_data_link": "https://doi.org/10.1007/s11600-025-01638-x",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Yangtze River Basin, China",
    },
    "USGS": {
        "data_source_name": "USGS NWIS",
        "source_data_link": "https://waterdata.usgs.gov/nwis",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
    },
    "Vanmaercke": {
        "data_source_name": "Vanmaercke et al. (2014) African Sediment Yield Database",
        "source_data_link": "https://doi.org/10.1016/j.earscirev.2014.06.004",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
    },
    "Yajiang": {
        "data_source_name": "Yajiang Dataset",
        "source_data_link": "https://doi.org/10.11888/Hydro.tpdc.270293",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Yarlung Tsangpo River Basin, China",
    },
    "bayern": {
        "data_source_name": "Bayern State Environmental Agency (LfU) River Monitoring Network",
        "source_data_link": "https://www.gkd.bayern.de/en/",
        "default_observation_type": OBS_IN_SITU,
        "default_source": SOURCE_IN_SITU,
        "default_geographic_coverage": "Bavaria, Germany",
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
            profile["default_source"] = SOURCE_IN_SITU

    profile["dataset_name"] = normalized
    return profile
