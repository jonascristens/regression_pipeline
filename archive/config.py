from collections import defaultdict

import pandas as pd

# paths
DATADIR_DEV = "data_dev"
DATADIR_PILOT = "data_pilot"
DATADIR_RTP = "data_rtp"
DATADIR = "Data/training-data-POC-countries"

# EDA_REPORT_NOTEBOOK = 'reports-generic-full.ipynb'
EDA_REPORT_NOTEBOOK = "reports-generic-full+.ipynb"
MON_REPORT_NOTEBOOK = "monitoring-1-reports.ipynb"
MON_BENEFIT_NOTEBOOK = "monitoring-1-reports-mdd-benefits.ipynb"

NB_CONFIG_FILENAME = "tmp_report_config.py"

PLOTDIR = "plots"
MODELS_DIR = "models/models_test"


# def get_org_id(datadir):
#     import pandas as pd
#     return pd.read_csv(f"{datadir}/organizations.csv", index_col=0).organization_id.to_dict()


def get_organizations(data_dir):
    ORGANIZATIONS = pd.read_csv(f"{data_dir}/organizations.csv", index_col=0)
    ORGANIZATION_ID = ORGANIZATIONS.organization_id.to_dict()
    return ORGANIZATIONS, ORGANIZATION_ID


# DEV countries
ORGANIZATION_ID = {
    "Argentina": "DHL AR",
    "Australia": "DHL AU",
    "Bangladesh": "DHL BD",
    "Belgium": "DHL Belgium",
    "Bulgaria": "DHL BG",
    "Brazil": "DHL BR",
    "Canada": "DHL CA",
    "Colombia": "DHL CO",
    "Cyprus": "DHL CY",
    "Croatia": "DHL HR",
    "Denmark": "DHL Denmark",
    "Dominican_Republic": "DHL DO",
    "Ecuador": "DHL EC",
    "El_Salvador": "DHL SV",
    "France": "DHL FR",
    "Germany": "DHL Germany",
    "Great_Britain": "DHL GB",
    "Greece": "DHL GR",
    "Honduras": "DHL HN",
    "Hong_Kong": "DHL HK",
    "Hungary": "DHL HU",
    "Indonesia": "DHL ID",
    "Ireland": "DHL Ireland",
    "Israel": "DHL Israel",
    "Italy": "DHL Italy",
    "Japan": "DHL JP",
    "Korea": "DHL KR",
    # 'Jamaica_Trinidad_n_Tobago' : 'DHL CARIBBEAN',
    "Malaysia": "DHL MY",
    "Mexico": "DHL MX",
    "Netherlands": "DHL Netherlands",
    "Nigeria": "DHL NG",
    "Norway": "DHL Norway",
    "Pakistan": "DHL PK",
    "Peru": "DHL PE",
    "Philippines": "DHL PH",
    "Poland": "DHL Poland-INT",
    "Portugal": "DHL Portugal",
    "Saudi_Arabia": "DHL SA",
    "South_Africa": "DHL ZA",
    "Spain": "DHL Spain",
    "Sweden": "DHL Sweden",
    "Switzerland": "DHL Switzerland",
    "Taiwan": "DHL TW",
    "Thailand": "DHL TH",
    "United_Arab_Emirates": "DHL AE",
    "Vietnam": "DHL VN",
    "Australia": "DHL AU",
    "Austria": "DHL AT",
    "Chile": "DHL CL",
    "Czech_Republic": "DHL CZ",
    "New_Zealand": "DHL NZ",
    # 'Switzerland' : 'DHL Switzerland',
    "Finland": "DHL Finland",
    "Russia": "DHL Russia",
    "United_States": "DHL US",
    "Kenya": "DHL KE",
    "Sri_Lanka": "DHL LK",
}


# Train/test spans for recent models
# SPANS = {
#     'Argentina': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Austria': ('2021-01-01', '2022-08-01', '2022-08-01', '2022-10-31'),
#     'Australia': ('2021-07-01', '2022-03-01', '2022-03-01', '2022-05-31'), # pre-mdd
#     'Bangladesh': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Belgium': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Brazil': ('2022-08-01', '2023-02-01', '2023-02-01', '2023-04-15'),
#     'Canada': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Chile': ('2021-01-01', '2022-08-01', '2022-08-01', '2022-10-31'),
#     # 'Colombia': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Colombia': ('2022-08-01', '2023-02-01', '2023-02-01', '2023-04-15'),
#     'Cyprus': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Czech_Republic': ('2021-01-01', '2022-08-01', '2022-08-01', '2022-10-31'),
#     'Denmark': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Dominican_Republic': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Ecuador': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'El_Salvador': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     # 'France': ('2021-01-01', '2022-08-01', '2022-08-01', '2022-10-31'),
#     'France': ('2021-01-01', '2021-08-01', '2021-08-01', '2021-10-31'), # pre-mdd
#     'Germany': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Great_Britain': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Greece': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Honduras': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Hong_Kong': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Hungary': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Indonesia': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Ireland': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Israel': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     # 'Italy': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Italy': ('2021-08-01', '2022-06-01', '2022-06-01', '2022-11-30'), # pre-mdd longer
#     'Jamaica': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Japan': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Korea': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Malaysia': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Mexico': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Netherlands': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'New_Zealand': ('2021-01-01', '2022-03-27', '2022-03-27', '2023-02-28'),
#     'Nigeria': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Norway': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Pakistan': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Peru': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Philippines': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Poland': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Portugal': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     # 'Saudi_Arabia': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Saudi_Arabia': ('2022-08-01', '2023-02-01', '2023-02-01', '2023-04-15'),
#     'South_Africa': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Spain': ('2021-01-01', '2022-08-01', '2022-08-01', '2022-10-31'),
#     'Sweden': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Switzerland': ('2021-01-01', '2022-08-01', '2022-08-01', '2022-10-31'),
#     # 'Taiwan': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Taiwan': ('2022-08-01', '2023-02-01', '2023-02-01', '2023-04-15'),
#     'Thailand': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     # 'United_Arab_Emirates': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'United_Arab_Emirates': ('2022-08-01', '2023-02-01', '2023-02-01', '2023-04-15'),
#     'Vietnam': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Finland': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
#     'Russia': ('2021-08-01', '2022-02-01', '2022-02-01', '2022-04-15'),
# }

# SPANS = defaultdict(lambda: ('2022-10-01', '2023-04-01', '2023-04-01', '2023-07-01'))
# SPANS = defaultdict(lambda: ('2022-12-01', '2023-06-01', '2023-06-01', '2023-08-01'))
SPANS = defaultdict(lambda: ("2022-11-01", "2023-05-01", "2023-05-01", "2023-08-01"))


# PILOT countries

PILOT_COUNTRIES = ["Germany", "Australia", "New_Zealand", "Chile", "Czech_Republic"]

# Model columns
target = "discount_shp"

bin_cols = ["ecomm"]

num_cols = [
    "log_published_revenue",
    "log_published_revenue_sq",
    "log_potential_revenue",
    "log_offer_published_revenue",
    "log_shipments",
    "log_wps",
    "log_weight",
    "PROD.MIX:TDEXPORT",
    "PROD.MIX:TDIMPORT",
    "PROD.MIX:TD3RD",
    "PROD.MIX:DDEXPORT",
    "PROD.MIX:DDIMPORT",
    "PROD.MIX:DD3RD",
    "PROD.MIX:TDDOM",
    "PROD.MIX:TD3DOM",
    "PROD.MIX:OTHER",
    "ecomm",
]


cat_cols = [
    "product_cluster_shp",
    "loyalty_code_cust",
    "industry",  # 'industry_code_cust',
    "opportunity_type_oppo",
    #    'reason_for_offering_off', # new, from offer
    "lead_cust",  # 'reason_for_lead_cust'
    #    'lead_oppo', # 'reason_for_lead_oppo' - new
    "initiator_cust",  # 'lead_source_type_cust'
    #    'initiator_oppo', # 'lead_source_type_oppo' - new
    "competitor_oppo",  # 'primary_competitor_oppo'
    #    'competitor_cust' # 'competitor_cust' - additional
]

# cat_cols_to_check = [
#     'sales_channel_off',
# ]

# num_cols_to_check = [
#     'opportunity_ocr_oppo',
#     'existing_committed_revenue_oppo',
#     'existing_potential_revenue_oppo',
#     'new_committed_revenue_oppo',
#     'new_potential_revenue_oppo',
#     'no_of_shpmnts_order_lns_oppo',
#     'committed_revenue_oppo',
#     'offer_discount',
#     'offer_products',
# ]


# PROD_ORGANIZATION_ID = {
#     'Austria': 'DHL AT',
#     'Chile': 'DHL CL',
#     'Czech_Republic': 'DHL CZ',
#     'France': 'DHL FR',
#     'New_Zealand': 'DHL NZ',
#     'Spain': 'DHL Spain',
#     'Switzerland': 'DHL Switzerland',
#     'Ireland': 'DHL Ireland',
#     'Australia': 'DHL AU',
#     'Argentina': 'DHL AR',
#     'Ecuador': 'DHL EC',
#     'Malaysia': 'DHL MY',
#     'Philippines': 'DHL PH'
# }

# COUNTRY_ISO_CODES = {
#     'Spain': 724,
#     'France': 250,
#     'Switzerland': 756
# }

FEATURE_NAMES_OUT = [
    "bin__ecomm",
    "cat__product_cluster_shp_TD3rd",
    "cat__product_cluster_shp_TDExport",
    "cat__product_cluster_shp_TDImport",
    "cat__loyalty_code_cust_Downtrader",
    "cat__loyalty_code_cust_Inactive",
    "cat__loyalty_code_cust_Lost",
    "cat__loyalty_code_cust_New",
    "cat__loyalty_code_cust_Non-Billed",
    "cat__loyalty_code_cust_Regained",
    "cat__loyalty_code_cust_Stable",
    "cat__loyalty_code_cust_Uptrader",
    "cat__loyalty_code_cust_missing",
    "cat__industry_IND:AUTO",
    "cat__industry_IND:CHEM",
    "cat__industry_IND:CSMR",
    "cat__industry_IND:ENGM",
    "cat__industry_IND:ENGY",
    "cat__industry_IND:FASH",
    "cat__industry_IND:FINS",
    "cat__industry_IND:LSHE",
    "cat__industry_IND:PROF",
    "cat__industry_IND:PSEC",
    "cat__industry_IND:RETL",
    "cat__industry_IND:TECH",
    "cat__industry_missing",
    "cat__opportunity_type_oppo_Acquisition",
    "cat__opportunity_type_oppo_Create Account",
    "cat__opportunity_type_oppo_Cross Selling",
    "cat__opportunity_type_oppo_One-Off/Seasonal",
    "cat__opportunity_type_oppo_Pricing",
    "cat__opportunity_type_oppo_Renegotiation",
    "cat__opportunity_type_oppo_Upselling",
    "cat__opportunity_type_oppo_missing",
    "cat__lead_cust_LEAD:CAMPAIGN",
    "cat__lead_cust_LEAD:CASH",
    "cat__lead_cust_LEAD:EXISTING",
    "cat__lead_cust_LEAD:NEW",
    "cat__lead_cust_LEAD:OTHER",
    "cat__lead_cust_LEAD:WEB",
    "cat__lead_cust_missing",
    "cat__initiator_cust_INIT:?",
    "cat__initiator_cust_INIT:CUSTOMER",
    "cat__initiator_cust_INIT:DHL",
    "cat__initiator_cust_missing",
    "cat__competitor_oppo_COMP:DHL",
    "cat__competitor_oppo_COMP:FEDEX",
    "cat__competitor_oppo_COMP:FR.FWD",
    "cat__competitor_oppo_COMP:NONE",
    "cat__competitor_oppo_COMP:NZPOST",
    "cat__competitor_oppo_COMP:OTHER",
    "cat__competitor_oppo_COMP:UPS",
    "cat__competitor_oppo_missing",
    "num__log_published_revenue",
    "num__log_published_revenue_sq",
    "num__log_potential_revenue",
    "num__log_offer_published_revenue",
    "num__log_shipments",
    "num__log_wps",
    "num__log_weight",
    "num__PROD.MIX:TDEXPORT",
    "num__PROD.MIX:TDIMPORT",
    "num__PROD.MIX:TD3RD",
    "num__PROD.MIX:DDEXPORT",
    "num__PROD.MIX:DDIMPORT",
    "num__PROD.MIX:DD3RD",
    "num__PROD.MIX:TDDOM",
    "num__PROD.MIX:TD3DOM",
    "num__PROD.MIX:OTHER",
]
