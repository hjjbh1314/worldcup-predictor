"""球队 → 所属大洲足联(confederation)映射。

用于修正纯 Elo 的结构性偏差:各大洲是半封闭的积分池,只在世界杯等
跨洲赛事里才真正互相检验,所以纯 Elo 会高估弱区强队、低估强区球队。

队名以 martj42/international_results 数据集为准(如 "China PR" / "DR Congo")。
"""
from __future__ import annotations

_GROUPS = {
    "UEFA": [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
        "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
        "Czech Republic", "Czechoslovakia", "Denmark", "England", "Estonia",
        "Faroe Islands", "Finland", "France", "Georgia", "Germany",
        "East Germany", "German DR", "Gibraltar", "Greece", "Hungary", "Iceland",
        "Republic of Ireland", "Israel", "Italy", "Kazakhstan", "Kosovo",
        "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta",
        "Moldova", "Montenegro", "Netherlands", "North Macedonia",
        "Northern Ireland", "Norway", "Poland", "Portugal", "Romania",
        "Russia", "Soviet Union", "San Marino", "Scotland", "Serbia",
        "Serbia and Montenegro", "Yugoslavia", "Slovakia", "Slovenia",
        "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine", "Wales",
    ],
    "CONMEBOL": [
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
        "Paraguay", "Peru", "Uruguay", "Venezuela",
    ],
    "CONCACAF": [
        "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados", "Belize",
        "Bermuda", "Canada", "Cayman Islands", "Costa Rica", "Cuba",
        "Curaçao", "Dominica", "Dominican Republic", "El Salvador",
        "Grenada", "Guadeloupe", "Guatemala", "Guyana", "Haiti", "Honduras",
        "Jamaica", "Martinique", "Mexico", "Montserrat", "Nicaragua",
        "Panama", "Puerto Rico",
        "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Vincent and the Grenadines", "Suriname",
        "Trinidad and Tobago", "United States",
    ],
    "CAF": [
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
        "Cameroon", "Cape Verde", "Central African Republic", "Chad",
        "Comoros", "Congo", "DR Congo", "Djibouti", "Egypt",
        "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon",
        "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya",
        "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali",
        "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia",
        "Niger", "Nigeria", "Rwanda", "São Tomé and Príncipe", "Senegal",
        "Seychelles", "Sierra Leone", "Somalia", "South Africa",
        "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda",
        "Zambia", "Zimbabwe", "Zaïre", "Zanzibar",
    ],
    "AFC": [
        "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan",
        "Brunei", "Cambodia", "China PR", "Chinese Taipei", "Taiwan", "Guam",
        "Hong Kong", "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan",
        "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Macau", "Malaysia",
        "Maldives", "Mongolia", "Myanmar", "Nepal", "North Korea", "Oman",
        "Pakistan", "Palestine", "Philippines", "Qatar", "Saudi Arabia",
        "Singapore", "South Korea", "Sri Lanka", "Syria", "Tajikistan",
        "Thailand", "Timor-Leste", "Turkmenistan", "United Arab Emirates",
        "Uzbekistan", "Vietnam", "Yemen",
    ],
    "OFC": [
        "American Samoa", "Cook Islands", "Fiji", "New Caledonia",
        "New Zealand", "Papua New Guinea", "Samoa", "Solomon Islands",
        "Tahiti", "Tonga", "Vanuatu",
    ],
}

CONFEDERATION: dict[str, str] = {
    team: conf for conf, teams in _GROUPS.items() for team in teams
}


def get_confederation(team: str) -> str | None:
    return CONFEDERATION.get(team)
