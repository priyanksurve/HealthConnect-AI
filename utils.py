"""
utils.py
--------
Location lookup, distance calculation, hospital search (Google Places),
and SIMULATED bed availability.

IMPORTANT: No public API (including Google Places) exposes real-time
hospital bed occupancy - that data lives only inside each hospital's
own internal systems. simulate_bed_availability() generates realistic
demo numbers, seeded by the hospital's place_id + today's date, so a
given hospital shows the SAME numbers all day (not random on every
refresh) but changes day to day.
"""

import math
import random
import hashlib
from datetime import date

import requests
import streamlit as st


# =========================================================
# LOCATION LOOKUP (OpenStreetMap Nominatim - free, no key needed)
# =========================================================

def get_coordinates(location):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location, "format": "json", "limit": 1}
    headers = {"User-Agent": "PNEUMO-AI-Healthcare-Research-Application"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None, None
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        return None, None


# =========================================================
# HAVERSINE DISTANCE
# =========================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    earth_radius = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius * c


# =========================================================
# SIMULATED BED AVAILABILITY
# =========================================================

def simulate_bed_availability(hospital_place_id: str):
    seed_str = f"{hospital_place_id}-{date.today().isoformat()}"
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)

    total_beds = rng.randint(50, 300)
    occupancy_rate = rng.uniform(0.55, 0.95)
    occupied_beds = int(total_beds * occupancy_rate)
    available_beds = max(total_beds - occupied_beds, 0)

    return {
        "total_beds": total_beds,
        "occupied_beds": occupied_beds,
        "available_beds": available_beds,
    }


# =========================================================
# GOOGLE PLACES - NEARBY HOSPITALS
# -----------------------------------------------------
# NOTE: This uses the NEW Places API (searchNearby), not the
# old "Nearby Search (Legacy)" endpoint. Most Google Cloud
# projects created recently only have "Places API (New)"
# enabled by default - the old legacy endpoint often returns
# REQUEST_DENIED / PERMISSION_DENIED on those projects, which
# is the most common reason hospital search silently fails.
#
# To use this, in Google Cloud Console:
#   1. Go to "APIs & Services" -> "Library"
#   2. Search "Places API (New)" and click Enable
#   3. Make sure Billing is enabled on the project
#   4. Make sure your API key has no restriction that blocks
#      server-side calls (an "HTTP referrer" restriction will
#      block this, since Streamlit calls it from the server,
#      not a browser). Use "None" or an IP restriction instead.
# =========================================================

def _normalize_place_name(place):
    display_name = place.get("displayName")
    if isinstance(display_name, dict):
        return display_name.get("text") or display_name.get("name") or "Hospital"
    if isinstance(display_name, str):
        return display_name
    return "Hospital"


def _normalize_place_address(place):
    address = place.get("formattedAddress") or place.get("address")
    if isinstance(address, dict):
        return (
            address.get("formattedAddress")
            or address.get("text")
            or address.get("address")
            or "Address unavailable"
        )
    if isinstance(address, str):
        return address
    return "Address unavailable"


def find_nearby_hospitals_osm(latitude, longitude, radius):
    query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="hospital"](around:{int(min(radius, 50000))},{latitude},{longitude});
          way["amenity"="hospital"](around:{int(min(radius, 50000))},{latitude},{longitude});
          relation["amenity"="hospital"](around:{int(min(radius, 50000))},{latitude},{longitude});
        );
        out center;
    """

    headers = {
        "User-Agent": "PNEUMO-AI-Healthcare-Research-Application",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    for overpass_url in overpass_urls:
        try:
            response = requests.post(overpass_url, data={"data": query}, headers=headers, timeout=30)
            if response.status_code != 200:
                continue

            data = response.json()
            elements = data.get("elements", [])

            hospitals = []
            for element in elements:
                tags = element.get("tags", {})
                hospital_lat = element.get("lat") or element.get("center", {}).get("lat")
                hospital_lon = element.get("lon") or element.get("center", {}).get("lon")
                if hospital_lat is None or hospital_lon is None:
                    continue

                distance = calculate_distance(latitude, longitude, hospital_lat, hospital_lon)
                place_id = f"osm-{element.get('type', 'unknown')}-{element.get('id', '')}"
                name = tags.get("name") or "Hospital"
                address_parts = []
                for key in [
                    "addr:street", "addr:housenumber", "addr:city",
                    "addr:state", "addr:postcode", "addr:country"
                ]:
                    if tags.get(key):
                        address_parts.append(tags.get(key))
                address = ", ".join(address_parts) if address_parts else tags.get("addr:full") or "Address unavailable"

                beds = simulate_bed_availability(place_id)
                hospitals.append({
                    "name": name,
                    "latitude": hospital_lat,
                    "longitude": hospital_lon,
                    "distance": distance,
                    "address": address,
                    "rating": tags.get("rating", "N/A"),
                    "reviews": tags.get("review_count", 0) or 0,
                    "open_now": None,
                    "place_id": place_id,
                    "total_beds": beds["total_beds"],
                    "occupied_beds": beds["occupied_beds"],
                    "available_beds": beds["available_beds"],
                })

            hospitals.sort(key=lambda x: x["distance"])
            return hospitals
        except requests.exceptions.RequestException:
            continue
        except Exception:
            continue

    st.error(
        "❌ OpenStreetMap hospital search failed on all endpoints. "
        "If your Google key is not valid for Places API, the app will still try to search by OpenStreetMap."
    )
    return []


def find_nearby_hospitals(latitude, longitude, radius):
    google_api_key = ""
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

    if not google_api_key or google_api_key.strip() == "your-key":
        st.warning(
            "⚠️ Google API key is not configured or is using the placeholder value. "
            "Falling back to OpenStreetMap hospital search."
        )
        return find_nearby_hospitals_osm(latitude, longitude, radius)

    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": google_api_key,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.formattedAddress,"
            "places.location,places.rating,places.userRatingCount,"
            "places.currentOpeningHours.openNow"
        ),
    }
    body = {
        "includedTypes": ["hospital"],
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": min(radius, 50000),
            }
        },
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
        if response.status_code != 200:
            try:
                error_detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                error_detail = response.text
            st.warning(
                f"⚠️ Google Places API failed ({response.status_code}): {error_detail}. "
                "Falling back to OpenStreetMap hospital search."
            )
            st.caption(
                "Common fixes: enable 'Places API (New)' in Google Cloud Console, "
                "make sure billing is enabled on the project, and make sure your "
                "API key has no HTTP-referrer restriction (server calls need "
                "'None' or IP-address restriction instead)."
            )
            return find_nearby_hospitals_osm(latitude, longitude, radius)

        data = response.json()
        places = data.get("places", [])
        if not places:
            st.info("ℹ️ Google Places returned no hospitals. Trying OpenStreetMap fallback.")
            return find_nearby_hospitals_osm(latitude, longitude, radius)

        hospitals = []
        for place in places:
            location_data = place.get("location", {})
            hospital_lat = location_data.get("latitude")
            hospital_lon = location_data.get("longitude")
            if hospital_lat is None or hospital_lon is None:
                continue

            distance = calculate_distance(latitude, longitude, hospital_lat, hospital_lon)
            place_id = place.get("id", "")
            display_name = _normalize_place_name(place)
            opening_hours = place.get("currentOpeningHours", {}) or {}
            beds = simulate_bed_availability(place_id)

            hospitals.append({
                "name": display_name,
                "latitude": hospital_lat,
                "longitude": hospital_lon,
                "distance": distance,
                "address": _normalize_place_address(place),
                "rating": place.get("rating", "N/A"),
                "reviews": place.get("userRatingCount", 0),
                "open_now": opening_hours.get("openNow", None),
                "place_id": place_id,
                "total_beds": beds["total_beds"],
                "occupied_beds": beds["occupied_beds"],
                "available_beds": beds["available_beds"],
            })

        hospitals.sort(key=lambda x: x["distance"])
        return hospitals
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error: {str(e)}")
        st.warning("⚠️ Falling back to OpenStreetMap hospital search.")
        return find_nearby_hospitals_osm(latitude, longitude, radius)
    except Exception as e:
        st.error(f"❌ Hospital search error: {str(e)}")
        st.warning("⚠️ Falling back to OpenStreetMap hospital search.")
        return find_nearby_hospitals_osm(latitude, longitude, radius)
