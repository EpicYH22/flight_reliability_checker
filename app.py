
import streamlit as st
import pandas as pd
import joblib
import calendar

# ==========================================
# 1. Website Configuration
# ==========================================

st.set_page_config(
    page_title="Flight Reliability Checker",
    page_icon="✈️",
    layout="centered"
)

# ==========================================
# 2. Load Model and Historical Files
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("flight_model.pkl")


@st.cache_resource
def load_lookups():
    return joblib.load("flight_lookups.pkl")


@st.cache_data
def load_flight_options():
    return joblib.load("flight_options.pkl")


model = load_model()
lookups = load_lookups()
flight_options = load_flight_options()

# ==========================================
# 3. Model Features
# ==========================================

model_features = [
    "month",
    "day_of_week",
    "op_unique_carrier",
    "schedule",
    "distance_group",
    "is_holiday",
    "carrier_historical_delay_rate",
    "origin_historical_delay_rate",
    "destination_historical_delay_rate",
    "avg_aircraft_seats",
    "historical_passengers",
    "historical_load_factor",
    "historical_departures",
    "temperature_2m",
    "precipitation",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "is_first_flight",
    "previous_flight_departure_delay"
]

# ==========================================
# 4. Historical Lookup Function
# ==========================================

def get_historical_values(
    carrier, origin, destination, month
):

    exact_key = (
        carrier,
        origin,
        destination,
        month
    )

    route_key = (
        carrier,
        origin,
        destination
    )

    exact_data = lookups["exact"].get(
        exact_key, {}
    )

    route_data = lookups["route"].get(
        route_key, {}
    )

    carrier_data = lookups["carrier"].get(
        carrier, {}
    )

    global_data = lookups["global"]

    values = {}

    features = (
        lookups["numerical_features"]
        + ["distance_group"]
    )

    # Select the most specific available value
    for feature in features:

        candidates = [
            exact_data.get(feature),
            route_data.get(feature),
            carrier_data.get(feature),
            global_data.get(feature)
        ]

        value = next(
            (
                x for x in candidates
                if x is not None
                and pd.notna(x)
            ),
            None
        )

        if value is None:
            raise ValueError(
                f"No historical value for {feature}"
            )

        values[feature] = value

    return values


# ==========================================
# 5. Prediction Function
# ==========================================

def predict_flight_reliability(
    month,
    day_of_week,
    carrier,
    schedule,
    origin,
    destination,
    holiday
):

    historical = get_historical_values(
        carrier,
        origin,
        destination,
        month
    )

    input_values = {
        "month": month,
        "day_of_week": day_of_week,
        "op_unique_carrier": carrier,
        "schedule": schedule,
        "is_holiday": holiday
    }

    input_values.update(historical)

    input_data = pd.DataFrame(
        [input_values]
    )

    # Ensure all model features exist
    missing = [
        col for col in model_features
        if col not in input_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model predictors: {missing}"
        )

    input_data = input_data[model_features]

    # Predict delay probability
    delay_probability = (
        model.predict_proba(input_data)[0, 1]
    )

    reliability_score = (
        1 - delay_probability
    ) * 100

    if reliability_score >= 85:
        tier = "High Reliability"
    elif reliability_score >= 75:
        tier = "Medium Reliability"
    else:
        tier = "Low Reliability"

    return {
        "Delay Probability (%)":
            round(delay_probability * 100, 2),
        "Reliability Score":
            round(reliability_score, 2),
        "Reliability Tier": tier
    }


# ==========================================
# 6. Website Header
# ==========================================

st.title("✈️ Flight Reliability Checker")

st.write(
    "Helping airlines anticipate and manage flight delays "
    "through data-driven predictions. Enter flight details "
    "to estimate departure delay probabilities and assess "
    "flight reliability."
)

st.divider()

st.subheader("Flight Details")

# ==========================================
# 7. Airline Selection
# ==========================================

airlines = sorted(
    flight_options["op_unique_carrier"]
    .dropna()
    .unique()
    .tolist()
)

carrier = st.selectbox(
    "Airline Code",
    airlines,
    index=None,
    placeholder="Select an airline",
    key="carrier"
)

if carrier is None:

    st.info(
        "Select an airline to display "
        "the available flight details."
    )

else:

    airline_df = flight_options[
        flight_options["op_unique_carrier"] == carrier
    ].copy()

    st.caption(
        f"Showing historical flight options for {carrier}"
    )

    # ======================================
    # 8. Available Dropdown Options
    # ======================================

    months = sorted(
        airline_df["month"]
        .dropna()
        .unique()
        .tolist()
    )

    schedules = sorted(
        airline_df["schedule"]
        .dropna()
        .unique()
        .tolist()
    )

    origins = sorted(
        airline_df["origin_state_nm"]
        .dropna()
        .unique()
        .tolist()
    )

    # ======================================
    # 9. Two-Column Inputs
    # ======================================

    col1, col2 = st.columns(2)

    with col1:

        month = st.selectbox(
            "Departure Month",
            months,
            format_func=lambda x:
                calendar.month_name[int(x)],
            key="month"
        )

        # Day of Week
        day_options = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        day_name = st.selectbox(
            "Day of Week",
            day_options,
            key="day_of_week"
        )

    with col2:

        holiday = st.selectbox(
            "National Holiday",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No",
            key="holiday"
        )

    # ======================================
    # 10. Departure Period
    # ======================================

    schedule = st.selectbox(
        "Departure Period",
        schedules,
        key="schedule"
    )

    # ======================================
    # 11. Origin State
    # ======================================

    origin = st.selectbox(
        "Origin State",
        origins,
        index=None,
        placeholder="Select origin state",
        key="origin"
    )

    # ======================================
    # 12. Destination State
    # ======================================

    if origin is not None:

        route_df = airline_df[
            airline_df["origin_state_nm"] == origin
        ]

        destinations = sorted(
            route_df["dest_state_nm"]
            .dropna()
            .unique()
            .tolist()
        )

    else:
        destinations = []

    if (
        "destination" in st.session_state
        and st.session_state["destination"]
        not in destinations
    ):
        st.session_state["destination"] = None

    destination = st.selectbox(
        "Destination State",
        destinations,
        index=None,
        placeholder="Select destination state",
        disabled=(origin is None),
        key="destination"
    )

    # ======================================
    # 13. Prediction Button
    # ======================================

    st.divider()

    st.caption(
        "Weather conditions, historical passenger "
        "volumes and previous-flight characteristics "
        "are estimated using historical reference data."
    )

    submitted = st.button(
        "Predict Flight Reliability",
        type="primary",
        use_container_width=True,
        disabled=(
            origin is None
            or destination is None
        )
    )

    # ======================================
    # 14. Prediction Results
    # ======================================

    if submitted:

        try:

            # IMPORTANT:
            # Adjust this mapping if your model
            # uses a different day-of-week encoding.
            day_of_week = day_options.index(
                day_name
            )

            result = predict_flight_reliability(
                month,
                day_of_week,
                carrier,
                schedule,
                origin,
                destination,
                holiday
            )

            st.divider()

            st.subheader(
                "Flight Reliability Results"
            )

            with st.expander(
                "Selected Flight Details",
                expanded=True
            ):

                st.write(
                    f"**Airline:** {carrier}"
                )

                st.write(
                    f"**Month:** "
                    f"{calendar.month_name[int(month)]}"
                )

                st.write(
                    f"**Day of Week:** {day_name}"
                )

                st.write(
                    f"**Departure Period:** {schedule}"
                )

                st.write(
                    f"**Origin:** {origin}"
                )

                st.write(
                    f"**Destination:** {destination}"
                )

                st.write(
                    f"**National Holiday:** "
                    f"{'Yes' if holiday == 1 else 'No'}"
                )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Delay Probability",
                    f"{result['Delay Probability (%)']}%"
                )

            with col2:

                st.metric(
                    "Flight Reliability Score",
                    f"{result['Reliability Score']}/100"
                )

            tier = result["Reliability Tier"]

            st.subheader(
                "Flight Reliability Tier"
            )

            if tier == "High Reliability":

                st.success(
                    "High Reliability"
                )

                st.write(
                    "This flight has a relatively "
                    "low predicted risk of delay."
                )

            elif tier == "Medium Reliability":

                st.warning(
                    "Medium Reliability"
                )

                st.write(
                    "This flight has a moderate "
                    "predicted risk of delay."
                )

            else:

                st.error(
                    "Low Reliability"
                )

                st.write(
                    "This flight has a relatively "
                    "high predicted risk of delay."
                )

            st.progress(
                result["Reliability Score"] / 100
            )

            st.caption(
                "Predictions are based on historical "
                "flight data and estimated conditions. "
                "They do not guarantee actual "
                "flight performance."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )




