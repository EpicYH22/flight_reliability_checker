
import streamlit as st
import pandas as pd
import joblib
import calendar

# ==========================================
# 1. Website Configuration
# ==========================================

st.set_page_config(
    page_title="Flight Delay Risk Assessment",
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

required_lookup_keys = {
    "exact",
    "route",
    "carrier",
    "global",
    "numerical_features"
}

missing_lookup_keys = (
    required_lookup_keys - set(lookups.keys())
)

if missing_lookup_keys:
    st.error(
        "Historical lookup file is outdated. "
        f"Missing keys: {sorted(missing_lookup_keys)}"
    )
    st.stop()

if hasattr(model, "feature_names_in_"):
    missing_model_features = (
        set(model_features) -
        set(model.feature_names_in_)
    )

    if missing_model_features:
        st.error(
            "The saved model does not contain all "
            "22 expected predictors."
        )
        st.stop()

# ==========================================
# 4. Historical Lookup Function
# ==========================================

def get_historical_values(
    carrier,
    origin,
    destination,
    month
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
                if x is not None and pd.notna(x)
            ),
            None
        )

        if value is None:
            raise ValueError(
                f"No historical value available "
                f"for {feature}"
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

    missing = [
        col for col in model_features
        if col not in input_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model predictors: {missing}"
        )

    input_data = input_data[model_features]

    # Predict probability of delay
    delay_probability = float(
        model.predict_proba(input_data)[0, 1]
    )

    # Calculate Flight Reliability Score
    reliability_score = (
        1 - delay_probability
    ) * 100

    # Assign reliability tier
    if reliability_score >= 85:
        tier = "High Reliability"
    elif reliability_score >= 75:
        tier = "Medium Reliability"
    else:
        tier = "Low Reliability"

    return {
        "Delay Probability (%)": round(
            delay_probability * 100, 2
        ),
        "Reliability Score": round(
            reliability_score, 2
        ),
        "Reliability Tier": tier
    }


# ==========================================
# 6. Website Header
# ==========================================

st.title("✈️ Flight Delay Risk Assessment")

st.write(
    "Supporting airlines in anticipating departure "
    "delays and making informed operational decisions "
    "through data-driven risk assessment."
)

st.divider()

st.subheader("Flight Operational Details")

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

    with col2:

        holiday = st.selectbox(
            "National Holiday",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No",
            key="holiday"
        )

    # ======================================
    # 10. Day of Week
    # ======================================

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

    # ======================================
    # 11. Departure Period
    # ======================================

    schedule = st.selectbox(
        "Departure Period",
        schedules,
        key="schedule"
    )

    # ======================================
    # 12. Origin State
    # ======================================

    origin = st.selectbox(
        "Origin State",
        origins,
        index=None,
        placeholder="Select origin state",
        key="origin"
    )

    # ======================================
    # 13. Destination State
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
    # 14. Prediction Button
    # ======================================

    st.divider()

    st.caption(
        "Historical passenger volumes, weather "
        "conditions and previous-flight characteristics "
        "are estimated from reference data. "
        "These are not live operational measurements."
    )

    submitted = st.button(
        "Assess Delay Risk",
        type="primary",
        use_container_width=True,
        disabled=(
            origin is None
            or destination is None
        )
    )

    # ======================================
    # 15. Prediction Results
    # ======================================

    if submitted:

        try:

            # Verify this matches training encoding:
            # Monday=0 through Sunday=6
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
                "Delay Risk Assessment"
            )

            # --------------------------------
            # Selected Operational Details
            # --------------------------------

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

            # --------------------------------
            # Prediction Metrics
            # --------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Delay Probability",
                    f"{result['Delay Probability (%)']}%"
                )

            with col2:

                st.metric(
                    "Flight Reliability Score",
                    f"{result['Reliability Score']}/100"
                )

            # --------------------------------
            # Operational Reliability Tier
            # --------------------------------

            tier = result["Reliability Tier"]

            st.subheader(
                "Operational Reliability Tier"
            )

            if tier == "High Reliability":

                st.success(
                    "High Reliability"
                )

                st.write(
                    "This flight has a relatively low "
                    "predicted delay risk. Continue "
                    "routine operational monitoring."
                )

            elif tier == "Medium Reliability":

                st.warning(
                    "Medium Reliability"
                )

                st.write(
                    "This flight has a moderate "
                    "predicted delay risk. Consider "
                    "reviewing relevant operational "
                    "conditions and monitoring updates."
                )

            else:

                st.error(
                    "Low Reliability"
                )

                st.write(
                    "This flight has a relatively high "
                    "predicted delay risk. Consider "
                    "prioritising further operational "
                    "assessment and contingency planning."
                )

            # --------------------------------
            # Reliability Progress Bar
            # --------------------------------

            st.progress(
                result["Reliability Score"] / 100
            )

            # --------------------------------
            # Disclaimer
            # --------------------------------

            st.caption(
                "This assessment uses historical "
                "flight characteristics and estimated "
                "operational conditions. Predictions "
                "are probabilistic and do not guarantee "
                "actual flight performance. Operational "
                "recommendations are general guidance."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )

# ==========================================
# 16. Model Information
# ==========================================

st.divider()

with st.expander("About This Predictive Model"):

    st.write(
        "This tool estimates departure delay "
        "probabilities using historical flight "
        "performance, airline characteristics, "
        "operational factors and weather variables."
    )

    st.write(
        "The Flight Reliability Score (FRS) "
        "is calculated as:"
    )

    st.latex(
        r"FRS = (1 - P(\mathrm{Delay})) \times 100"
    )

    st.write("**Reliability Tier Classification**")

    st.write(
        "High Reliability: FRS ≥ 85"
    )

    st.write(
        "Medium Reliability: 75 ≤ FRS < 85"
    )

    st.write(
        "Low Reliability: FRS < 75"
    )

    st.write("**Model Limitations**")

    st.write(
        "The model relies on historical flight "
        "performance and estimated operational "
        "conditions. It does not directly incorporate "
        "live weather forecasts, current aircraft "
        "rotations or real-time flight disruptions."
    )

    st.write(
        "Some input variables are automatically "
        "populated using historical reference values "
        "rather than flight-specific observations. "
        "Predictions should therefore be interpreted "
        "as indicative risk estimates."
    )





