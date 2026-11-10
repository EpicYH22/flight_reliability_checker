
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
# 2. Load Model and Historical Data
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
# 3. Prediction Function
# ==========================================

def predict_flight_reliability(
    month,
    carrier,
    origin,
    destination,
    day_type,
    schedule,
    holiday,
    bad_weather
):

    # Retrieve historical airline information
    carrier_data = lookups["carrier"][carrier]

    # Retrieve historical state delay rates
    origin_delay = lookups["origin"][origin]
    destination_delay = lookups["destination"][destination]

    # Retrieve typical distance category
    distance = lookups["distance"].get(
        (carrier, origin, destination)
    )

    if distance is None:
        raise ValueError(
            "No historical distance category is available "
            "for the selected airline and route."
        )

    # Construct all 14 model predictors
    input_data = pd.DataFrame([{
        "month": month,
        "op_unique_carrier": carrier,
        "origin_state_nm": origin,
        "dest_state_nm": destination,
        "day_type": day_type,
        "schedule": schedule,
        "distance_group": distance,
        "is_holiday": holiday,
        "bad_weather": bad_weather,
        "carrier_historical_delay_rate":
            carrier_data["carrier_historical_delay_rate"],
        "origin_historical_delay_rate": origin_delay,
        "destination_historical_delay_rate": destination_delay,
        "avg_aircraft_seats":
            carrier_data["avg_aircraft_seats"],
        "historical_load_factor":
            carrier_data["historical_load_factor"]
    }])

    # Ensure correct model feature order
    input_data = input_data[list(model.feature_names_in_)]

    # Predict probability of delay
    delay_probability = model.predict_proba(input_data)[0, 1]

    # Calculate Flight Reliability Score
    reliability_score = (1 - delay_probability) * 100

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
# 4. Website Header
# ==========================================

st.title("✈️ Flight Reliability Checker")

st.write(
    "Planning your next flight? Select your airline "
    "and enter your flight details to estimate "
    "the probability of a departure delay."
)

st.divider()

# ==========================================
# 5. Flight Details
# ==========================================

st.subheader("Flight Details")

# ------------------------------------------
# 5.1 Airline Selection
# ------------------------------------------

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

    # Filter records for selected airline
    airline_df = flight_options[
        flight_options["op_unique_carrier"] == carrier
    ].copy()

    st.caption(
        f"Showing historical flight options for {carrier}"
    )

    # --------------------------------------
    # 5.2 Available Options
    # --------------------------------------

    months = sorted(
        airline_df["month"]
        .dropna()
        .unique()
        .tolist()
    )

    day_types = sorted(
        airline_df["day_type"]
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

    # --------------------------------------
    # 5.3 Dropdown Interface
    # --------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        # Departure Month
        month = st.selectbox(
            "Departure Month",
            months,
            format_func=lambda x:
                calendar.month_name[int(x)],
            key="month"
        )

        # Day Type
        day_type = st.selectbox(
            "Day Type",
            day_types,
            key="day_type"
        )

        # Origin State
        origin = st.selectbox(
            "Origin State",
            origins,
            index=None,
            placeholder="Select origin state",
            key="origin"
        )

        # National Holiday
        holiday = st.selectbox(
            "National Holiday",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No",
            key="holiday"
        )

    with col2:

        # Departure Period
        schedule = st.selectbox(
            "Departure Period",
            schedules,
            key="schedule"
        )

        # Bad Weather
        bad_weather = st.selectbox(
            "Bad Weather",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No",
            key="bad_weather"
        )

    # --------------------------------------
    # 5.4 Destination Depends on Origin
    # --------------------------------------

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

    # Clear previous destination if no longer valid
    if (
        "destination" in st.session_state
        and st.session_state["destination"] not in destinations
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

    # --------------------------------------
    # 5.5 Prediction Button
    # --------------------------------------

    st.divider()

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
    # 6. Prediction Results
    # ======================================

    if submitted:

        try:

            result = predict_flight_reliability(
                month,
                carrier,
                origin,
                destination,
                day_type,
                schedule,
                holiday,
                bad_weather
            )

            st.divider()

            st.subheader(
                "Flight Reliability Results"
            )

            # --------------------------------
            # Selected Flight Details
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
                    f"**Origin:** {origin}"
                )

                st.write(
                    f"**Destination:** {destination}"
                )

                st.write(
                    f"**Day Type:** {day_type}"
                )

                st.write(
                    f"**Departure Period:** {schedule}"
                )

                st.write(
                    f"**National Holiday:** "
                    f"{'Yes' if holiday == 1 else 'No'}"
                )

                st.write(
                    f"**Bad Weather:** "
                    f"{'Yes' if bad_weather == 1 else 'No'}"
                )

            # --------------------------------
            # Prediction Metrics
            # --------------------------------

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

            # --------------------------------
            # Reliability Tier
            # --------------------------------

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
                    "low predicted risk of "
                    "departure delay."
                )

            elif tier == "Medium Reliability":

                st.warning(
                    "Medium Reliability"
                )

                st.write(
                    "This flight has a moderate "
                    "predicted risk of "
                    "departure delay."
                )

            else:

                st.error(
                    "Low Reliability"
                )

                st.write(
                    "This flight has a relatively "
                    "high predicted risk of "
                    "departure delay."
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
                "The Flight Reliability Score is "
                "calculated using historical flight "
                "characteristics and predicted delay "
                "probabilities. It is not a guarantee "
                "of actual flight performance."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )



