
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. Website configuration
# ==========================================

st.set_page_config(
    page_title="Flight Reliability Checker",
    page_icon="✈️",
    layout="centered"
)

# ==========================================
# 2. Load trained model and dropdowns
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("flight_model.pkl")

@st.cache_resource
def load_lookups():
    return joblib.load("flight_lookups.pkl")

model = load_model()
lookups = load_lookups()

# ==========================================
# 3. Prediction function
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

    # Retrieve historical carrier information
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
            "for this airline and state pair."
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

    # Predict probability
    delay_probability = model.predict_proba(input_data)[0, 1]

    # Calculate reliability score
    reliability_score = (1 - delay_probability) * 100

    # Assign reliability tier
    if reliability_score >= 85:
        tier = "High Reliability"
    elif reliability_score >= 75:
        tier = "Medium Reliability"
    else:
        tier = "Low Reliability"

    return {
        "Delay Probability (%)": round(delay_probability * 100, 2),
        "Reliability Score": round(reliability_score, 2),
        "Reliability Tier": tier
    }


# ==========================================
# 4. Website header
# ==========================================

st.title("✈️ Flight Reliability Checker")

st.write(
    "Planning your next flight? Enter your flight "
    "details to estimate the likelihood of a "
    "departure delay based on historical flight data."
)

st.divider()

# ==========================================
# 5. Flight details dropdowns
# ==========================================


# ==========================================
# Flight Details
# ==========================================

st.subheader("Flight Details")

with st.form("flight_form"):

    col1, col2 = st.columns(2)

    with col1:

        month_option = st.selectbox(
            "Departure Month",
            options["months"],
            format_func=lambda x: x[0]
        )
        month = month_option[1]

        carrier = st.selectbox(
            "Airline",
            options["carriers"]
        )

        origin = st.selectbox(
            "Origin State",
            options["origins"]
        )

        day_type = st.selectbox(
            "Day Type",
            options["day_types"]
        )

    with col2:

        destination = st.selectbox(
            "Destination State",
            options["destinations"]
        )

        schedule = st.selectbox(
            "Departure Period",
            options["schedules"]
        )

        holiday_option = st.selectbox(
            "National Holiday",
            options["holidays"],
            format_func=lambda x: x[0]
        )
        holiday = holiday_option[1]

        bad_weather_label = st.selectbox(
            "Bad Weather",
            ["No", "Yes"]
        )
        bad_weather = 1 if bad_weather_label == "Yes" else 0

    submitted = st.form_submit_button(
        "Predict Flight Reliability",
        type="primary",
        use_container_width=True
    )


# ==========================================
# 6. Display prediction results
# ==========================================


# ==========================================
# Display Prediction Results
# ==========================================

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
        st.subheader("Flight Reliability Results")

        # Selected Flight Details
        with st.expander("Selected Flight Details", expanded=True):
            st.write(f"**Month:** {month_option[0]}")
            st.write(f"**Airline:** {carrier}")
            st.write(f"**Origin:** {origin}")
            st.write(f"**Destination:** {destination}")
            st.write(f"**Day Type:** {day_type}")
            st.write(f"**Departure Period:** {schedule}")
            st.write(f"**National Holiday:** {holiday_option[0]}")
            st.write(f"**Bad Weather:** {bad_weather_label}")

        # Prediction Metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Delay Probability",
                f"{result['Delay Probability (%)']}%"
            )

        with col2:
            st.metric(
                "Reliability Score",
                f"{result['Reliability Score']}/100"
            )

        # Reliability Tier
        tier = result["Reliability Tier"]

        if tier == "High Reliability":
            st.success("High Reliability")
            st.write(
                "This flight has a relatively low "
                "predicted risk of departure delay."
            )

        elif tier == "Medium Reliability":
            st.warning("Medium Reliability")
            st.write(
                "This flight has a moderate "
                "predicted risk of departure delay."
            )

        else:
            st.error("Low Reliability")
            st.write(
                "This flight has a relatively high "
                "predicted risk of departure delay."
            )

        # Reliability Score Progress Bar
        st.progress(
            result["Reliability Score"] / 100
        )

        st.caption(
            "The Flight Reliability Score is based on "
            "historical flight characteristics and "
            "predicted delay probabilities. It is not "
            "a guarantee of actual flight performance."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
