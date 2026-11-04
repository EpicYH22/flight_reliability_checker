
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

@st.cache_data
def load_options():
    return joblib.load("dropdown_options.pkl")

model = load_model()
options = load_options()

# ==========================================
# 3. Prediction function
# ==========================================

def predict_flight_reliability(
    month, carrier, origin, destination,
    day_type, schedule, holiday
):

    input_data = pd.DataFrame([{
        "month": month,
        "op_unique_carrier": carrier,
        "origin_state_nm": origin,
        "dest_state_nm": destination,
        "day_type": day_type,
        "schedule": schedule,
        "is_holiday": holiday
    }])

    delay_probability = model.predict_proba(input_data)[0, 1]

    reliability_score = (1 - delay_probability) * 100

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

    submitted = st.form_submit_button(
        "Predict Flight Reliability",
        type="primary",
        use_container_width=True
    )

# ==========================================
# 6. Display prediction results
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
            holiday
        )

        st.divider()

        st.subheader("Flight Reliability Results")

        with st.expander("Selected Flight Details"):
            st.write(f"**Month:** {month_option[0]}")
            st.write(f"**Airline:** {carrier}")
            st.write(f"**Origin:** {origin}")
            st.write(f"**Destination:** {destination}")
            st.write(f"**Day Type:** {day_type}")
            st.write(f"**Departure Period:** {schedule}")
            st.write(f"**National Holiday:** {holiday_option[0]}")

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
