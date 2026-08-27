import os
import joblib


# ==============================
# LOAD MODEL AND VECTORIZER
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "fake_review_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "tfidf_vectorizer.pkl"
)

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)


# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_review(review):

    # Convert review into TF-IDF numerical features
    review_tfidf = tfidf.transform([review])

    # Predict class
    prediction = model.predict(review_tfidf)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(review_tfidf)[0]

    # Get confidence
    confidence = max(probabilities) * 100

    return prediction, confidence


# ==============================
# USER INPUT
# ==============================

print("=" * 60)
print("       FAKE REVIEW DETECTOR")
print("=" * 60)

print("\nEnter a hotel review to analyze.")
print("Type 'exit' to quit.\n")


while True:

    try:
        review = input("Enter review: ").strip()

        # Exit condition
        if review.lower() == "exit":
            print("\nThanks for using Fake Review Detector!")
            break

        # Empty input validation
        if not review:
            print("Review cannot be empty. Please enter a valid review.\n")
            continue

        # Minimum length validation
        if len(review) < 10:
            print(
                "Review is too short. "
                "Please enter at least 10 characters.\n"
            )
            continue

        # Make prediction
        prediction, confidence = predict_review(review)

        # Display result
        print("\n" + "-" * 60)

        if prediction == "deceptive":
            print("Prediction: POTENTIALLY DECEPTIVE ⚠️")
        else:
            print("Prediction: LIKELY TRUTHFUL ✅")

        print(f"Confidence: {confidence:.2f}%")
        print("-" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user.")
        break

    except Exception as error:
        print(f"\nAn unexpected error occurred: {error}\n")