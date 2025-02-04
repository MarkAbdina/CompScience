# Dataset
emails = [
    {"Free": 1, "Offer": 0, "Money": 0, "Spam": 1},  # Spam
    {"Free": 1, "Offer": 1, "Money": 0, "Spam": 1},  # Spam
    {"Free": 0, "Offer": 0, "Money": 0, "Spam": 0},  # Not Spam
    {"Free": 1, "Offer": 1, "Money": 1, "Spam": 1},  # Spam
    {"Free": 0, "Offer": 0, "Money": 1, "Spam": 0},  # Not Spam
    {"Free": 0, "Offer": 1, "Money": 0, "Spam": 0},  # Not Spam
    {"Free": 1, "Offer": 0, "Money": 1, "Spam": 0},  # Spam
]

# New email to classify
new_email = {"Free": 0, "Offer": 0, "Money": 1}  # Free=Yes, Offer=No, Money=Yes

# Step 1: Calculate Prior Probabilities
def calculate_priors(data):
    total = len(data)
    spam_count = sum(1 for email in data if email["Spam"] == 1)
    not_spam_count = total - spam_count
    p_spam = spam_count / total
    p_not_spam = not_spam_count / total
    return p_spam, p_not_spam

p_spam, p_not_spam = calculate_priors(emails)

# Step 2: Calculate Likelihoods
def calculate_likelihoods(data, feature, value, label):
    filtered_data = [email for email in data if email["Spam"] == label]
    total_label = len(filtered_data)
    feature_count = sum(1 for email in filtered_data if email[feature] == value)
    return (feature_count + 1) / (total_label + 2)  # Laplace smoothing

# Likelihoods for Free=Yes
p_free_yes_spam = calculate_likelihoods(emails, "Free", 1, 1)
p_free_yes_not_spam = calculate_likelihoods(emails, "Free", 1, 0)

# Likelihoods for Offer=No
p_offer_no_spam = calculate_likelihoods(emails, "Offer", 0, 1)
p_offer_no_not_spam = calculate_likelihoods(emails, "Offer", 0, 0)

# Likelihoods for Money=Yes
p_money_yes_spam = calculate_likelihoods(emails, "Money", 1, 1)
p_money_yes_not_spam = calculate_likelihoods(emails, "Money", 1, 0)

# Step 3: Apply Naive Bayes Formula
def naive_bayes(new_email, p_spam, p_not_spam):
    # Calculate P(Spam | Features)
    p_spam_given_features = (
        p_spam * p_free_yes_spam * p_offer_no_spam * p_money_yes_spam
    )
    
    # Calculate P(Not Spam | Features)
    p_not_spam_given_features = (
        p_not_spam * p_free_yes_not_spam * p_offer_no_not_spam * p_money_yes_not_spam
    )
    
    # Normalize probabilities
    total = p_spam_given_features + p_not_spam_given_features
    p_spam_normalized = p_spam_given_features / total
    p_not_spam_normalized = p_not_spam_given_features / total
    
    return p_spam_normalized, p_not_spam_normalized

p_spam_normalized, p_not_spam_normalized = naive_bayes(new_email, p_spam, p_not_spam)

# Step 4: Make Prediction
if p_spam_normalized > p_not_spam_normalized:
    prediction = "Spam"
else:
    prediction = "Not Spam"

# Output
print(f"Probability of Spam: {p_spam_normalized:.4f}")
print(f"Probability of Not Spam: {p_not_spam_normalized:.4f}")
print(f"Prediction: {prediction}")