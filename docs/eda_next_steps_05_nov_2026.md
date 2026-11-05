# EDA Next Steps and Analysis Notes

## Project
EconoCausal

## Contributor
Arsh Murtaza

## Date
05 November 2026

## Purpose
This document records the current analysis direction for the EconoCausal project.

## Dataset Review
- The dataset contains customer-level campaign information.
- Customer recency is an important numerical feature.
- Customer history provides information about previous activity.
- Discount usage is represented in the dataset.
- BOGO usage is also represented.
- Referral status provides additional customer information.
- Marketing channel is an important categorical feature.
- The campaign offer is another important treatment-related feature.
- Conversion is used as the main response variable.

## EDA Review
- Conversion distribution should be checked before modeling.
- Feature distributions should be reviewed for unusual values.
- Numerical features should be examined for scale differences.
- Categorical variables should be inspected for their unique values.
- Group-wise conversion rates can reveal useful patterns.
- Offer-wise conversion should be compared.
- Discount usage can be compared against conversion.
- BOGO usage can be compared against conversion.
- Referral behavior can be compared across customers.
- Marketing channels can be compared using conversion rates.

## Customer Behavior
- Recent customers may respond differently from older customers.
- Historical activity may influence campaign response.
- Previous discount usage may indicate price sensitivity.
- Previous BOGO usage may indicate offer preference.
- Referral customers may have different engagement behavior.
- Different channels may produce different response rates.
- Different offers may influence customer conversion.
- These relationships should be validated statistically.
- Correlation alone should not be interpreted as causation.
- Customer segments should be analyzed carefully.

## Modeling Considerations
- Data preprocessing should remain consistent across experiments.
- Training and testing data should be separated correctly.
- Target leakage should be avoided.
- Categorical variables may require encoding.
- Numerical features should be checked for missing values.
- Model evaluation should use appropriate metrics.
- Accuracy alone may not be sufficient.
- Precision and recall should also be reviewed.
- F1-score can provide an additional evaluation perspective.
- Model results should be compared using the same validation strategy.

## Causal Analysis
- Prediction and causal inference answer different questions.
- Treatment assignment must be considered carefully.
- Treatment and control groups should be identified correctly.
- Matching can help create comparable customer groups.
- Propensity-related information can support matching analysis.
- Treatment effects should be evaluated at the customer level where possible.
- Uplift modeling focuses on incremental treatment response.
- Uplift analysis can support better customer targeting.
- Uplift curves can help evaluate targeting quality.
- Campaign optimization should consider business constraints.

## Next Development Steps
- Continue improving the existing EDA.
- Review feature quality before model training.
- Continue matching analysis.
- Continue uplift model development.
- Compare treatment and control behavior.
- Evaluate model outputs using suitable metrics.
- Review uplift curve implementation.
- Document important findings in the repository.
- Keep project changes on the assigned branch.
- Avoid unnecessary changes to the main branch.

## Status
Current analysis documentation has been updated.

The next focus is to strengthen the modeling workflow and connect the EDA findings with the matching and uplift stages of the project.

This document is intended to maintain a clear record of the current development direction.
