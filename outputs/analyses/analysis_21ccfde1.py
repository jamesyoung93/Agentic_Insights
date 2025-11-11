# Research Question: What are the main reasons for customer complaints and how do they correlate with customer satisfaction levels?

merged_data = {'complaint_reason': 'reason', 'satisfaction_score': 5}
complaints = merged_data.get('complaint_reason', merged_data.get('complaints'))
satisfaction = merged_data['satisfaction_score']