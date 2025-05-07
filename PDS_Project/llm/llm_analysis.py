import google.generativeai as genai
import markdown
# Step 1: Configure the API Key
genai.configure(api_key="AIzaSyAtIbq119gVw5snvqDrLYs2URzp5EftpTs")  # Replace with your actual API key

def generate_llm_insight(customer_info, question, prediction):
    print(question)
    print(customer_info)
    if prediction == 0:  # If prediction is 0 (no churn)
        prompt = f"""
        You are a helpful customer retention expert.
        Given this customer data: {customer_info},
        - The model predicts that the customer will NOT churn.
        - Explain why the customer is likely to stay with the service.
        - Highlight 2 reasons why they are satisfied with the service and are unlikely to churn.
        - Provide 1 suggestion to enhance the customer's experience and further strengthen retention.
        - Summarize the customer's profile, and explain what features make them a loyal customer.
        - Be concise and highlight key positive aspects of the customer.
        """
    else:  # If prediction is not 0 (churn)
        prompt = f"""
        You are a helpful customer retention expert.
        Given this customer data: {customer_info},
        - The model predicts that the customer is likely to churn.
        - Give 2 reasons why they might be exiting the service.
        - Provide 1 strong plan to retain them and prevent churn.
        - Summarize the customer's profile and explain what features might be influencing the churn prediction.
        - Be concise and highlight key risk factors that might lead to churn.
        """


    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text
